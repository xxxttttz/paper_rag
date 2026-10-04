"""Resolve ready graph relationships back to verified Milvus text chunks."""

from __future__ import annotations

import json
import os
from collections import Counter

import graph_store
from search_tokenizer import tokenize_search_text


RELATION_HINTS = (
    (("分类器", "classifier"), {"USES_CLASSIFIER"}),
    (("特征", "feature"), {"HAS_FEATURE"}),
    (("指标", "准确率", "精度", "召回", "延迟", "accuracy", "latency"),
     {"REPORTS_METRIC", "REPORTS"}),
    (("场景", "网络", "交换机", "部署", "isp", "network"), {"DEPLOYED_ON"}),
    (("数据", "dataset"), {"EVALUATED_ON", "HAS_DATA_SOURCE"}),
)


def _hinted_relations(question: str) -> set[str]:
    lowered = question.casefold()
    return {
        relation
        for keywords, relations in RELATION_HINTS
        if any(keyword in lowered for keyword in keywords)
        for relation in relations
    }


def _reference_score(ref: dict, question: str, hinted: set[str], tokens: set[str]) -> float:
    lowered = question.casefold()
    names = (ref.get("source_name", ""), ref.get("target_name", ""))
    named = sum(len(name) >= 4 and name.casefold() in lowered for name in names)
    quote_tokens = set(tokenize_search_text(ref.get("quote", "")))
    lexical = min(len(tokens & quote_tokens), 5)
    return (
        4 * (ref.get("relation_type") in hinted)
        + 1.5 * named
        + 0.3 * lexical
        + float(ref.get("confidence") or 0)
    )


def enrich_with_graph(
    milvus_client,
    database_url: str,
    text_collection: str,
    question: str,
    baseline: list[dict],
    top_k: int,
    graph_slots: int,
) -> tuple[list[dict], list[dict]]:
    """Reserve a few answer slots for distinct, source-backed graph evidence.

    Graph metadata is never used directly as an answer citation. We only return
    a hit after its chunk ID, source, and page agree with the active text index.
    """
    if not database_url or top_k <= 0 or graph_slots <= 0:
        return baseline, []

    anchors = [str(hit["id"]) for hit in baseline if hit.get("id") is not None]
    refs = graph_store.find_related_evidence(
        database_url, text_collection, question, anchors, limit=250
    )
    if not refs:
        return baseline, []

    requested_sources = {
        ref["source"] for ref in refs
        if os.path.splitext(os.path.basename(ref["source"]))[0].casefold()
        in question.casefold()
    }
    if requested_sources:
        refs = [ref for ref in refs if ref["source"] in requested_sources]

    ids = list(dict.fromkeys(str(ref["chunk_id"]) for ref in refs))
    milvus_client.load_collection(text_collection)
    rows = milvus_client.query(
        collection_name=text_collection,
        filter=f"id in {json.dumps(ids)}",
        output_fields=["id", "text", "source", "page"],
        limit=len(ids),
    )
    by_id = {str(row["id"]): row for row in rows}
    used_pages = {(hit["source"], hit["page"]) for hit in baseline}
    source_counts = Counter(hit["source"] for hit in baseline)
    hinted = _hinted_relations(question)
    tokens = set(tokenize_search_text(question))
    options = {}
    for ref in refs:
        row = by_id.get(str(ref["chunk_id"]))
        if not row or not row.get("text"):
            continue
        page = (row["source"], row["page"])
        if page != (ref["source"], ref["page"]) or page in used_pages:
            continue
        score = _reference_score(ref, question, hinted, tokens)
        previous = options.get(str(ref["chunk_id"]))
        if previous is None or score > previous[0]:
            options[str(ref["chunk_id"])] = (score, row)

    graph_hits = []
    while options and len(graph_hits) < min(graph_slots, top_k):
        best_id = max(
            options,
            key=lambda chunk_id: (
                options[chunk_id][0] - 1.5 * source_counts[options[chunk_id][1]["source"]],
                -source_counts[options[chunk_id][1]["source"]],
                chunk_id,
            ),
        )
        _, row = options.pop(best_id)
        page = (row["source"], row["page"])
        if page in used_pages:
            continue
        graph_hits.append({
            "id": row["id"],
            "text": row["text"],
            "source": row["source"],
            "page": row["page"],
            "score": None,  # Graph evidence has no score comparable to rerank scores.
        })
        used_pages.add(page)
        source_counts[row["source"]] += 1

    if not graph_hits:
        return baseline, []
    return baseline[: top_k - len(graph_hits)] + graph_hits, graph_hits
