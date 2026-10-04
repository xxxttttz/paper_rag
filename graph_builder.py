"""Optional, evidence-checked relationship extraction for one text index version."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import config
import graph_store
from postgres_database import connection


logger = logging.getLogger(__name__)
client_openai = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)

ENTITY_TYPES = {
    "paper", "method", "dataset", "task", "metric", "feature",
    "classifier", "device", "network", "system",
}
RELATION_TYPES = {
    "PROPOSES", "USES", "EVALUATED_ON", "REPORTS", "DEPLOYED_ON",
    "OUTPERFORMS", "ADDRESSES", "DERIVES_FROM", "HAS_FEATURE",
    "USES_CLASSIFIER", "REPORTS_METRIC", "HAS_GRANULARITY", "HAS_DATA_SOURCE",
}

EXTRACTION_PROMPT = """Extract only explicit, factual relationships from this paper chunk.
Return one JSON object with a `relations` array. Each relation must have:
`source`: {`type`, `name`}, `type` (uppercase relationship type),
`target`: {`type`, `name`}, and `quote` (an exact, contiguous substring of the chunk
that supports the relationship). Use only entity types: paper, method, dataset,
task, metric, feature, classifier, device, network, system. Use only relation
types: PROPOSES, USES, EVALUATED_ON, REPORTS, DEPLOYED_ON, OUTPERFORMS,
ADDRESSES, DERIVES_FROM, HAS_FEATURE, USES_CLASSIFIER, REPORTS_METRIC,
HAS_GRANULARITY, HAS_DATA_SOURCE. If evidence is ambiguous, omit it. Never infer
cross-paper comparisons. Never invent a quote. Return {"relations": []} if none.
"""


def _entity(value: object) -> tuple[str, str] | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    name = value.get("name")
    if not isinstance(kind, str) or kind not in ENTITY_TYPES or not isinstance(name, str):
        return None
    name = " ".join(name.split())
    if not 2 <= len(name) <= 200:
        return None
    return kind, name


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def parse_relations(payload: object, chunk_text: str) -> list[dict]:
    """Reject unsupported types and any quote absent from the source chunk."""
    if not isinstance(payload, dict) or not isinstance(payload.get("relations"), list):
        raise ValueError("Graph extraction must return a relations array")
    parsed = []
    for relation in payload["relations"]:
        if not isinstance(relation, dict):
            continue
        source = _entity(relation.get("source"))
        target = _entity(relation.get("target"))
        relation_type = relation.get("type")
        quote = relation.get("quote")
        if (
            not source or not target or not isinstance(relation_type, str)
            or relation_type not in RELATION_TYPES
            or not isinstance(quote, str)
        ):
            continue
        quote = quote.strip()
        if not 8 <= len(quote) <= 1500 or quote not in chunk_text:
            continue
        parsed.append({
            "source": source,
            "target": target,
            "type": relation_type,
            "quote": quote,
        })
    return parsed


def extract_relations(record: dict) -> list[dict]:
    response = client_openai.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Paper: {record['source']}\nChunk:\n{record['text']}"},
        ],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    return parse_relations(json.loads(content), record["text"])


def _write_chunk(database_url: str, build_id: str, record: dict, relations: list[dict]) -> None:
    with connection(database_url) as conn:
        for relation in relations:
            endpoint_ids = []
            for kind, name in (relation["source"], relation["target"]):
                node_id = _stable_id(kind, name.casefold())
                endpoint_ids.append(node_id)
                conn.execute(
                    """INSERT INTO graph_nodes
                       (build_id, id, entity_type, canonical_name)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (build_id, id) DO NOTHING""",
                    (build_id, node_id, kind, name),
                )
                conn.execute(
                    """INSERT INTO graph_mentions
                       (build_id, node_id, chunk_id, source, page)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT DO NOTHING""",
                    (build_id, node_id, record["id"], record["source"], record["page"]),
                )
            edge_id = _stable_id(endpoint_ids[0], relation["type"], endpoint_ids[1])
            conn.execute(
                """INSERT INTO graph_edges
                   (build_id, id, source_node_id, relation_type, target_node_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (build_id, edge_id, endpoint_ids[0], relation["type"], endpoint_ids[1]),
            )
            conn.execute(
                """INSERT INTO graph_edge_evidence
                   (build_id, edge_id, chunk_id, source, page, quote)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (
                    build_id, edge_id, record["id"], record["source"],
                    record["page"], relation["quote"],
                ),
            )


def build_graph_index(
    records: list[dict],
    text_collection: str,
    database_url: str,
    progress: Callable[[int, int], None] | None = None,
    extraction_workers: int = 1,
) -> dict:
    """Build a private graph; publish it only after every chunk succeeds."""
    if not records or not database_url:
        raise ValueError("Graph build requires text records and PostgreSQL")
    if not 1 <= extraction_workers <= 4:
        raise ValueError("extraction_workers must be between 1 and 4")
    build_id = graph_store.begin_graph_build(database_url, text_collection)
    try:
        with ThreadPoolExecutor(max_workers=extraction_workers) as executor:
            for start in range(0, len(records), extraction_workers):
                batch = records[start:start + extraction_workers]
                # A batch completes before the next is submitted: a malformed
                # response cannot trigger calls for every remaining chunk.
                extracted = list(executor.map(extract_relations, batch))
                for offset, (record, relations) in enumerate(zip(batch, extracted), start=1):
                    _write_chunk(database_url, build_id, record, relations)
                    if progress is not None:
                        progress(start + offset, len(records))
        with connection(database_url) as conn:
            nodes = conn.execute(
                "SELECT count(*) AS count FROM graph_nodes WHERE build_id = %s", (build_id,)
            ).fetchone()["count"]
            edges = conn.execute(
                "SELECT count(*) AS count FROM graph_edges WHERE build_id = %s", (build_id,)
            ).fetchone()["count"]
        if edges == 0:
            raise RuntimeError("Graph extraction produced no supported relationships")
        graph_store.mark_graph_ready(database_url, build_id, nodes, edges)
        return {"build_id": build_id, "nodes": nodes, "edges": edges}
    except Exception as error:
        try:
            graph_store.mark_graph_failed(database_url, build_id, str(error))
        except Exception:
            logger.exception("Could not mark graph build %s failed", build_id)
        raise
