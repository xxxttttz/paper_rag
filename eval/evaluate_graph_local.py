"""Compare local BM25 with BM25 plus graph, without any model API calls."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
from eval.evidence_metrics import score_question, summarize
from eval.evaluate import load_qa_set, print_report
from graph_retriever import enrich_with_graph
from graph_store import get_ready_graph
from milvus_store import active_collection_name, create_milvus_client
from retriever import _bm25_search, _load_all_chunks


def evaluate_local(questions: list[dict], top_k: int, graph_slots: int) -> dict:
    if not database.DATABASE_URL:
        raise RuntimeError("图评测需要 PostgreSQL DATABASE_URL")
    client = create_milvus_client()
    try:
        physical = active_collection_name(client, config.COLLECTION_NAME)
        graph = get_ready_graph(database.DATABASE_URL, physical)
        if graph is None:
            raise RuntimeError("当前物理文本索引没有 ready 图")
        chunks = _load_all_chunks(client, collection_name=physical)
        before = []
        after = []
        added = []
        for qa in questions:
            baseline = _bm25_search(chunks, qa["question"], top_k)
            combined, graph_hits = enrich_with_graph(
                client, database.DATABASE_URL, physical, qa["question"],
                baseline, top_k, graph_slots,
            )
            before.append(score_question(qa, baseline))
            after.append(score_question(qa, combined))
            added.append([{"source": hit["source"], "page": hit["page"]} for hit in graph_hits])
    finally:
        client.close()
    return {
        "graph_id": graph["id"],
        "text_chunks": len(chunks),
        "before": summarize(before),
        "after": summarize(after),
        "added": added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="仅本地 BM25+图证据的对照评测")
    parser.add_argument("--qa-set", default="eval/relationship_qa_set.json")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--graph-slots", type=int, default=config.GRAPH_EVIDENCE_K)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 <= args.graph_slots <= args.top_k:
        parser.error("top-k 必须大于零，graph-slots 必须在 0 到 top-k 之间")
    questions = load_qa_set(args.qa_set)
    result = evaluate_local(questions, args.top_k, args.graph_slots)
    print(f"图批次: {result['graph_id']}；文本片段: {result['text_chunks']}")
    print_report("仅本地 BM25", result["before"])
    print_report("本地 BM25 + 图证据", result["after"])
    for index, added in enumerate(result["added"], start=1):
        pages = ", ".join(f"{hit['source']}:{hit['page']}" for hit in added)
        print(f"  题 {index} 新增图证据: {pages or '无'}")
    print("注意：这是无模型调用的离线代理评测，不等同于在线向量+rerank链路。")


if __name__ == "__main__":
    main()
