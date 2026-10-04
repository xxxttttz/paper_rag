"""Evaluate only local BM25 recall; no embedding, rerank or generation calls."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evidence_metrics import expected_evidence, score_question, summarize
from eval.evaluate import load_qa_set, print_report
from milvus_store import create_milvus_client
from retriever import _bm25_search, _load_all_chunks


def main():
    parser = argparse.ArgumentParser(description="只使用本地 BM25 的检索诊断")
    parser.add_argument("--qa-set", default="eval/relationship_qa_set.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--page-tolerance", type=int, default=0)
    args = parser.parse_args()

    if args.top_k < 1 or args.page_tolerance < 0:
        parser.error("top-k 必须大于零，page-tolerance 不能为负")

    questions = load_qa_set(args.qa_set)
    client = create_milvus_client()
    try:
        chunks = _load_all_chunks(client)
    finally:
        client.close()

    indexed_pages = {(os.path.basename(c["source"]), c["page"]) for c in chunks}
    missing_pages = {
        (os.path.basename(item["source"]), item["page"])
        for qa in questions
        for item in expected_evidence(qa)
        if (os.path.basename(item["source"]), item["page"]) not in indexed_pages
    }
    details = []
    rank_details = []
    for qa in questions:
        hits = _bm25_search(chunks, qa["question"], args.top_k)
        details.append(score_question(qa, hits, args.page_tolerance))
        ranks = []
        for item in expected_evidence(qa):
            rank = next(
                (
                    index
                    for index, hit in enumerate(hits, start=1)
                    if os.path.basename(hit["source"]) == os.path.basename(item["source"])
                    and abs(hit["page"] - item["page"]) <= args.page_tolerance
                ),
                None,
            )
            ranks.append(rank)
        rank_details.append(ranks)
    print(f"加载 {len(chunks)} 个文本片段、{len(questions)} 道题；Top-K={args.top_k}")
    if missing_pages:
        print("以下标注页不在当前文本索引中:", sorted(missing_pages))
    else:
        print("所有标注页均存在于当前文本索引中")
    print_report("仅本地 BM25（不调用模型 API）", summarize(details))
    for index, ranks in enumerate(rank_details, start=1):
        print(f"  题 {index} 各证据的 BM25 排名: {ranks}")


if __name__ == "__main__":
    main()
