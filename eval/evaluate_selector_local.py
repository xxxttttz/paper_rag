"""Compare evidence selectors on local BM25 candidates; no model API calls."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from eval.evidence_metrics import score_question, summarize
from eval.evaluate import load_qa_set, print_report
from evidence_selector import select_diverse_evidence, select_requirement_evidence
from milvus_store import create_milvus_client
from retriever import _bm25_search, _load_all_chunks


def evaluate_local(questions: list[dict], top_k: int, candidate_k: int) -> dict:
    client = create_milvus_client()
    try:
        chunks = _load_all_chunks(client)
    finally:
        client.close()

    reports = {"candidates": [], "baseline": [], "page_diverse": [], "requirements": []}
    for qa in questions:
        candidates = _bm25_search(chunks, qa["question"], candidate_k)
        reports["candidates"].append(score_question(qa, candidates))
        reports["baseline"].append(score_question(qa, candidates[:top_k]))
        reports["page_diverse"].append(
            score_question(qa, select_diverse_evidence(candidates, top_k))
        )
        reports["requirements"].append(
            score_question(
                qa, select_requirement_evidence(qa["question"], candidates, top_k)
            )
        )
    return {key: summarize(details) for key, details in reports.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 BM25 候选的证据选择对照，不调用模型 API")
    parser.add_argument("--qa-set", default="eval/relationship_qa_set.json")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--candidate-k", type=int, default=config.RERANK_CANDIDATE_K)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("top-k 必须大于零，candidate-k 不能小于 top-k")
    questions = load_qa_set(args.qa_set)
    reports = evaluate_local(questions, args.top_k, args.candidate_k)
    print(f"{len(questions)} 道题；候选 Top-{args.candidate_k} → 最终 Top-{args.top_k}")
    print_report("候选上限（Top-K 前）", reports["candidates"], show_details=not args.quiet)
    print_report("原始 BM25 顺序", reports["baseline"], show_details=not args.quiet)
    print_report("按页多样性", reports["page_diverse"], show_details=not args.quiet)
    print_report("按证据需求选择", reports["requirements"], show_details=not args.quiet)
    print("注意：这只是本地 BM25 代理评测，不等于在线向量+rerank效果。")


if __name__ == "__main__":
    main()
