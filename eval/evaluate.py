"""
eval/evaluate.py
-----------------
轻量级检索质量评测脚本。

设计思路：
- 标准答案使用"来源文件 + 页码"，避免重建索引后 chunk ID 改变导致评测集失效。
- 兼容旧的单页 expected_source/expected_page，以及多页 expected_evidence。
- Hit Rate@K 要求一道题的全部标注证据页都进入 Top-K；证据覆盖率允许部分命中。
- MRR 按找齐全部证据时的最后一个排名计算；单页题与原定义相同。

用法：
    python eval/evaluate.py                    # 用qa_set.json，按当前config.py的设置跑一遍
    python eval/evaluate.py --compare-rerank    # 同一份问题，分别跑"开Reranker"和"关Reranker"，对比效果
    python eval/evaluate.py --compare-hybrid    # 对比"混合检索"和"纯向量检索"
    python eval/evaluate.py --qa-set eval/my_qa_set.json --top-k 3 --page-tolerance 1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
from eval.evidence_metrics import score_question, summarize, validate_question
from evidence_selector import select_diverse_evidence, select_requirement_evidence
from graph_store import get_ready_graph_for_alias
from milvus_store import create_milvus_client
from retriever import retrieve

EVAL_DIR = os.path.dirname(__file__)
DEFAULT_QA_SET_PATH = next(
    (
        os.path.join(EVAL_DIR, filename)
        for filename in ("qa_set.json", "my_qa_set.json")
        if os.path.exists(os.path.join(EVAL_DIR, filename))
    ),
    os.path.join(EVAL_DIR, "qa_set.json"),
)


def load_qa_set(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到评测集文件: {path}\n"
            f"请复制 eval/qa_set.example.json 为 eval/qa_set.json，"
            f"并按照里面的格式说明填入你自己的问题。"
        )
    with open(path, "r", encoding="utf-8") as f:
        qa_set = json.load(f)
    # 过滤掉模板里那条纯说明用的占位条目（没有question字段的）
    qa_set = [qa for qa in qa_set if "question" in qa]
    for index, qa in enumerate(qa_set, start=1):
        validate_question(qa, index)
    return qa_set


def evaluate(
    qa_set: list[dict], top_k: int, page_tolerance: int = 0, show_trace: bool = False
) -> dict:
    """
    对每道题跑一次retrieve()，返回：
    {
        "hit_rate": float, "mrr": float, "evidence_coverage": float,
        "details": [{"question", "expected", "hit", "rank", "evidence_coverage"}, ...]
    }
    """
    details = []
    for qa in qa_set:
        trace = {} if show_trace else None
        results = (
            retrieve(qa["question"], top_k=top_k, trace=trace)
            if show_trace
            else retrieve(qa["question"], top_k=top_k)
        )
        detail = score_question(qa, results, page_tolerance)
        if trace is not None:
            detail["trace"] = trace
        details.append(detail)
    report = summarize(details)
    if show_trace:
        report["stages"] = {}
        for stage in ("vector", "bm25", "candidates", "reranked", "graph", "final"):
            if not any(stage in detail["trace"] for detail in details):
                continue
            stage_details = [
                score_question(qa, detail["trace"].get(stage, []), page_tolerance)
                for qa, detail in zip(qa_set, details)
            ]
            report["stages"][stage] = summarize(stage_details)
    return report


def evaluate_selection(
    qa_set: list[dict], top_k: int, page_tolerance: int = 0, show_trace: bool = False
) -> dict[str, dict]:
    """Compare three selectors on exactly one complete rerank per question.

    The normal retrieval pipeline supplies the same candidates to all selectors.
    Graph enrichment and runtime selection are disabled only during this paired
    evaluation; all configuration flags are restored even if a request fails.
    """
    reports = {key: [] for key in ("candidates", "reranked", "baseline", "page_diverse", "requirements")}
    flags = (
        config.ENABLE_RERANK,
        config.ENABLE_GRAPH_RETRIEVAL,
        config.ENABLE_DIVERSE_RERANK,
        config.ENABLE_REQUIREMENT_SELECTION,
    )
    try:
        config.ENABLE_RERANK = True
        config.ENABLE_GRAPH_RETRIEVAL = False
        config.ENABLE_DIVERSE_RERANK = False
        config.ENABLE_REQUIREMENT_SELECTION = False

        for qa in qa_set:
            trace = {}
            reranked = []
            retrieve(qa["question"], top_k=top_k, trace=trace, reranked_output=reranked)
            candidates = trace.get("candidates", [])
            if len(reranked) != len(candidates):
                raise RuntimeError(
                    f"完整 rerank 结果不齐：候选 {len(candidates)} 条，返回 {len(reranked)} 条；"
                    "已停止对照，避免把部分结果当成完整排名"
                )

            baseline = reranked[:top_k]
            diverse = select_diverse_evidence(reranked, top_k)
            required = select_requirement_evidence(qa["question"], reranked, top_k)
            for key, hits in (
                ("candidates", candidates),
                ("reranked", reranked),
                ("baseline", baseline),
                ("page_diverse", diverse),
                ("requirements", required),
            ):
                detail = score_question(qa, hits, page_tolerance)
                if show_trace and key == "baseline":
                    detail["trace"] = {
                        "candidates": candidates,
                        "reranked": trace.get("reranked", []),
                        "original": trace.get("final", []),
                        "page_diverse": [
                            {"source": hit["source"], "page": hit["page"]}
                            for hit in diverse
                        ],
                        "requirements": [
                            {"source": hit["source"], "page": hit["page"]}
                            for hit in required
                        ],
                    }
                reports[key].append(detail)
    finally:
        (
            config.ENABLE_RERANK,
            config.ENABLE_GRAPH_RETRIEVAL,
            config.ENABLE_DIVERSE_RERANK,
            config.ENABLE_REQUIREMENT_SELECTION,
        ) = flags
    return {key: summarize(details) for key, details in reports.items()}


def print_report(name: str, result: dict, show_details: bool = True):
    print(f"\n=== {name} ===")
    print(
        f"Hit Rate（证据找齐）: {result['hit_rate']:.1%}   "
        f"证据覆盖率: {result['evidence_coverage']:.1%}   MRR: {result['mrr']:.3f}"
    )
    for stage, stage_report in result.get("stages", {}).items():
        print(
            f"  {stage}: 找齐 {stage_report['hit_rate']:.1%}，"
            f"证据覆盖 {stage_report['evidence_coverage']:.1%}"
        )
    if show_details:
        for d in result["details"]:
            mark = "✅" if d["hit"] else "❌"
            rank_info = (
                f"第{d['rank']}位找齐" if d["hit"]
                else f"只找到 {d['evidence_coverage']:.0%} 证据"
            )
            print(f"  {mark} [{rank_info}] {d['question']}  (期望: {d['expected']})")
            if "trace" in d:
                for stage, hits in d["trace"].items():
                    pages = ", ".join(
                        f"{os.path.basename(hit['source'])}:{hit['page']}"
                        for hit in hits
                    )
                    print(f"    {stage}: {pages or '无'}")


def require_ready_graph() -> dict:
    """Reject graph comparisons that would silently run against no graph."""
    if not database.DATABASE_URL:
        raise RuntimeError("图对比需要 PostgreSQL DATABASE_URL")
    client = create_milvus_client()
    try:
        graph = get_ready_graph_for_alias(
            database.DATABASE_URL, client, config.COLLECTION_NAME
        )
    finally:
        client.close()
    if graph is None:
        raise RuntimeError("当前文本索引没有 ready 图；请先启用图构建并重建索引")
    return graph


def main():
    parser = argparse.ArgumentParser(description="RAG检索质量评测")
    parser.add_argument("--qa-set", default=DEFAULT_QA_SET_PATH, help="评测集json路径")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="检索TopK")
    parser.add_argument("--page-tolerance", type=int, default=0,
                         help="页码允许的误差范围（比如答案横跨相邻两页时设为1）")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-rerank", action="store_true", help="对比开/关Reranker的效果")
    comparison.add_argument("--compare-hybrid", action="store_true", help="对比混合检索/纯向量检索的效果")
    comparison.add_argument("--compare-graph", action="store_true", help="对比关闭/开启图证据检索")
    comparison.add_argument("--compare-selection", action="store_true", help="每题一次完整 rerank，对比原始、页码多样性和证据需求选择")
    parser.add_argument("--quiet", action="store_true", help="只看汇总数字，不看每题详情")
    parser.add_argument(
        "--show-trace", action="store_true",
        help="显示向量、BM25、融合及最终结果的来源和页码；不显示原文",
    )
    parser.add_argument(
        "--diverse-rerank", action="store_true",
        help="实验性证据多样性选择；会要求 rerank 返回全部候选排名",
    )
    args = parser.parse_args()

    if args.diverse_rerank:
        config.ENABLE_DIVERSE_RERANK = True

    qa_set = load_qa_set(args.qa_set)
    print(f"加载了 {len(qa_set)} 道评测题，Top-K={args.top_k}，页码容差={args.page_tolerance}")

    if args.compare_rerank:
        original = config.ENABLE_RERANK
        try:
            config.ENABLE_RERANK = False
            r_off = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
            config.ENABLE_RERANK = True
            r_on = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
        finally:
            config.ENABLE_RERANK = original
        print_report("关闭 Reranker", r_off, show_details=not args.quiet)
        print_report("开启 Reranker", r_on, show_details=not args.quiet)
        print(f"\n>>> Hit Rate 变化: {r_off['hit_rate']:.1%} -> {r_on['hit_rate']:.1%}"
              f"   MRR 变化: {r_off['mrr']:.3f} -> {r_on['mrr']:.3f}")
        return

    if args.compare_hybrid:
        original = config.USE_HYBRID_SEARCH
        try:
            config.USE_HYBRID_SEARCH = False
            r_off = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
            config.USE_HYBRID_SEARCH = True
            r_on = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
        finally:
            config.USE_HYBRID_SEARCH = original
        print_report("纯向量检索", r_off, show_details=not args.quiet)
        print_report("混合检索(向量+BM25)", r_on, show_details=not args.quiet)
        print(f"\n>>> Hit Rate 变化: {r_off['hit_rate']:.1%} -> {r_on['hit_rate']:.1%}"
              f"   MRR 变化: {r_off['mrr']:.3f} -> {r_on['mrr']:.3f}")
        return

    if args.compare_graph:
        graph = require_ready_graph()
        print(f"使用图构建批次 {graph['id']}；两组检索均会调用现有 embedding/rerank API")
        original = config.ENABLE_GRAPH_RETRIEVAL
        try:
            config.ENABLE_GRAPH_RETRIEVAL = False
            r_off = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
            config.ENABLE_GRAPH_RETRIEVAL = True
            r_on = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
        finally:
            config.ENABLE_GRAPH_RETRIEVAL = original
        print_report("关闭图检索", r_off, show_details=not args.quiet)
        print_report("开启图检索", r_on, show_details=not args.quiet)
        print(
            f"\n>>> 找齐率: {r_off['hit_rate']:.1%} -> {r_on['hit_rate']:.1%}"
            f"   证据覆盖率: {r_off['evidence_coverage']:.1%} -> "
            f"{r_on['evidence_coverage']:.1%}"
        )
        return

    if args.compare_selection:
        reports = evaluate_selection(qa_set, args.top_k, args.page_tolerance, args.show_trace)
        print_report("融合候选上限", reports["candidates"], show_details=False)
        print_report("完整 rerank 上限", reports["reranked"], show_details=False)
        print_report("原始最终 Top-K", reports["baseline"], show_details=not args.quiet)
        print_report("按页多样性", reports["page_diverse"], show_details=not args.quiet)
        print_report("按证据需求选择", reports["requirements"], show_details=not args.quiet)
        print(
            f"\n>>> 找齐率: 原始 {reports['baseline']['hit_rate']:.1%} / "
            f"多样性 {reports['page_diverse']['hit_rate']:.1%} / "
            f"需求选择 {reports['requirements']['hit_rate']:.1%}"
        )
        return

    result = evaluate(qa_set, args.top_k, args.page_tolerance, args.show_trace)
    print_report("当前配置", result, show_details=not args.quiet)


if __name__ == "__main__":
    main()
