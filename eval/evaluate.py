"""
eval/evaluate.py
-----------------
轻量级检索质量评测脚本。

设计思路：
- 标准答案用"来源文件 + 页码"而不是具体某个chunk——因为chunk是每次重建索引时
  自动切分生成的，改一次CHUNK_SIZE或分块策略，所有chunk都会变，但"答案在哪一页"
  这件事是不变的，评测集不会因此失效。
- 指标：
    Hit Rate@K —— TopK个检索结果里，只要有一个命中标准答案的(来源,页码)，就算命中
    MRR（平均倒数排名）—— 命中的排名越靠前分数越高（第1位=1分，第2位=0.5分，
                          第3位=0.33分…），能反映"检索到了但排得靠不靠前"

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
    required_fields = {"question", "expected_source", "expected_page"}
    for index, qa in enumerate(qa_set, start=1):
        missing = required_fields - qa.keys()
        if missing:
            raise ValueError(f"评测集第 {index} 条缺少字段: {sorted(missing)}")
        if not isinstance(qa["expected_page"], int) or qa["expected_page"] < 1:
            raise ValueError(f"评测集第 {index} 条 expected_page 必须是正整数")
    return qa_set


def _is_hit(chunk: dict, expected_source: str, expected_page: int, page_tolerance: int) -> bool:
    return (
        os.path.basename(chunk["source"]) == os.path.basename(expected_source)
        and abs(chunk["page"] - expected_page) <= page_tolerance
    )


def evaluate(qa_set: list[dict], top_k: int, page_tolerance: int = 0) -> dict:
    """
    对每道题跑一次retrieve()，返回：
    {
        "hit_rate": float,
        "mrr": float,
        "details": [{"question", "expected", "hit", "rank"}, ...]
    }
    """
    details = []
    for qa in qa_set:
        results = retrieve(qa["question"], top_k=top_k)
        rank = None
        for i, r in enumerate(results, start=1):
            if _is_hit(r, qa["expected_source"], qa["expected_page"], page_tolerance):
                rank = i
                break
        details.append({
            "question": qa["question"],
            "expected": f"{qa['expected_source']} 第{qa['expected_page']}页",
            "hit": rank is not None,
            "rank": rank,
        })

    hits = [d for d in details if d["hit"]]
    n = len(details)
    hit_rate = len(hits) / n if n else 0.0
    mrr = sum(1 / d["rank"] for d in hits) / n if n else 0.0
    return {"hit_rate": hit_rate, "mrr": mrr, "details": details}


def print_report(name: str, result: dict, show_details: bool = True):
    print(f"\n=== {name} ===")
    print(f"Hit Rate: {result['hit_rate']:.1%}   MRR: {result['mrr']:.3f}")
    if show_details:
        for d in result["details"]:
            mark = "✅" if d["hit"] else "❌"
            rank_info = f"第{d['rank']}位命中" if d["hit"] else "未命中"
            print(f"  {mark} [{rank_info}] {d['question']}  (期望: {d['expected']})")


def main():
    parser = argparse.ArgumentParser(description="RAG检索质量评测")
    parser.add_argument("--qa-set", default=DEFAULT_QA_SET_PATH, help="评测集json路径")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="检索TopK")
    parser.add_argument("--page-tolerance", type=int, default=0,
                         help="页码允许的误差范围（比如答案横跨相邻两页时设为1）")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-rerank", action="store_true", help="对比开/关Reranker的效果")
    comparison.add_argument("--compare-hybrid", action="store_true", help="对比混合检索/纯向量检索的效果")
    parser.add_argument("--quiet", action="store_true", help="只看汇总数字，不看每题详情")
    args = parser.parse_args()

    qa_set = load_qa_set(args.qa_set)
    print(f"加载了 {len(qa_set)} 道评测题，Top-K={args.top_k}，页码容差={args.page_tolerance}")

    if args.compare_rerank:
        original = config.ENABLE_RERANK
        try:
            config.ENABLE_RERANK = False
            r_off = evaluate(qa_set, args.top_k, args.page_tolerance)
            config.ENABLE_RERANK = True
            r_on = evaluate(qa_set, args.top_k, args.page_tolerance)
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
            r_off = evaluate(qa_set, args.top_k, args.page_tolerance)
            config.USE_HYBRID_SEARCH = True
            r_on = evaluate(qa_set, args.top_k, args.page_tolerance)
        finally:
            config.USE_HYBRID_SEARCH = original
        print_report("纯向量检索", r_off, show_details=not args.quiet)
        print_report("混合检索(向量+BM25)", r_on, show_details=not args.quiet)
        print(f"\n>>> Hit Rate 变化: {r_off['hit_rate']:.1%} -> {r_on['hit_rate']:.1%}"
              f"   MRR 变化: {r_off['mrr']:.3f} -> {r_on['mrr']:.3f}")
        return

    result = evaluate(qa_set, args.top_k, args.page_tolerance)
    print_report("当前配置", result, show_details=not args.quiet)


if __name__ == "__main__":
    main()
