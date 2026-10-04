"""Pure scoring helpers shared by single-page and multi-evidence retrieval evaluations."""

import os


def expected_evidence(qa: dict) -> list[dict]:
    """Return the evidence list, including for legacy single-page questions."""
    if "expected_evidence" in qa:
        return qa["expected_evidence"]
    return [{"source": qa["expected_source"], "page": qa["expected_page"]}]


def validate_question(qa: dict, index: int) -> None:
    if not isinstance(qa.get("question"), str) or not qa["question"].strip():
        raise ValueError(f"评测集第 {index} 条 question 必须是非空字符串")
    has_evidence = "expected_evidence" in qa
    has_legacy = "expected_source" in qa or "expected_page" in qa
    if has_evidence and has_legacy:
        raise ValueError(f"评测集第 {index} 条不能同时使用两种证据格式")
    if has_evidence:
        evidence = qa["expected_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"评测集第 {index} 条 expected_evidence 必须是非空列表")
    else:
        missing = {"expected_source", "expected_page"} - qa.keys()
        if missing:
            raise ValueError(f"评测集第 {index} 条缺少字段: {sorted(missing)}")
        evidence = expected_evidence(qa)
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"评测集第 {index} 条证据必须是对象")
        source, page = item.get("source"), item.get("page")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"评测集第 {index} 条证据 source 必须是非空字符串")
        if type(page) is not int or page < 1:
            raise ValueError(f"评测集第 {index} 条证据 page 必须是正整数")
        key = (os.path.basename(source), page)
        if key in seen:
            raise ValueError(f"评测集第 {index} 条重复证据: {source} 第{page}页")
        seen.add(key)


def _matched_count(evidence: list[dict], results: list[dict], page_tolerance: int) -> int:
    """Maximum one-to-one matches, so a nearby page cannot satisfy two facts."""
    owners = {}

    def assign(evidence_index: int, seen: set[int]) -> bool:
        item = evidence[evidence_index]
        for result_index, chunk in enumerate(results):
            if result_index in seen:
                continue
            if (
                os.path.basename(chunk["source"]) != os.path.basename(item["source"])
                or abs(chunk["page"] - item["page"]) > page_tolerance
            ):
                continue
            seen.add(result_index)
            if result_index not in owners or assign(owners[result_index], seen):
                owners[result_index] = evidence_index
                return True
        return False

    for evidence_index in range(len(evidence)):
        assign(evidence_index, set())
    return len(owners)


def score_question(qa: dict, results: list[dict], page_tolerance: int = 0) -> dict:
    """A question hits only when distinct results cover every required page."""
    evidence = expected_evidence(qa)
    matched = _matched_count(evidence, results, page_tolerance)
    complete = matched == len(evidence)
    completion_rank = next(
        (
            rank
            for rank in range(1, len(results) + 1)
            if _matched_count(evidence, results[:rank], page_tolerance) == len(evidence)
        ),
        None,
    ) if complete else None
    return {
        "question": qa["question"],
        "expected": "、".join(f"{item['source']} 第{item['page']}页" for item in evidence),
        "hit": complete,
        "rank": completion_rank,
        "evidence_coverage": matched / len(evidence),
    }


def summarize(details: list[dict]) -> dict:
    n = len(details)
    hits = [item for item in details if item["hit"]]
    return {
        "hit_rate": len(hits) / n if n else 0.0,
        "mrr": sum(1 / item["rank"] for item in hits) / n if n else 0.0,
        "evidence_coverage": sum(item["evidence_coverage"] for item in details) / n if n else 0.0,
        "details": details,
    }
