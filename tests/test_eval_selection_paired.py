"""Paired selector evaluation uses one retrieval and one ranking per question."""

from unittest.mock import patch

import pytest

from eval import evaluate as evaluator


QUESTION = {
    "question": "Alpha 和 Beta 分别做了什么？",
    "expected_evidence": [
        {"source": "Alpha.pdf", "page": 1},
        {"source": "Beta.pdf", "page": 2},
    ],
}
RERANKED = [
    {"source": "Alpha.pdf", "page": 1, "text": "secret first", "score": 0.9},
    {"source": "Alpha.pdf", "page": 1, "text": "secret duplicate", "score": 0.8},
    {"source": "Beta.pdf", "page": 2, "text": "secret second", "score": 0.7},
]


def test_paired_comparison_retrieves_once_per_question_and_restores_flags():
    def fake_retrieve(query, top_k, trace, reranked_output):
        assert evaluator.config.ENABLE_RERANK
        assert not evaluator.config.ENABLE_GRAPH_RETRIEVAL
        assert not evaluator.config.ENABLE_DIVERSE_RERANK
        assert not evaluator.config.ENABLE_REQUIREMENT_SELECTION
        reranked_output.extend(RERANKED)
        trace["candidates"] = [
            {"source": hit["source"], "page": hit["page"], "score": hit["score"]}
            for hit in RERANKED
        ]
        trace["reranked"] = trace["candidates"]
        trace["final"] = trace["candidates"][:top_k]
        return RERANKED[:top_k]

    with (
        patch.object(evaluator.config, "ENABLE_RERANK", False),
        patch.object(evaluator.config, "ENABLE_GRAPH_RETRIEVAL", True),
        patch.object(evaluator.config, "ENABLE_DIVERSE_RERANK", True),
        patch.object(evaluator.config, "ENABLE_REQUIREMENT_SELECTION", True),
        patch.object(evaluator, "retrieve", side_effect=fake_retrieve) as retrieve,
    ):
        reports = evaluator.evaluate_selection([QUESTION, QUESTION], 2, show_trace=True)
        assert evaluator.config.ENABLE_RERANK is False
        assert evaluator.config.ENABLE_GRAPH_RETRIEVAL is True
        assert evaluator.config.ENABLE_DIVERSE_RERANK is True
        assert evaluator.config.ENABLE_REQUIREMENT_SELECTION is True

    assert retrieve.call_count == 2
    assert reports["candidates"]["hit_rate"] == 1.0
    assert reports["baseline"]["hit_rate"] == 0.0
    assert reports["page_diverse"]["hit_rate"] == 1.0
    assert reports["requirements"]["hit_rate"] == 1.0
    trace = reports["baseline"]["details"][0]["trace"]
    assert all("text" not in hit for hits in trace.values() for hit in hits)


def test_partial_rerank_aborts_comparison_and_restores_flags():
    def fake_retrieve(query, top_k, trace, reranked_output):
        trace["candidates"] = [{"source": "Alpha.pdf", "page": 1}]
        return []

    with (
        patch.object(evaluator.config, "ENABLE_RERANK", False),
        patch.object(evaluator.config, "ENABLE_GRAPH_RETRIEVAL", True),
        patch.object(evaluator, "retrieve", side_effect=fake_retrieve),
    ):
        with pytest.raises(RuntimeError, match="完整 rerank 结果不齐"):
            evaluator.evaluate_selection([QUESTION], 2)
        assert evaluator.config.ENABLE_RERANK is False
        assert evaluator.config.ENABLE_GRAPH_RETRIEVAL is True
