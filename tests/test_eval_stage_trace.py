"""Stage-level evaluation remains independent of external API responses."""

import unittest
from unittest.mock import patch

import pytest

from eval import evaluate as evaluator


class StageTraceTests(unittest.TestCase):
    def test_candidate_coverage_and_final_loss_are_separate(self):
        question = {
            "question": "关联题",
            "expected_evidence": [
                {"source": "A.pdf", "page": 1},
                {"source": "B.pdf", "page": 2},
            ],
        }
        first = {"source": "A.pdf", "page": 1, "score": 0.9}
        second = {"source": "B.pdf", "page": 2, "score": 0.8}

        def fake_retrieve(query, top_k, trace):
            trace.update({
                "vector": [first, second],
                "bm25": [],
                "candidates": [first, second],
                "final": [first],
            })
            return [first]

        with patch.object(evaluator, "retrieve", side_effect=fake_retrieve):
            report = evaluator.evaluate([question], top_k=5, show_trace=True)

        self.assertEqual(report["stages"]["candidates"]["hit_rate"], 1)
        self.assertEqual(report["stages"]["final"]["hit_rate"], 0)
        self.assertEqual(report["evidence_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()


def test_graph_comparison_requires_ready_graph(monkeypatch):
    monkeypatch.setattr(evaluator.database, "DATABASE_URL", "")
    with patch.object(evaluator, "create_milvus_client") as create_client:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            evaluator.require_ready_graph()
    create_client.assert_not_called()

    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(evaluator.database, "DATABASE_URL", "postgres://test")
    with (
        patch.object(evaluator, "create_milvus_client", return_value=client),
        patch.object(evaluator, "get_ready_graph_for_alias", return_value=None),
    ):
        with pytest.raises(RuntimeError, match="没有 ready 图"):
            evaluator.require_ready_graph()
    assert client.closed


def test_graph_stage_is_scored_separately():
    question = {
        "question": "关系题",
        "expected_evidence": [
            {"source": "A.pdf", "page": 1},
            {"source": "B.pdf", "page": 2},
        ],
    }
    first = {"source": "A.pdf", "page": 1, "score": 0.9}
    second = {"source": "B.pdf", "page": 2, "score": 0.0}

    def fake_retrieve(query, top_k, trace):
        trace.update({"candidates": [first], "graph": [second], "final": [first, second]})
        return [first, second]

    with patch.object(evaluator, "retrieve", side_effect=fake_retrieve):
        report = evaluator.evaluate([question], top_k=2, show_trace=True)

    assert report["stages"]["candidates"]["hit_rate"] == 0
    assert report["stages"]["graph"]["evidence_coverage"] == 0.5
    assert report["stages"]["final"]["hit_rate"] == 1
