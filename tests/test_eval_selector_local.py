"""The local selector comparison must score the candidate ceiling separately."""

import unittest
from unittest.mock import MagicMock, patch

from eval.evaluate_selector_local import evaluate_local


class LocalSelectorEvaluationTests(unittest.TestCase):
    @patch("eval.evaluate_selector_local._bm25_search")
    @patch("eval.evaluate_selector_local._load_all_chunks", return_value=[])
    @patch("eval.evaluate_selector_local.create_milvus_client")
    def test_reports_candidate_ceiling_and_selection(
        self, create_client, _load_chunks, bm25_search
    ):
        create_client.return_value = MagicMock()
        bm25_search.return_value = [
            {"source": "Alpha.pdf", "page": 1, "text": "overview", "score": 4},
            {"source": "Alpha.pdf", "page": 1, "text": "more overview", "score": 3},
            {"source": "Beta.pdf", "page": 2, "text": "classification", "score": 2},
        ]
        questions = [{
            "question": "Alpha 和 Beta 分别做了什么？",
            "expected_evidence": [
                {"source": "Alpha.pdf", "page": 1},
                {"source": "Beta.pdf", "page": 2},
            ],
        }]

        reports = evaluate_local(questions, top_k=2, candidate_k=3)

        self.assertEqual(reports["candidates"]["hit_rate"], 1.0)
        self.assertEqual(reports["baseline"]["hit_rate"], 0.0)
        self.assertEqual(reports["requirements"]["hit_rate"], 1.0)
        create_client.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
