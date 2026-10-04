"""Tests for retrieval metrics that do not require Milvus or API keys."""

import json
import unittest
from pathlib import Path

from eval.evidence_metrics import score_question, summarize, validate_question


class EvidenceMetricsTests(unittest.TestCase):
    def test_relationship_questions_are_well_formed(self):
        path = Path(__file__).resolve().parents[1] / "eval" / "relationship_qa_set.json"
        questions = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(questions), 5)
        for index, question in enumerate(questions, start=1):
            validate_question(question, index)
            self.assertGreaterEqual(len(question["expected_evidence"]), 2)

    def test_legacy_single_page_question(self):
        question = {"question": "例题", "expected_source": "A.pdf", "expected_page": 2}
        validate_question(question, 1)
        detail = score_question(question, [{"source": "/tmp/A.pdf", "page": 2}])
        self.assertTrue(detail["hit"])
        self.assertEqual(detail["rank"], 1)
        self.assertEqual(detail["evidence_coverage"], 1)

    def test_all_evidence_must_be_present(self):
        question = {
            "question": "关联题",
            "expected_evidence": [
                {"source": "A.pdf", "page": 2},
                {"source": "B.pdf", "page": 3},
            ],
        }
        partial = score_question(question, [{"source": "A.pdf", "page": 2}])
        self.assertFalse(partial["hit"])
        self.assertIsNone(partial["rank"])
        self.assertEqual(partial["evidence_coverage"], 0.5)

        complete = score_question(
            question,
            [
                {"source": "A.pdf", "page": 2},
                {"source": "other.pdf", "page": 3},
                {"source": "B.pdf", "page": 3},
            ],
        )
        self.assertTrue(complete["hit"])
        self.assertEqual(complete["rank"], 3)
        self.assertAlmostEqual(summarize([partial, complete])["mrr"], 1 / 6)

    def test_one_nearby_result_cannot_cover_two_pages(self):
        question = {
            "question": "相邻页",
            "expected_evidence": [
                {"source": "A.pdf", "page": 2},
                {"source": "A.pdf", "page": 3},
            ],
        }
        detail = score_question(question, [{"source": "A.pdf", "page": 2}], 1)
        self.assertFalse(detail["hit"])
        self.assertEqual(detail["evidence_coverage"], 0.5)

    def test_duplicate_evidence_is_rejected(self):
        question = {
            "question": "重复页",
            "expected_evidence": [
                {"source": "A.pdf", "page": 2},
                {"source": "A.pdf", "page": 2},
            ],
        }
        with self.assertRaises(ValueError):
            validate_question(question, 1)


if __name__ == "__main__":
    unittest.main()
