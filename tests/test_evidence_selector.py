"""Pure tests for opt-in evidence diversity selection."""

import unittest

from evidence_selector import select_diverse_evidence, select_requirement_evidence


class EvidenceSelectorTests(unittest.TestCase):
    def test_prefers_different_pages_to_duplicate_top_hits(self):
        candidates = [
            {"source": "A.pdf", "page": 1, "text": "same method and feature", "score": 0.95},
            {"source": "A.pdf", "page": 1, "text": "same method and feature", "score": 0.94},
            {"source": "B.pdf", "page": 2, "text": "different dataset evaluation", "score": 0.75},
        ]
        selected = select_diverse_evidence(candidates, 2)
        self.assertEqual([(c["source"], c["page"]) for c in selected], [("A.pdf", 1), ("B.pdf", 2)])

    def test_fills_with_same_page_if_no_alternative(self):
        candidates = [
            {"source": "A.pdf", "page": 1, "text": "method", "score": 0.9},
            {"source": "A.pdf", "page": 1, "text": "result", "score": 0.8},
        ]
        self.assertEqual(len(select_diverse_evidence(candidates, 2)), 2)

    def test_empty_and_nonpositive_k(self):
        self.assertEqual(select_diverse_evidence([], 5), [])
        self.assertEqual(select_diverse_evidence([{"source": "A", "page": 1, "text": "x"}], 0), [])


if __name__ == "__main__":
    unittest.main()


class RequirementSelectorTests(unittest.TestCase):
    def test_explicit_two_papers_each_get_an_evidence_slot(self):
        candidates = [
            {"source": "DeviceRadar.pdf", "page": 1, "text": "overview", "score": 0.99},
            {"source": "DeviceRadar.pdf", "page": 2, "text": "other overview", "score": 0.95},
            {"source": "ProfilIoT.pdf", "page": 3, "text": "classifier result", "score": 0.65},
        ]
        selected = select_requirement_evidence(
            "DeviceRadar 和 ProfilIoT 各自用什么分类器？", candidates, 2
        )
        self.assertEqual({hit["source"] for hit in selected}, {
            "DeviceRadar.pdf", "ProfilIoT.pdf"
        })

    def test_implicit_two_paper_comparison_uses_both_sources(self):
        candidates = [
            {"source": "A.pdf", "page": 1, "text": "overview", "score": 0.9},
            {"source": "A.pdf", "page": 2, "text": "method", "score": 0.8},
            {"source": "B.pdf", "page": 1, "text": "dataset", "score": 0.7},
        ]
        selected = select_requirement_evidence("两篇论文分别用什么数据？", candidates, 2)
        self.assertEqual({hit["source"] for hit in selected}, {"A.pdf", "B.pdf"})

    def test_numeric_facets_can_rescue_a_lower_ranked_page(self):
        candidates = [
            {"source": "A.pdf", "page": 1, "text": "network overview", "score": 0.9},
            {"source": "A.pdf", "page": 2, "text": "packet pipeline", "score": 0.8},
            {"source": "A.pdf", "page": 3, "text": "40 Gbps latency", "score": 0.7},
        ]
        selected = select_requirement_evidence("40 Gbps latency?", candidates, 2)
        self.assertEqual([hit["page"] for hit in selected], [1, 3])

    def test_no_candidate_is_fabricated(self):
        candidates = [{"source": "A.pdf", "page": 1, "text": "only", "score": 0.9}]
        self.assertEqual(
            select_requirement_evidence("A and missing B", candidates, 5), candidates
        )
        self.assertEqual(select_requirement_evidence("question", candidates, 0), [])
