"""BM25 tests: no Milvus connection and no model calls."""

import unittest

from retriever import _bm25_search


class BM25SearchTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"text": "DeviceRadar uses key packets.", "source": "A.pdf", "page": 1},
            {"text": "ProﬁlIoT uses TCP sessions.", "source": "B.pdf", "page": 1},
            {"text": "设备指纹识别基于流量。", "source": "C.pdf", "page": 1},
            {"text": "Unrelated measurement study.", "source": "D.pdf", "page": 1},
        ]

    def test_english_term_in_chinese_question(self):
        hits = _bm25_search(self.chunks, "DeviceRadar 的关键包", 3)
        self.assertEqual(hits[0]["source"], "A.pdf")

    def test_pdf_ligature_matches_normal_name(self):
        hits = _bm25_search(self.chunks, "ProfilIoT 的 TCP 会话", 3)
        self.assertEqual(hits[0]["source"], "B.pdf")

    def test_chinese_bigrams_match(self):
        hits = _bm25_search(self.chunks, "如何设备指纹识别？", 3)
        self.assertEqual(hits[0]["source"], "C.pdf")

    def test_no_overlap_does_not_return_arbitrary_chunks(self):
        self.assertEqual(_bm25_search(self.chunks, "完全不同的词", 3), [])
        self.assertEqual(_bm25_search([], "DeviceRadar", 3), [])


if __name__ == "__main__":
    unittest.main()
