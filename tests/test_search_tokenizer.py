"""Tokenization tests that run without external services or API calls."""

import unittest

from search_tokenizer import tokenize_search_text


class SearchTokenizerTests(unittest.TestCase):
    def test_names_case_punctuation_and_pdf_ligature(self):
        self.assertEqual(
            tokenize_search_text("DeviceRadar、ProﬁlIoT：packet embedding/40 Gbps"),
            ["deviceradar", "profiliot", "packet", "embedding", "40", "gbps"],
        )

    def test_chinese_bigrams(self):
        self.assertEqual(
            tokenize_search_text("设备指纹识别"),
            ["设备", "备指", "指纹", "纹识", "识别"],
        )

    def test_empty_or_punctuation_only(self):
        self.assertEqual(tokenize_search_text("  / — … "), [])


if __name__ == "__main__":
    unittest.main()
