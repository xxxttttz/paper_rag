"""Stable chunk IDs support graph evidence links across unchanged rebuilds."""

import tempfile
import unittest
from pathlib import Path

from chunk_identity import file_sha256, stable_chunk_id


class ChunkIdentityTests(unittest.TestCase):
    def test_same_pdf_content_and_chunk_produce_same_id(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "paper.pdf"
            pdf.write_bytes(b"same PDF bytes")
            digest = file_sha256(str(pdf))
            first = stable_chunk_id("paper.pdf", digest, 2, 0, "method text")
            second = stable_chunk_id("paper.pdf", file_sha256(str(pdf)), 2, 0, "method text")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)

    def test_id_changes_with_page_position_text_or_document(self):
        base = stable_chunk_id("A.pdf", "digest-a", 2, 0, "text")
        variants = [
            stable_chunk_id("B.pdf", "digest-a", 2, 0, "text"),
            stable_chunk_id("A.pdf", "digest-b", 2, 0, "text"),
            stable_chunk_id("A.pdf", "digest-a", 3, 0, "text"),
            stable_chunk_id("A.pdf", "digest-a", 2, 1, "text"),
            stable_chunk_id("A.pdf", "digest-a", 2, 0, "other text"),
        ]
        self.assertTrue(all(candidate != base for candidate in variants))


if __name__ == "__main__":
    unittest.main()
