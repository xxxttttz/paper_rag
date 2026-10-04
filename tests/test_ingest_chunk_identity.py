"""Ingestion uses repeatable IDs without calling embedding or Milvus."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ingest


class IngestChunkIdentityTests(unittest.TestCase):
    def test_same_pdf_builds_same_distinct_chunk_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "paper.pdf"
            pdf.write_bytes(b"original PDF")
            with (
                patch.object(ingest, "extract_pages_markdown", return_value=[(1, "page text")]),
                patch.object(ingest, "chunk_markdown", return_value=["first", "second"]),
            ):
                first = ingest.build_chunks_from_pdfs(directory)
                second = ingest.build_chunks_from_pdfs(directory)
                pdf.write_bytes(b"updated PDF")
                changed = ingest.build_chunks_from_pdfs(directory)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertNotEqual(first[0]["id"], first[1]["id"])
        self.assertNotEqual(first[0]["id"], changed[0]["id"])


if __name__ == "__main__":
    unittest.main()
