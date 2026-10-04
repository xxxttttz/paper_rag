"""Stable identities for source PDFs and their text chunks."""

import hashlib
import uuid


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(
    source: str, document_sha256: str, page: int, chunk_index: int, text: str
) -> str:
    """Stay stable across rebuilds with unchanged PDF and chunking settings."""
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    name = f"{source}\0{document_sha256}\0{page}\0{chunk_index}\0{text_sha256}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))
