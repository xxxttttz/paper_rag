"""Synchronous knowledge-base ingestion orchestration."""

from __future__ import annotations

from collections.abc import Callable

import config
from image_ingest import rebuild_image_index
from ingest import build_chunks_from_pdfs, build_collection, embed_texts
from milvus_store import create_milvus_client


ProgressCallback = Callable[[str, dict], None]


def rebuild_knowledge_base(
    progress: ProgressCallback | None = None,
) -> dict:
    """Rebuild text and image collections and return a serializable summary."""
    notify = progress or (lambda _stage, _details: None)
    notify("parsing", {})
    records = build_chunks_from_pdfs(config.PDF_DIR)
    if not records:
        raise RuntimeError("Knowledge base is empty; upload at least one PDF")

    notify("embedding_text", {"text_chunks": len(records)})
    vectors = embed_texts([record["text"] for record in records])
    for record, vector in zip(records, vectors):
        record["vector"] = vector

    notify("writing_text", {"text_chunks": len(records)})
    milvus = create_milvus_client()
    try:
        build_collection(milvus)
        milvus.insert(collection_name=config.COLLECTION_NAME, data=records)
        milvus.flush(config.COLLECTION_NAME)

        notify("embedding_images", {"text_chunks": len(records)})
        try:
            image_records = rebuild_image_index(milvus)
            image_error = None
        except Exception as error:
            image_records = []
            image_error = str(error)
    finally:
        milvus.close()

    result = {
        "text_chunks": len(records),
        "images": len(image_records),
        "image_error": image_error,
    }
    notify("completed", result)
    return result
