"""Check source filtering with a real, temporary Milvus Lite index."""

import sys

import pytest
from pymilvus import MilvusClient

import retriever


@pytest.mark.skipif(sys.platform == "win32", reason="Milvus Lite runs in Linux/WSL")
def test_comparison_filters_sources_in_actual_index(tmp_path, monkeypatch):
    pytest.importorskip("milvus_lite")
    client = MilvusClient(uri=str(tmp_path / "comparison.db"))
    collection = "comparison_test"
    sources = ['Alpha "quoted".pdf', "Beta.pdf"]
    client.create_collection(
        collection_name=collection, dimension=2, id_type="string", max_length=64,
        auto_id=False, metric_type="COSINE", enable_dynamic_field=True,
    )
    client.insert(collection, [
        {"id": f"{index}-{page}", "source": source, "page": page,
         "text": f"paper {index} method {page}", "vector": [1.0, 0.1 * page]}
        for index, source in enumerate(sources + ["unselected.pdf"])
        for page in (1, 2, 3)
    ])
    monkeypatch.setattr(retriever, "create_milvus_client", lambda: client)
    monkeypatch.setattr(retriever, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(retriever.config, "COLLECTION_NAME", collection)
    monkeypatch.setattr(retriever.config, "USE_HYBRID_SEARCH", False)
    monkeypatch.setattr(retriever.config, "ENABLE_RERANK", False)

    hits = retriever.retrieve_comparison_evidence("method", sources, per_source_k=2)

    assert [hit["source"] for hit in hits] == [sources[0], sources[0], sources[1], sources[1]]
    assert len({(hit["source"], hit["page"]) for hit in hits}) == 4
