"""Comparison recall is source scoped and preserves paper coverage."""

import json
from unittest.mock import MagicMock, patch

import pytest

import retriever


@pytest.mark.parametrize("hybrid", [True, False])
def test_single_embedding_and_rerank_preserve_each_selected_paper(hybrid):
    sources = ["Alpha.pdf", "Beta.pdf"]
    chunks = [
        {"id": f"{source}-{page}", "source": source, "page": page, "text": f"method {page}", "score": 1 / page}
        for source in sources for page in (1, 2, 3)
    ]
    mc = MagicMock()

    def vector_search(client, vector, top_k, physical, source):
        assert physical == "physical-version"
        return [item.copy() for item in chunks if item["source"] == source]

    with (
        patch.object(retriever, "create_milvus_client", return_value=mc),
        patch.object(retriever, "active_collection_name", return_value="physical-version"),
        patch.object(retriever, "_load_all_chunks", return_value=chunks),
        patch.object(retriever, "embed_query", return_value=[0.0]) as embed,
        patch.object(retriever, "_vector_search", side_effect=vector_search),
        patch.object(retriever, "rerank", side_effect=lambda q, hits, top_n: hits) as rank,
        patch.object(retriever.config, "ENABLE_RERANK", True),
        patch.object(retriever.config, "USE_HYBRID_SEARCH", hybrid),
    ):
        hits = retriever.retrieve_comparison_evidence("method", sources, per_source_k=2)
    embed.assert_called_once()
    rank.assert_called_once()
    assert len(hits) == 4
    assert [hit["source"] for hit in hits] == [sources[0], sources[0], sources[1], sources[1]]
    mc.close.assert_called_once()


def test_unindexed_paper_fails_before_model_calls():
    mc = MagicMock()
    with (
        patch.object(retriever, "create_milvus_client", return_value=mc),
        patch.object(retriever, "active_collection_name", return_value="physical"),
        patch.object(retriever, "_load_all_chunks", return_value=[{"source": "Alpha.pdf"}]),
        patch.object(retriever, "embed_query") as embed,
    ):
        with pytest.raises(retriever.ComparisonIndexError, match="Beta.pdf"):
            retriever.retrieve_comparison_evidence("question", ["Alpha.pdf", "Beta.pdf"])
    embed.assert_not_called()
    mc.close.assert_called_once()


def test_vector_filter_escapes_source_name():
    mc = MagicMock()
    mc.search.return_value = [[]]
    source = 'paper" or source == "other.pdf'
    retriever._vector_search(mc, [0.0], 5, source=source)
    expression = mc.search.call_args.kwargs["filter"]
    assert expression == "source == " + json.dumps(source, ensure_ascii=False)
