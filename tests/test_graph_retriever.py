"""Graph retrieval must only surface evidence from the active text index."""

from unittest.mock import patch

import graph_retriever
import retriever


class FakeMilvus:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def load_collection(self, name):
        assert name == "paper_chunks_v2"

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return self.rows


def test_graph_evidence_is_verified_and_adds_a_distinct_page():
    baseline = [
        {"id": "anchor", "text": "baseline", "source": "A.pdf", "page": 1, "score": 0.9},
        {"id": "other", "text": "second", "source": "A.pdf", "page": 2, "score": 0.8},
    ]
    references = [
        {"chunk_id": "stale", "source": "B.pdf", "page": 2, "confidence": 0.99},
        {"chunk_id": "evidence", "source": "B.pdf", "page": 3, "confidence": 0.8},
    ]
    milvus = FakeMilvus([
        {"id": "stale", "text": "wrong version", "source": "B.pdf", "page": 4},
        {"id": "evidence", "text": "verified", "source": "B.pdf", "page": 3},
    ])
    with patch.object(graph_retriever.graph_store, "find_related_evidence", return_value=references) as lookup:
        result, graph_hits = graph_retriever.enrich_with_graph(
            milvus, "postgres://test", "paper_chunks_v2", "Question",
            baseline, top_k=2, graph_slots=1,
        )

    assert [(hit["source"], hit["page"]) for hit in result] == [
        ("A.pdf", 1), ("B.pdf", 3)
    ]
    assert graph_hits[0]["text"] == "verified"
    assert lookup.call_args.args[1:4] == ("paper_chunks_v2", "Question", ["anchor", "other"])
    assert milvus.queries[0]["collection_name"] == "paper_chunks_v2"


def test_graph_disabled_without_database_or_when_no_ready_build():
    baseline = [{"id": "anchor", "text": "baseline", "source": "A.pdf", "page": 1, "score": 0.9}]
    milvus = FakeMilvus([])
    with patch.object(graph_retriever.graph_store, "find_related_evidence", return_value=[]) as lookup:
        assert graph_retriever.enrich_with_graph(
            milvus, "", "paper_chunks_v2", "Question", baseline, 5, 2
        ) == (baseline, [])
        assert graph_retriever.enrich_with_graph(
            milvus, "postgres://test", "paper_chunks_v2", "Question", baseline, 5, 2
        ) == (baseline, [])
    lookup.assert_called_once()
    assert not milvus.queries


def test_question_named_source_is_preferred_over_unrelated_anchor_source():
    baseline = [
        {"id": "anchor", "text": "baseline", "source": "DeviceRadar.pdf", "page": 1, "score": 0.9}
    ]
    references = [
        {"chunk_id": "a", "source": "DeviceRadar.pdf", "page": 3, "confidence": 0},
        {"chunk_id": "b", "source": "ProfilIoT.pdf", "page": 2, "confidence": 0},
    ]
    milvus = FakeMilvus([
        {"id": "a", "text": "unrelated", "source": "DeviceRadar.pdf", "page": 3},
        {"id": "b", "text": "session features", "source": "ProfilIoT.pdf", "page": 2},
    ])
    with patch.object(graph_retriever.graph_store, "find_related_evidence", return_value=references):
        result, graph_hits = graph_retriever.enrich_with_graph(
            milvus, "postgres://test", "paper_chunks_v2", "How does ProfilIoT work?",
            baseline, top_k=2, graph_slots=1,
        )
    assert [(hit["source"], hit["page"]) for hit in graph_hits] == [
        ("ProfilIoT.pdf", 2)
    ]
    assert result[-1]["source"] == "ProfilIoT.pdf"


def test_retrieve_uses_one_physical_version_for_text_and_graph(monkeypatch):
    client = FakeMilvus([])
    client.close = lambda: None
    baseline = [{"id": "anchor", "text": "baseline", "source": "A.pdf", "page": 1, "score": 0.9}]
    monkeypatch.setattr(retriever.config, "ENABLE_GRAPH_RETRIEVAL", True)
    monkeypatch.setattr(retriever.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(retriever, "create_milvus_client", lambda: client)
    monkeypatch.setattr(retriever, "active_collection_name", lambda *_: "paper_chunks_v2")
    with (
        patch.object(retriever, "_retrieve_with_client", return_value=baseline) as text_retrieve,
        patch.object(retriever, "enrich_with_graph", return_value=(baseline, [])) as graph_retrieve,
    ):
        assert retriever.retrieve("question") == baseline
    assert text_retrieve.call_args.kwargs["collection_name"] == "paper_chunks_v2"
    assert graph_retrieve.call_args.args[2] == "paper_chunks_v2"


def test_graph_failure_does_not_discard_text_results(monkeypatch):
    client = FakeMilvus([])
    client.close = lambda: None
    baseline = [{"id": "anchor", "text": "baseline", "source": "A.pdf", "page": 1, "score": 0.9}]
    monkeypatch.setattr(retriever.config, "ENABLE_GRAPH_RETRIEVAL", True)
    monkeypatch.setattr(retriever.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(retriever, "create_milvus_client", lambda: client)
    monkeypatch.setattr(retriever, "active_collection_name", lambda *_: "paper_chunks_v2")
    with (
        patch.object(retriever, "_retrieve_with_client", return_value=baseline),
        patch.object(retriever, "enrich_with_graph", side_effect=RuntimeError("unavailable")),
    ):
        assert retriever.retrieve("question") == baseline
