"""Offline graph comparison never requires embedding or rerank calls."""

from unittest.mock import Mock

import pytest

from eval import evaluate_graph_local


def test_local_comparison_scores_verified_graph_page(monkeypatch):
    question = {
        "question": "How do A and B compare?",
        "expected_evidence": [
            {"source": "A.pdf", "page": 1},
            {"source": "B.pdf", "page": 2},
        ],
    }
    first = {"id": "a", "text": "A", "source": "A.pdf", "page": 1, "score": 1.0}
    second = {"id": "b", "text": "B", "source": "B.pdf", "page": 2, "score": None}
    client = Mock()
    monkeypatch.setattr(evaluate_graph_local.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(evaluate_graph_local, "create_milvus_client", lambda: client)
    monkeypatch.setattr(evaluate_graph_local, "active_collection_name", lambda *_: "physical-v1")
    monkeypatch.setattr(evaluate_graph_local, "get_ready_graph", lambda *_: {"id": "graph-v1"})
    monkeypatch.setattr(evaluate_graph_local, "_load_all_chunks", lambda *_args, **_kwargs: [first, second])
    monkeypatch.setattr(evaluate_graph_local, "_bm25_search", lambda *_args: [first])
    monkeypatch.setattr(
        evaluate_graph_local, "enrich_with_graph",
        lambda *_args: ([first, second], [second]),
    )

    result = evaluate_graph_local.evaluate_local([question], top_k=2, graph_slots=1)

    assert result["before"]["hit_rate"] == 0
    assert result["after"]["hit_rate"] == 1
    assert result["added"] == [[{"source": "B.pdf", "page": 2}]]
    client.close.assert_called_once()


def test_local_comparison_rejects_missing_graph(monkeypatch):
    client = Mock()
    monkeypatch.setattr(evaluate_graph_local.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(evaluate_graph_local, "create_milvus_client", lambda: client)
    monkeypatch.setattr(evaluate_graph_local, "active_collection_name", lambda *_: "physical-v1")
    monkeypatch.setattr(evaluate_graph_local, "get_ready_graph", lambda *_: None)
    with pytest.raises(RuntimeError, match="没有 ready 图"):
        evaluate_graph_local.evaluate_local([], top_k=5, graph_slots=2)
    client.close.assert_called_once()
