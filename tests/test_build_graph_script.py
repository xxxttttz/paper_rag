"""The standalone graph tool must not call the model for status checks."""

import sys
from unittest.mock import Mock

import pytest

from scripts import build_graph


class FakeMilvus:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def load_collection(self, name):
        assert name == "paper_chunks_version_test"

    def query(self, **kwargs):
        assert kwargs["collection_name"] == "paper_chunks_version_test"
        if kwargs["output_fields"] == ["count(*)"]:
            return [{"count(*)": len(self.rows)}]
        return self.rows

    def close(self):
        self.closed = True


def test_load_current_index_uses_one_physical_version(monkeypatch):
    rows = [{"id": "one", "text": "text", "source": "A.pdf", "page": 1}]
    client = FakeMilvus(rows)
    monkeypatch.setattr(build_graph, "create_milvus_client", lambda: client)
    monkeypatch.setattr(build_graph, "active_collection_name", lambda *_: "paper_chunks_version_test")

    assert build_graph.load_current_index() == ("paper_chunks_version_test", 1, rows)
    assert client.closed


def test_status_check_does_not_call_extraction(monkeypatch, capsys):
    rows = [{"id": "one", "text": "text", "source": "A.pdf", "page": 1}]
    monkeypatch.setattr(sys, "argv", ["build_graph.py"])
    monkeypatch.setattr(build_graph.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(build_graph, "load_current_index", lambda: ("paper_chunks_version_test", 1, rows))
    monkeypatch.setattr(build_graph.graph_store, "get_ready_graph", lambda *_: None)
    extract = Mock()
    build = Mock()
    monkeypatch.setattr(build_graph, "extract_relations", extract)
    monkeypatch.setattr(build_graph, "build_graph_index", build)

    build_graph.main()

    assert "1 次" in capsys.readouterr().out
    extract.assert_not_called()
    build.assert_not_called()


def test_incomplete_fetch_is_rejected(monkeypatch):
    client = FakeMilvus([{"id": "one", "text": "text", "source": "A.pdf", "page": 1}])
    client.query = Mock(side_effect=[[{"count(*)": 2}], client.rows])
    monkeypatch.setattr(build_graph, "create_milvus_client", lambda: client)
    monkeypatch.setattr(build_graph, "active_collection_name", lambda *_: "paper_chunks_version_test")

    with pytest.raises(RuntimeError, match="complete text index"):
        build_graph.load_current_index()
    assert client.closed
