"""Query construction remains constrained to the ready text-index build."""

from contextlib import contextmanager

import graph_store


def test_named_pdf_source_can_seed_evidence_without_named_entity(monkeypatch):
    captured = {}

    class Cursor:
        def execute(self, statement, params):
            captured["statement"] = statement
            captured["params"] = params
            return self

        def fetchall(self):
            return [{"chunk_id": "chunk-b", "source": "ProfilIoT.pdf", "page": 2}]

    @contextmanager
    def fake_connection(database_url):
        assert database_url == "postgres://test"
        yield Cursor()

    monkeypatch.setattr(graph_store, "get_ready_graph", lambda *_: {"id": "build-1"})
    monkeypatch.setattr(graph_store, "connection", fake_connection)
    rows = graph_store.find_related_evidence(
        "postgres://test", "paper_chunks_v2", "ProfilIoT session classifier", [],
    )

    assert rows[0]["source"] == "ProfilIoT.pdf"
    assert "split_part(evidence.source" in captured["statement"]
    assert captured["params"] == (
        "build-1", "ProfilIoT session classifier", [], "build-1",
        "ProfilIoT session classifier", 20,
    )
