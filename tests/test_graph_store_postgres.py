"""Opt-in PostgreSQL schema test; all writes roll back after the test."""

import os
from contextlib import contextmanager

import psycopg
import pytest
from psycopg.rows import dict_row

import database
import graph_builder
import graph_store


@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1", reason="requires local PostgreSQL"
)
def test_graph_build_is_evidence_backed_and_bound_to_text_version(monkeypatch):
    assert database.DATABASE_URL, "DATABASE_URL must point to the local test PostgreSQL"
    conn = psycopg.connect(database.DATABASE_URL, row_factory=dict_row)

    @contextmanager
    def uncommitted_connection(database_url):
        assert database_url == database.DATABASE_URL
        yield conn

    monkeypatch.setattr(graph_store, "connection", uncommitted_connection)
    try:
        assert graph_store.get_ready_graph(database.DATABASE_URL, "paper_chunks_version_test") is None
        build_id = graph_store.begin_graph_build(
            database.DATABASE_URL, "paper_chunks_version_test"
        )
        assert graph_store.get_ready_graph(database.DATABASE_URL, "paper_chunks_version_test") is None

        conn.execute(
            """INSERT INTO graph_nodes (build_id, id, entity_type, canonical_name)
               VALUES (%s, 'paper', 'paper', 'Test Paper'),
                      (%s, 'method', 'method', 'Test Method')""",
            (build_id, build_id),
        )
        conn.execute(
            """INSERT INTO graph_edges
               (build_id, id, source_node_id, relation_type, target_node_id)
               VALUES (%s, 'edge', 'paper', 'PROPOSES', 'method')""",
            (build_id,),
        )

        with pytest.raises(RuntimeError, match="without source evidence"):
            graph_store.mark_graph_ready(database.DATABASE_URL, build_id, 2, 1)

        conn.execute(
            """INSERT INTO graph_edge_evidence
               (build_id, edge_id, chunk_id, source, page, quote)
               VALUES (%s, 'edge', 'chunk-1', 'paper.pdf', 2, 'We propose Test Method.')""",
            (build_id,),
        )
        graph_store.mark_graph_ready(database.DATABASE_URL, build_id, 2, 1)

        conn.execute(
            """INSERT INTO graph_mentions
               (build_id, node_id, chunk_id, source, page)
               VALUES (%s, 'paper', 'anchor', 'paper.pdf', 1)""",
            (build_id,),
        )
        by_name = graph_store.find_related_evidence(
            database.DATABASE_URL, "paper_chunks_version_test", "Explain Test Paper", [],
        )
        by_anchor = graph_store.find_related_evidence(
            database.DATABASE_URL, "paper_chunks_version_test", "Explain it", ["anchor"],
        )
        by_filename = graph_store.find_related_evidence(
            database.DATABASE_URL, "paper_chunks_version_test", "paper.pdf", [],
        )
        assert [item["chunk_id"] for item in by_name] == ["chunk-1"]
        assert [item["chunk_id"] for item in by_anchor] == ["chunk-1"]
        assert [item["chunk_id"] for item in by_filename] == ["chunk-1"]
        assert graph_store.find_related_evidence(
            database.DATABASE_URL, "paper_chunks_version_other", "Test Paper", ["anchor"],
        ) == []

        assert graph_store.get_ready_graph(
            database.DATABASE_URL, "paper_chunks_version_test"
        )["id"] == build_id
        assert graph_store.get_ready_graph(
            database.DATABASE_URL, "paper_chunks_version_other"
        ) is None

        class FakeMilvus:
            def __init__(self, target):
                self.target = target

            def list_collections(self):
                return [self.target]

            def has_collection(self, name):
                return name == "paper_chunks"

            def describe_alias(self, alias):
                return {"collection_name": self.target}

        assert graph_store.get_ready_graph_for_alias(
            database.DATABASE_URL, FakeMilvus("paper_chunks_version_test"), "paper_chunks"
        )["id"] == build_id
        assert graph_store.get_ready_graph_for_alias(
            database.DATABASE_URL, FakeMilvus("paper_chunks_version_other"), "paper_chunks"
        ) is None
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1", reason="requires local PostgreSQL"
)
def test_graph_builder_writes_verbatim_edge_evidence(monkeypatch):
    assert database.DATABASE_URL
    conn = psycopg.connect(database.DATABASE_URL, row_factory=dict_row)

    @contextmanager
    def uncommitted_connection(database_url):
        assert database_url == database.DATABASE_URL
        yield conn

    monkeypatch.setattr(graph_store, "connection", uncommitted_connection)
    monkeypatch.setattr(graph_builder, "connection", uncommitted_connection)
    record = {
        "id": "chunk-build-test", "source": "DeviceRadar.pdf", "page": 3,
        "text": "DeviceRadar uses packet size as a feature.",
    }
    relation = {
        "source": ("paper", "DeviceRadar"), "target": ("feature", "packet size"),
        "type": "HAS_FEATURE", "quote": record["text"],
    }
    monkeypatch.setattr(graph_builder, "extract_relations", lambda _: [relation])
    try:
        result = graph_builder.build_graph_index(
            [record], "paper_chunks_version_builder_test", database.DATABASE_URL
        )
        assert result["nodes"] == 2
        assert result["edges"] == 1
        evidence = graph_store.find_related_evidence(
            database.DATABASE_URL, "paper_chunks_version_builder_test", "DeviceRadar", []
        )
        assert evidence[0]["chunk_id"] == record["id"]
        assert evidence[0]["page"] == 3
    finally:
        conn.rollback()
        conn.close()
