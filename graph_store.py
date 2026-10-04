"""Versioned PostgreSQL storage for evidence-backed paper relationships.

Graph builds are keyed to a physical Milvus text collection. A build becomes
queryable only after it is marked ready; no graph is used for a different text
index version or for an incomplete build.
"""

from __future__ import annotations

import uuid

from milvus_store import active_collection_name
from postgres_database import connection


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS graph_builds (
        id TEXT PRIMARY KEY,
        text_collection TEXT NOT NULL CHECK (length(text_collection) > 0),
        status TEXT NOT NULL CHECK (status IN ('building', 'ready', 'failed')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        ready_at TIMESTAMPTZ,
        error TEXT,
        CHECK (status <> 'ready' OR ready_at IS NOT NULL)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_nodes (
        build_id TEXT NOT NULL REFERENCES graph_builds(id) ON DELETE CASCADE,
        id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
        PRIMARY KEY (build_id, id),
        UNIQUE (build_id, entity_type, canonical_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edges (
        build_id TEXT NOT NULL REFERENCES graph_builds(id) ON DELETE CASCADE,
        id TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        confidence DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
        PRIMARY KEY (build_id, id),
        FOREIGN KEY (build_id, source_node_id)
            REFERENCES graph_nodes(build_id, id) ON DELETE CASCADE,
        FOREIGN KEY (build_id, target_node_id)
            REFERENCES graph_nodes(build_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_mentions (
        build_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        source TEXT NOT NULL,
        page INTEGER NOT NULL CHECK (page > 0),
        PRIMARY KEY (build_id, node_id, chunk_id),
        FOREIGN KEY (build_id, node_id)
            REFERENCES graph_nodes(build_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edge_evidence (
        build_id TEXT NOT NULL,
        edge_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        source TEXT NOT NULL,
        page INTEGER NOT NULL CHECK (page > 0),
        quote TEXT NOT NULL CHECK (length(quote) > 0),
        PRIMARY KEY (build_id, edge_id, chunk_id),
        FOREIGN KEY (build_id, edge_id)
            REFERENCES graph_edges(build_id, id) ON DELETE CASCADE
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_graph_builds_collection
       ON graph_builds(text_collection, status, ready_at DESC)""",
    """CREATE INDEX IF NOT EXISTS idx_graph_edges_source
       ON graph_edges(build_id, source_node_id, relation_type)""",
    """CREATE INDEX IF NOT EXISTS idx_graph_edges_target
       ON graph_edges(build_id, target_node_id, relation_type)""",
    """CREATE INDEX IF NOT EXISTS idx_graph_mentions_chunk
       ON graph_mentions(build_id, chunk_id)""",
    """CREATE INDEX IF NOT EXISTS idx_graph_edge_evidence_chunk
       ON graph_edge_evidence(build_id, chunk_id)""",
)


def init_graph_schema(database_url: str) -> None:
    """Create versioned graph tables without touching chat or Milvus data."""
    if not database_url:
        raise ValueError("Graph storage requires a PostgreSQL DATABASE_URL")
    with connection(database_url) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


def begin_graph_build(database_url: str, text_collection: str) -> str:
    """Register an unpublished graph build for one physical text collection."""
    if not text_collection or not text_collection.strip():
        raise ValueError("text_collection must be a physical collection name")
    init_graph_schema(database_url)
    build_id = str(uuid.uuid4())
    with connection(database_url) as conn:
        conn.execute(
            """INSERT INTO graph_builds (id, text_collection, status)
               VALUES (%s, %s, 'building')""",
            (build_id, text_collection),
        )
    return build_id


def mark_graph_ready(
    database_url: str, build_id: str, expected_nodes: int, expected_edges: int
) -> None:
    """Publish only after node/edge counts match the completed extraction."""
    if expected_nodes < 0 or expected_edges < 0:
        raise ValueError("expected counts cannot be negative")
    with connection(database_url) as conn:
        build = conn.execute(
            "SELECT status FROM graph_builds WHERE id = %s FOR UPDATE", (build_id,)
        ).fetchone()
        if not build or build["status"] != "building":
            raise ValueError("graph build is not in building state")
        node_count = conn.execute(
            "SELECT count(*) AS count FROM graph_nodes WHERE build_id = %s", (build_id,)
        ).fetchone()["count"]
        edge_count = conn.execute(
            "SELECT count(*) AS count FROM graph_edges WHERE build_id = %s", (build_id,)
        ).fetchone()["count"]
        if (node_count, edge_count) != (expected_nodes, expected_edges):
            raise RuntimeError(
                f"Graph count mismatch: nodes {node_count}/{expected_nodes}, "
                f"edges {edge_count}/{expected_edges}"
            )
        unbacked_edges = conn.execute(
            """SELECT count(*) AS count FROM graph_edges AS edge
               WHERE edge.build_id = %s AND NOT EXISTS (
                   SELECT 1 FROM graph_edge_evidence AS evidence
                   WHERE evidence.build_id = edge.build_id
                     AND evidence.edge_id = edge.id
               )""",
            (build_id,),
        ).fetchone()["count"]
        if unbacked_edges:
            raise RuntimeError(f"Graph has {unbacked_edges} edges without source evidence")
        conn.execute(
            """UPDATE graph_builds SET status = 'ready', ready_at = now(), error = NULL
               WHERE id = %s""",
            (build_id,),
        )


def mark_graph_failed(database_url: str, build_id: str, error: str) -> None:
    with connection(database_url) as conn:
        conn.execute(
            """UPDATE graph_builds SET status = 'failed', error = %s
               WHERE id = %s AND status = 'building'""",
            (error[:1000], build_id),
        )


def get_ready_graph(database_url: str, text_collection: str) -> dict | None:
    """Find a ready graph for exactly the requested physical text version."""
    if not database_url:
        return None
    with connection(database_url) as conn:
        table = conn.execute(
            "SELECT to_regclass('graph_builds') AS table_name"
        ).fetchone()["table_name"]
        if table is None:
            return None
        return conn.execute(
            """SELECT id, text_collection, ready_at FROM graph_builds
               WHERE text_collection = %s AND status = 'ready'
               ORDER BY ready_at DESC, created_at DESC LIMIT 1""",
            (text_collection,),
        ).fetchone()


def get_ready_graph_for_alias(database_url: str, milvus_client, public_name: str) -> dict | None:
    """Return None while the current Milvus text version has no ready graph."""
    if not database_url:
        return None  # Lite/SQLite mode keeps the existing non-graph retrieval.
    physical_collection = active_collection_name(milvus_client, public_name)
    return get_ready_graph(database_url, physical_collection)


def find_related_evidence(
    database_url: str,
    text_collection: str,
    question: str,
    anchor_chunk_ids: list[str],
    limit: int = 20,
) -> list[dict]:
    """Find source-backed edge evidence for named or retrieved graph nodes.

    Only the ready build for this exact physical text collection is visible.
    Returned chunk IDs still need to be resolved against that collection before
    they can be used as answer context or citations.
    """
    if not database_url or limit <= 0:
        return []
    graph = get_ready_graph(database_url, text_collection)
    if graph is None:
        return []
    with connection(database_url) as conn:
        rows = conn.execute(
            """WITH seeds AS (
                   SELECT DISTINCT node.id
                   FROM graph_nodes AS node
                   WHERE node.build_id = %s AND (
                       (length(node.canonical_name) >= 4
                        AND position(lower(node.canonical_name) in lower(%s)) > 0)
                       OR EXISTS (
                           SELECT 1 FROM graph_mentions AS mention
                           WHERE mention.build_id = node.build_id
                             AND mention.node_id = node.id
                             AND mention.chunk_id = ANY(%s)
                       )
                   )
               )
               SELECT evidence.chunk_id, evidence.source, evidence.page,
                      coalesce(edge.confidence, 0) AS confidence,
                      edge.relation_type,
                      source_node.canonical_name AS source_name,
                      target_node.canonical_name AS target_name,
                      evidence.quote
               FROM graph_edges AS edge
               JOIN graph_edge_evidence AS evidence
                 ON evidence.build_id = edge.build_id AND evidence.edge_id = edge.id
               JOIN graph_nodes AS source_node
                 ON source_node.build_id = edge.build_id
                AND source_node.id = edge.source_node_id
               JOIN graph_nodes AS target_node
                 ON target_node.build_id = edge.build_id
                AND target_node.id = edge.target_node_id
               WHERE edge.build_id = %s AND (
                   edge.source_node_id IN (SELECT id FROM seeds)
                   OR edge.target_node_id IN (SELECT id FROM seeds)
                   OR (
                       length(split_part(evidence.source, '.', 1)) >= 4
                       AND position(
                           lower(split_part(evidence.source, '.', 1)) in lower(%s)
                       ) > 0
                   )
               )
               ORDER BY confidence DESC, evidence.source, evidence.page,
                        evidence.chunk_id
               LIMIT %s""",
            (graph["id"], question, anchor_chunk_ids, graph["id"], question, limit),
        ).fetchall()
    return list(rows)
