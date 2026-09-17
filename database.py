"""

用于对话、消息及回答引用的 SQLite 持久化存储
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterable


CHAT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chat_history.db")


def _now() -> str:
    """Return a sortable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CHAT_DB_PATH), exist_ok=True)
    connection = sqlite3.connect(CHAT_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    """Create the chat database and tables when they do not exist."""
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                score REAL,
                FOREIGN KEY (message_id)
                    REFERENCES messages(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_citations_message
                ON citations(message_id);
            """
        )


def create_conversation(title: str = "新会话") -> str:
    conversation_id = str(uuid.uuid4())
    timestamp = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title.strip() or "新会话", timestamp, timestamp),
        )
    return conversation_id


def list_conversations() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> None:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Conversation title cannot be empty")
    with _connect() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (clean_title, _now(), conversation_id),
        )


def delete_conversation(conversation_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )


def add_message(conversation_id: str, role: str, content: str) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    message_id = str(uuid.uuid4())
    timestamp = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, timestamp),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (timestamp, conversation_id),
        )
    return message_id


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict]:
    with _connect() as connection:
        if limit is None:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT id, conversation_id, role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (conversation_id, max(limit, 0)),
            ).fetchall()
    return [dict(row) for row in rows]


def add_citations(message_id: str, citations: Iterable[dict]) -> None:
    values = [
        (
            message_id,
            citation["source"],
            int(citation["page"]),
            citation.get("chunk_text", citation.get("text", "")),
            citation.get("score"),
        )
        for citation in citations
    ]
    if not values:
        return

    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO citations
                (message_id, source, page, chunk_text, score)
            VALUES (?, ?, ?, ?, ?)
            """,
            values,
        )


def get_citations(message_id: str) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, message_id, source, page, chunk_text, score
            FROM citations
            WHERE message_id = ?
            ORDER BY id ASC
            """,
            (message_id,),
        ).fetchall()
    return [dict(row) for row in rows]
