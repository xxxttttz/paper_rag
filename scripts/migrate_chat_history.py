"""Migrate Paper RAG chat history from SQLite to PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database
import postgres_database


TABLES = ("conversations", "messages", "citations", "image_citations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy chat history from SQLite to PostgreSQL.",
    )
    parser.add_argument("--source", default=database.CHAT_DB_PATH)
    parser.add_argument(
        "--target-url",
        default=(
            "postgresql://paper_rag:paper_rag_dev@127.0.0.1:5432/paper_rag"
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing target chat data before importing.",
    )
    return parser.parse_args()


def _timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _source_rows(source: sqlite3.Connection, table: str) -> list[dict]:
    return [dict(row) for row in source.execute(f"SELECT * FROM {table}")]


def migrate(args: argparse.Namespace) -> None:
    source_path = os.path.abspath(args.source)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"SQLite source does not exist: {source_path}")

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    try:
        source_data = {table: _source_rows(source, table) for table in TABLES}
    finally:
        source.close()

    postgres_database.init_database(args.target_url)
    with postgres_database.connection(args.target_url) as target:
        target_counts = {
            table: target.execute(f"SELECT COUNT(*) AS count FROM {table}")
            .fetchone()["count"]
            for table in TABLES
        }
        nonempty = {table: count for table, count in target_counts.items() if count}
        if nonempty and not args.replace:
            details = ", ".join(
                f"{table}={count}" for table, count in nonempty.items()
            )
            raise RuntimeError(
                f"Target already contains chat data ({details}). "
                "Rerun with --replace only if overwriting it is intended."
            )

        if args.replace:
            target.execute("DELETE FROM conversations")

        conversations = [
            (
                row["id"],
                row["title"],
                _timestamp(row["created_at"]),
                _timestamp(row["updated_at"]),
            )
            for row in source_data["conversations"]
        ]
        messages = [
            (
                row["id"],
                row["conversation_id"],
                row["role"],
                row["content"],
                _timestamp(row["created_at"]),
            )
            for row in source_data["messages"]
        ]
        citations = [
            (
                row["message_id"],
                row["source"],
                row["page"],
                row["chunk_text"],
                row["score"],
            )
            for row in source_data["citations"]
        ]
        image_citations = [
            (
                row["message_id"],
                row["image_path"],
                row["source"],
                row["page"],
                row["score"],
            )
            for row in source_data["image_citations"]
        ]

        with target.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO conversations
                (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)""",
                conversations,
            )
            cursor.executemany(
                """INSERT INTO messages
                (id, conversation_id, role, content, created_at)
                VALUES (%s, %s, %s, %s, %s)""",
                messages,
            )
            cursor.executemany(
                """INSERT INTO citations
                (message_id, source, page, chunk_text, score)
                VALUES (%s, %s, %s, %s, %s)""",
                citations,
            )
            cursor.executemany(
                """INSERT INTO image_citations
                (message_id, image_path, source, page, score)
                VALUES (%s, %s, %s, %s, %s)""",
                image_citations,
            )

        migrated_counts = {
            table: target.execute(f"SELECT COUNT(*) AS count FROM {table}")
            .fetchone()["count"]
            for table in TABLES
        }
        source_counts = {
            table: len(source_data[table])
            for table in TABLES
        }
        if source_counts != migrated_counts:
            raise RuntimeError(
                f"Count mismatch: source={source_counts}, target={migrated_counts}"
            )

    print(f"Source: {source_path}")
    print(f"Target: {args.target_url.rsplit('@', 1)[-1]}")
    for table in TABLES:
        print(f"  {table}: {len(source_data[table])}")
    print("Migration completed successfully.")


if __name__ == "__main__":
    migrate(parse_args())
