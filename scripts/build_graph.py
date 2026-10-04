"""Build a graph for the current Milvus text index without re-embedding PDFs.

Run `python scripts/build_graph.py` for a read-only status check, then add
`--execute` to send each current text chunk to the configured chat model.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
import graph_store
from graph_builder import build_graph_index, extract_relations
from milvus_store import active_collection_name, create_milvus_client


MAX_CHUNKS = 10000


def load_current_index() -> tuple[str, int, list[dict]]:
    """Read one immutable physical version, rejecting incomplete fetches."""
    client = create_milvus_client()
    try:
        physical = active_collection_name(client, config.COLLECTION_NAME)
        client.load_collection(physical)
        count = int(client.query(
            collection_name=physical,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )[0]["count(*)"])
        if count < 1 or count > MAX_CHUNKS:
            raise RuntimeError(f"Text index has {count} chunks; supported range is 1-{MAX_CHUNKS}")
        rows = client.query(
            collection_name=physical,
            filter="",
            output_fields=["id", "text", "source", "page"],
            limit=count,
            consistency_level="Strong",
        )
        if len(rows) != count or any(
            not all(key in row for key in ("id", "text", "source", "page")) for row in rows
        ):
            raise RuntimeError("Could not fetch the complete text index")
        return physical, count, rows
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a graph for the active text index")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Call the chat model once per chunk and publish a ready graph")
    action.add_argument("--probe", action="store_true", help="Call the chat model for one chunk and print only relation counts")
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5), help="Parallel extraction calls, 1-4 (default: 3)")
    args = parser.parse_args()

    if not database.DATABASE_URL:
        parser.error("DATABASE_URL is required for graph storage")
    physical, count, rows = load_current_index()
    print(f"当前物理文本索引: {physical}; 片段数: {count}", flush=True)
    existing = graph_store.get_ready_graph(database.DATABASE_URL, physical)
    if existing:
        print(f"已有 ready 图: {existing['id']}；不重复调用模型", flush=True)
        return
    if not args.execute and not args.probe:
        print("当前版本没有 ready 图。加 --execute 构建；将额外调用对话模型约 "
              f"{count} 次，不会重算 embedding 或图片索引。", flush=True)
        return
    if not config.OPENAI_API_KEY:
        parser.error("OPENAI_API_KEY is required for graph extraction")
    if args.probe:
        relations = extract_relations(rows[0])
        print(f"单片段试抽取: {len(relations)} 条通过证据校验的关系", flush=True)
        return

    def show_progress(done: int, total: int) -> None:
        if done == 1 or done % 10 == 0 or done == total:
            print(f"关系抽取进度: {done}/{total}", flush=True)

    result = build_graph_index(
        rows, physical, database.DATABASE_URL,
        progress=show_progress, extraction_workers=args.workers,
    )
    print(f"图构建完成: {result['nodes']} 个节点, {result['edges']} 条关系, "
          f"build_id={result['build_id']}", flush=True)


if __name__ == "__main__":
    main()
