"""Migrate Paper RAG collections from Milvus Lite to Milvus Server.

The script copies existing vectors and metadata. It never calls embedding APIs.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from pymilvus import MilvusClient


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from image_ingest import build_image_collection
from ingest import build_collection
from milvus_store import is_milvus_lite, staged_collection


CollectionBuilder = Callable[[MilvusClient, str], None]

COLLECTIONS: dict[str, tuple[list[str], CollectionBuilder]] = {
    config.COLLECTION_NAME: (
        ["id", "text", "source", "page", "vector"],
        build_collection,
    ),
    config.IMAGE_COLLECTION_NAME: (
        ["id", "image_path", "source", "page", "vector"],
        build_image_collection,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Paper RAG collections from Milvus Lite to a server.",
    )
    parser.add_argument(
        "--source",
        default=config.MILVUS_DB_PATH,
        help="Path to the source Milvus Lite database.",
    )
    parser.add_argument(
        "--target-uri",
        default="http://127.0.0.1:19530",
        help="Milvus Server URI.",
    )
    parser.add_argument(
        "--target-database",
        default="default",
        help="Target Milvus database name.",
    )
    parser.add_argument(
        "--target-token",
        default="",
        help="Optional target token in user:password form.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace same-named collections that already exist on the target.",
    )
    return parser.parse_args()


def _row_count(client: MilvusClient, collection_name: str) -> int:
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count", 0))


def _copy_collection(
    source: MilvusClient,
    target: MilvusClient,
    collection_name: str,
    output_fields: list[str],
    build_target: CollectionBuilder,
    batch_size: int,
) -> tuple[int, int]:
    source.load_collection(collection_name)
    source_count = _row_count(source, collection_name)
    copied = 0
    with staged_collection(target, collection_name, build_target, source_count) as staging:
        iterator = source.query_iterator(
            collection_name=collection_name,
            batch_size=batch_size,
            output_fields=output_fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                target.insert(collection_name=staging, data=batch)
                copied += len(batch)
                print(f"  {collection_name}: {copied}/{source_count}")
        finally:
            iterator.close()

    target_count = _row_count(target, collection_name)
    return source_count, target_count


def migrate(args: argparse.Namespace) -> None:
    source_path = os.path.abspath(args.source)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Milvus Lite source does not exist: {source_path}")
    if not is_milvus_lite(source_path):
        raise ValueError("--source must point to a Milvus Lite database file")
    if is_milvus_lite(args.target_uri):
        raise ValueError("--target-uri must be an http(s) or tcp Milvus Server URI")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    target_options: dict[str, str] = {
        "uri": args.target_uri,
        "db_name": args.target_database,
    }
    if args.target_token:
        target_options["token"] = args.target_token

    source = MilvusClient(uri=source_path)
    target = MilvusClient(**target_options)
    try:
        selected = [name for name in COLLECTIONS if source.has_collection(name)]
        if not selected:
            raise RuntimeError("The source contains no Paper RAG collections")

        conflicts = [name for name in selected if target.has_collection(name)]
        if conflicts and not args.replace:
            names = ", ".join(conflicts)
            raise RuntimeError(
                f"Target collections already exist: {names}. "
                "Rerun with --replace only if overwriting them is intended."
            )

        print(f"Source: {source_path}")
        print(f"Target: {args.target_uri} (database={args.target_database})")
        print(f"Collections: {', '.join(selected)}")

        for name in selected:
            fields, builder = COLLECTIONS[name]
            source_count, target_count = _copy_collection(
                source,
                target,
                name,
                fields,
                builder,
                args.batch_size,
            )
            if source_count != target_count:
                raise RuntimeError(
                    f"Count mismatch for {name}: "
                    f"source={source_count}, target={target_count}"
                )
            print(f"  verified {name}: {target_count} rows")

        print("Migration completed successfully.")
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    migrate(parse_args())
