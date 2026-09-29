"""Shared Milvus connection helpers for Lite and server deployments."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from pymilvus import MilvusClient

import config


SERVER_URI_PREFIXES = ("http://", "https://", "tcp://")
logger = logging.getLogger(__name__)


def collection_aliases(client: MilvusClient, name: str) -> list[str]:
    """Normalize alias responses across PyMilvus versions."""
    response = client.list_aliases(collection_name=name)
    return response.get("aliases", []) if isinstance(response, dict) else response


def _publish_collection(client: MilvusClient, name: str, staging: str) -> None:
    """Publish a ready collection, retaining the previous physical version."""
    if name in client.list_collections():
        # Upgrade legacy physical names to aliases on the first safe rebuild.
        backup = f"{name}_backup_{uuid.uuid4().hex}"
        try:
            client.rename_collection(old_name=name, new_name=backup)
            client.create_alias(collection_name=staging, alias=name)
        except Exception:
            # Also handle a rename that succeeded remotely but timed out locally.
            if client.has_collection(backup) and not client.has_collection(name):
                client.rename_collection(old_name=backup, new_name=name)
            raise
    elif client.has_collection(name):
        client.alter_alias(collection_name=staging, alias=name)
    else:
        client.create_alias(collection_name=staging, alias=name)


@contextmanager
def staged_collection(
    client: MilvusClient,
    name: str,
    builder: Callable[[MilvusClient, str], None],
    expected_count: int,
) -> Iterator[str]:
    """Build and validate a private version before switching the public alias."""
    staging = f"{name}_version_{uuid.uuid4().hex}"
    try:
        builder(client, staging)
        yield staging
        client.flush(staging)
        client.load_collection(staging)
        counts = client.query(
            collection_name=staging,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        actual_count = int(counts[0]["count(*)"])
        if actual_count != expected_count:
            raise RuntimeError(
                f"Index row count mismatch: expected {expected_count}, got {actual_count}"
            )
        _publish_collection(client, name, staging)
    except Exception:
        try:
            # A publish RPC may succeed before its response times out. Never
            # remove a version that already has a public alias attached.
            if client.has_collection(staging) and not collection_aliases(client, staging):
                client.drop_collection(staging)
        except Exception:
            logger.exception("Could not clean up unpublished index %s", staging)
        raise


def is_milvus_lite(uri: str | None = None) -> bool:
    """Return whether the target is a local Milvus Lite database file."""
    target = uri or config.MILVUS_URI
    return not target.lower().startswith(SERVER_URI_PREFIXES)


def ensure_local_storage_directory() -> None:
    """Create the Lite database parent directory; server mode needs no path."""
    if is_milvus_lite():
        parent = os.path.dirname(os.path.abspath(config.MILVUS_URI))
        os.makedirs(parent, exist_ok=True)


def create_milvus_client() -> MilvusClient:
    """Create a client using the current Lite or server configuration."""
    if is_milvus_lite():
        ensure_local_storage_directory()
        return MilvusClient(uri=config.MILVUS_URI)

    options: dict[str, str] = {
        "uri": config.MILVUS_URI,
        "db_name": config.MILVUS_DATABASE,
    }
    if config.MILVUS_TOKEN:
        options["token"] = config.MILVUS_TOKEN
    return MilvusClient(**options)


def collection_exists(collection_name: str) -> bool:
    """Check collection existence without leaking a client connection."""
    if is_milvus_lite() and not os.path.exists(config.MILVUS_URI):
        return False

    client = create_milvus_client()
    try:
        return client.has_collection(collection_name)
    finally:
        client.close()


def configured_target() -> str:
    """Return a display-safe description of the active Milvus target."""
    if is_milvus_lite():
        return os.path.abspath(config.MILVUS_URI)
    return f"{config.MILVUS_URI} (database={config.MILVUS_DATABASE})"
