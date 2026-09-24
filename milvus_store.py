"""Shared Milvus connection helpers for Lite and server deployments."""

from __future__ import annotations

import os

from pymilvus import MilvusClient

import config


SERVER_URI_PREFIXES = ("http://", "https://", "tcp://")


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
