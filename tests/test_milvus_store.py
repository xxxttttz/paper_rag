"""Tests for Lite/server Milvus connection selection."""

from unittest.mock import patch

import config
import milvus_store


def test_milvus_mode_detection():
    assert milvus_store.is_milvus_lite("data/paper_rag.db") is True
    assert milvus_store.is_milvus_lite("http://127.0.0.1:19530") is False
    assert milvus_store.is_milvus_lite("https://milvus.example.com") is False


def test_server_client_uses_database_and_token(monkeypatch):
    monkeypatch.setattr(config, "MILVUS_URI", "http://milvus:19530")
    monkeypatch.setattr(config, "MILVUS_DATABASE", "paper_rag")
    monkeypatch.setattr(config, "MILVUS_TOKEN", "user:password")

    with patch.object(milvus_store, "MilvusClient") as client_class:
        milvus_store.create_milvus_client()

    client_class.assert_called_once_with(
        uri="http://milvus:19530",
        db_name="paper_rag",
        token="user:password",
    )


def test_missing_lite_file_has_no_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MILVUS_URI", str(tmp_path / "missing.db"))

    with patch.object(milvus_store, "MilvusClient") as client_class:
        exists = milvus_store.collection_exists(config.COLLECTION_NAME)

    assert exists is False
    client_class.assert_not_called()


def test_active_collection_name_uses_physical_name_for_legacy_index():
    client = type("FakeClient", (), {
        "list_collections": lambda self: ["paper_chunks"],
    })()

    assert milvus_store.active_collection_name(client, "paper_chunks") == "paper_chunks"


def test_active_collection_name_resolves_alias_target():
    class FakeClient:
        def list_collections(self):
            return ["paper_chunks_version_123"]

        def has_collection(self, name):
            return name == "paper_chunks"

        def describe_alias(self, alias):
            assert alias == "paper_chunks"
            return {"alias": alias, "collection_name": "paper_chunks_version_123"}

    assert milvus_store.active_collection_name(FakeClient(), "paper_chunks") == "paper_chunks_version_123"
