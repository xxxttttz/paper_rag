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
