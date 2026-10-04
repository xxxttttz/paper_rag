"""Regression tests for preserving live indexes during failed rebuilds."""

from unittest.mock import Mock

import pytest

import config
import ingestion_service
import image_ingest
from milvus_store import staged_collection
from scripts.migrate_milvus import _copy_collection


class FakeMilvus:
    """Track physical versions and aliases independently, like Milvus does."""

    def __init__(self):
        self.collections = {}
        self.aliases = {}
        self.insert_error = False
        self.alias_error = None
        self.validation_error = None
        self.wrapped_aliases = False
        self.closed = False

    def has_collection(self, name):
        return name in self.collections or name in self.aliases

    def list_collections(self):
        return list(self.collections)

    def create_collection(self, name):
        assert not self.has_collection(name)
        self.collections[name] = 0

    def insert(self, collection_name, data):
        if self.insert_error:
            raise RuntimeError("insert failed")
        self.collections[collection_name] += len(data)

    def flush(self, name):
        if self.validation_error == "flush":
            raise RuntimeError("flush failed")

    def load_collection(self, name):
        if self.validation_error == "load":
            raise RuntimeError("load failed")

    def query(self, collection_name, **kwargs):
        target = self.aliases.get(collection_name, collection_name)
        return [{"count(*)": self.collections[target]}]

    def get_collection_stats(self, name):
        return {"row_count": self.query(name)[0]["count(*)"]}

    def rename_collection(self, old_name, new_name):
        assert not self.has_collection(new_name)
        self.collections[new_name] = self.collections.pop(old_name)

    def create_alias(self, collection_name, alias):
        self._set_alias(collection_name, alias)

    def alter_alias(self, collection_name, alias):
        assert alias in self.aliases
        self._set_alias(collection_name, alias)

    def _set_alias(self, target, alias):
        if self.alias_error == "before":
            raise RuntimeError("alias failed")
        self.aliases[alias] = target
        if self.alias_error == "after":
            raise RuntimeError("alias response timed out")

    def list_aliases(self, collection_name):
        aliases = [alias for alias, target in self.aliases.items() if target == collection_name]
        return {"aliases": aliases, "collection_name": collection_name} if self.wrapped_aliases else aliases

    def drop_collection(self, name):
        assert name not in self.aliases.values(), "Cannot remove a published index"
        del self.collections[name]

    def close(self):
        self.closed = True


def build_collection(client, name):
    client.create_collection(name)


@pytest.mark.parametrize("legacy", [True, False])
def test_ready_version_replaces_public_name_and_preserves_old_version(legacy):
    client = FakeMilvus()
    old_name = "papers" if legacy else "papers_previous"
    client.collections[old_name] = 7
    if not legacy:
        client.aliases["papers"] = old_name

    with staged_collection(client, "papers", build_collection, 2) as staging:
        client.insert(staging, [{}, {}])
        assert client.query("papers")[0]["count(*)"] == 7

    assert client.aliases["papers"] == staging
    assert client.query("papers")[0]["count(*)"] == 2
    assert 7 in client.collections.values()


def test_first_build_publishes_a_new_alias():
    client = FakeMilvus()
    with staged_collection(client, "papers", build_collection, 1) as staging:
        client.insert(staging, [{}])
    assert client.aliases["papers"] == staging


def test_failed_build_cleans_staging_with_wrapped_alias_response():
    client = FakeMilvus()
    client.collections["papers"] = 7
    client.wrapped_aliases = True
    with pytest.raises(RuntimeError, match="writer failed"):
        with staged_collection(client, "papers", build_collection, 1):
            raise RuntimeError("writer failed")
    assert client.collections == {"papers": 7}


@pytest.mark.parametrize("failure", ["insert", "flush", "load", "count"])
def test_build_failure_keeps_original_index_and_cleans_unpublished_version(failure):
    client = FakeMilvus()
    client.collections["papers"] = 7
    client.insert_error = failure == "insert"
    client.validation_error = failure

    with pytest.raises(RuntimeError):
        with staged_collection(client, "papers", build_collection, 2) as staging:
            client.insert(staging, [{}] if failure == "count" else [{}, {}])

    assert client.collections == {"papers": 7}
    assert not client.aliases


@pytest.mark.parametrize("legacy", [True, False])
def test_publish_failure_restores_or_keeps_old_public_index(legacy):
    client = FakeMilvus()
    old_name = "papers" if legacy else "papers_previous"
    client.collections[old_name] = 7
    if not legacy:
        client.aliases["papers"] = old_name
    client.alias_error = "before"

    with pytest.raises(RuntimeError, match="alias failed"):
        with staged_collection(client, "papers", build_collection, 1) as staging:
            client.insert(staging, [{}])

    assert client.collections == {old_name: 7}
    assert client.query("papers")[0]["count(*)"] == 7


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("wrapped_aliases", [True, False])
def test_publish_timeout_does_not_delete_an_already_active_version(legacy, wrapped_aliases):
    client = FakeMilvus()
    old_name = "papers" if legacy else "papers_previous"
    client.collections[old_name] = 7
    client.wrapped_aliases = wrapped_aliases
    if not legacy:
        client.aliases["papers"] = old_name
    client.alias_error = "after"

    with pytest.raises(RuntimeError, match="timed out"):
        with staged_collection(client, "papers", build_collection, 1) as staging:
            client.insert(staging, [{}])

    assert client.aliases["papers"] == staging
    assert client.query("papers")[0]["count(*)"] == 1
    assert 7 in client.collections.values()


def test_embedding_count_mismatch_never_opens_milvus(monkeypatch):
    monkeypatch.setattr(ingestion_service, "build_chunks_from_pdfs", lambda _: [{"text": "one"}])
    monkeypatch.setattr(ingestion_service, "embed_texts", lambda _: [])
    create_client = Mock()
    monkeypatch.setattr(ingestion_service, "create_milvus_client", create_client)

    with pytest.raises(RuntimeError, match="Embedding count"):
        ingestion_service.rebuild_knowledge_base()

    create_client.assert_not_called()


def test_image_insert_failure_keeps_original_image_index(monkeypatch):
    client = FakeMilvus()
    client.collections[config.IMAGE_COLLECTION_NAME] = 7
    client.insert_error = True
    monkeypatch.setattr(image_ingest, "save_images_from_pdfs", lambda *_: [{"image_path": "test.png"}])
    monkeypatch.setattr(image_ingest, "embed_image", lambda _: [0.0])
    monkeypatch.setattr(image_ingest, "build_image_collection", build_collection)

    with pytest.raises(RuntimeError, match="insert failed"):
        image_ingest.rebuild_image_index(client)

    assert client.collections == {config.IMAGE_COLLECTION_NAME: 7}
    assert not client.closed


def test_text_can_finish_while_failed_image_rebuild_retains_old_images(monkeypatch):
    client = FakeMilvus()
    client.collections = {config.COLLECTION_NAME: 7, config.IMAGE_COLLECTION_NAME: 3}
    monkeypatch.setattr(ingestion_service, "build_chunks_from_pdfs", lambda _: [{"text": "new"}])
    monkeypatch.setattr(ingestion_service, "embed_texts", lambda _: [[0.0]])
    monkeypatch.setattr(ingestion_service, "create_milvus_client", lambda: client)
    monkeypatch.setattr(ingestion_service, "build_collection", build_collection)
    monkeypatch.setattr(ingestion_service, "rebuild_image_index", Mock(side_effect=RuntimeError("image failed")))

    result = ingestion_service.rebuild_knowledge_base()

    assert result == {"text_chunks": 1, "images": 0, "image_error": "image failed"}
    assert client.query(config.COLLECTION_NAME)[0]["count(*)"] == 1
    assert client.collections[config.IMAGE_COLLECTION_NAME] == 3
    assert client.closed


def test_optional_graph_build_uses_published_physical_text_version(monkeypatch):
    client = FakeMilvus()
    records = [{"id": "chunk-1", "text": "new", "source": "paper.pdf", "page": 1}]
    monkeypatch.setattr(config, "ENABLE_GRAPH_BUILD", True)
    monkeypatch.setattr(ingestion_service.database, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr(ingestion_service, "build_chunks_from_pdfs", lambda _: records)
    monkeypatch.setattr(ingestion_service, "embed_texts", lambda _: [[0.0]])
    monkeypatch.setattr(ingestion_service, "create_milvus_client", lambda: client)
    monkeypatch.setattr(ingestion_service, "build_collection", build_collection)
    monkeypatch.setattr(ingestion_service, "rebuild_image_index", lambda _: [])
    graph_build = Mock(return_value={"build_id": "graph-1", "nodes": 2, "edges": 1})
    monkeypatch.setattr(ingestion_service, "build_graph_index", graph_build)

    result = ingestion_service.rebuild_knowledge_base()

    physical = client.aliases[config.COLLECTION_NAME]
    assert graph_build.call_args.args == (records, physical, "postgres://test")
    assert result["graph"]["edges"] == 1
    assert result["graph_error"] is None


def test_incomplete_migration_preserves_target_index():
    source = Mock()
    source.get_collection_stats.return_value = {"row_count": 3}
    iterator = source.query_iterator.return_value
    iterator.next.side_effect = [[{}, {}], []]
    target = FakeMilvus()
    target.collections["papers"] = 7

    with pytest.raises(RuntimeError, match="row count mismatch"):
        _copy_collection(source, target, "papers", [], build_collection, 10)

    assert target.collections == {"papers": 7}
    iterator.close.assert_called_once()
