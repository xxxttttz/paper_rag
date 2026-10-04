"""Graph extraction rejects unsupported or non-verbatim evidence."""

from unittest.mock import Mock

import pytest

import graph_builder


def test_parse_relations_requires_exact_quote_and_supported_types():
    text = "DeviceRadar uses packet size and direction for device recognition."
    payload = {"relations": [
        {
            "source": {"type": "paper", "name": "DeviceRadar"},
            "type": "HAS_FEATURE",
            "target": {"type": "feature", "name": "packet size"},
            "quote": text,
        },
        {
            "source": {"type": "paper", "name": "DeviceRadar"},
            "type": "OUTPERFORMS",
            "target": {"type": "paper", "name": "Other"},
            "quote": "invented comparison",
        },
        {
            "source": {"type": "paper", "name": "DeviceRadar"},
            "type": "MADE_UP",
            "target": {"type": "feature", "name": "packet size"},
            "quote": text,
        },
    ]}
    parsed = graph_builder.parse_relations(payload, text)
    assert len(parsed) == 1
    assert parsed[0]["source"] == ("paper", "DeviceRadar")
    assert parsed[0]["quote"] == text


def test_graph_build_failure_never_publishes(monkeypatch):
    monkeypatch.setattr(graph_builder.graph_store, "begin_graph_build", lambda *_: "build-1")
    failed = Mock()
    ready = Mock()
    monkeypatch.setattr(graph_builder.graph_store, "mark_graph_failed", failed)
    monkeypatch.setattr(graph_builder.graph_store, "mark_graph_ready", ready)
    monkeypatch.setattr(graph_builder, "extract_relations", Mock(side_effect=RuntimeError("model failed")))

    with pytest.raises(RuntimeError, match="model failed"):
        graph_builder.build_graph_index(
            [{"id": "chunk-1", "text": "text", "source": "paper.pdf", "page": 1}],
            "paper_chunks_version_test", "postgres://test",
        )
    failed.assert_called_once_with("postgres://test", "build-1", "model failed")
    ready.assert_not_called()
