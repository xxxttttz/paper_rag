"""Comparison facts, provenance and history verified without paid model calls."""

from unittest.mock import patch

import pytest

import comparison_service as service
import database
from comparison_models import ComparisonDraft, ComparisonRequest
from retriever import ComparisonIndexError


SOURCES = ["Alpha.pdf", "Beta.pdf"]
CHUNKS = [
    {"source": SOURCES[0], "page": 2, "text": "We train a decision tree. Dataset A uses recall.", "score": 0.9},
    {"source": SOURCES[1], "page": 3, "text": "We train XGBoost. Dataset B uses accuracy.", "score": 0.8},
]


def request(conversation_id="test"):
    return ComparisonRequest(conversation_id=conversation_id, sources=SOURCES, requirements="部署在交换机，关注吞吐与延迟")


def draft():
    return ComparisonDraft.model_validate({
        "cells": [
            {"source": SOURCES[0], "dimension": "method", "text": "使用决策树", "evidence": [{"evidence_id": "E1", "quote": "We train a decision tree."}]},
            {"source": SOURCES[1], "dimension": "method", "text": "使用 XGBoost", "evidence": [{"evidence_id": "E2", "quote": "We train XGBoost."}]},
        ],
    })


def test_invalid_source_quote_or_id_cannot_be_displayed_as_a_fact():
    invalid = ComparisonDraft.model_validate({"cells": [
        {"source": SOURCES[0], "dimension": "data", "text": "wrong source", "evidence": [{"evidence_id": "E2", "quote": "We train XGBoost."}]},
        {"source": SOURCES[0], "dimension": "deployment", "text": "invented fact", "evidence": [{"evidence_id": "E1", "quote": "Runs at 100 Gbps"}]},
        {"source": SOURCES[1], "dimension": "method", "text": "unknown ID", "evidence": [{"evidence_id": "E99", "quote": "anything"}]},
    ]})
    report = service.validate_report(request(), invalid, CHUNKS)
    assert len(report.cells) == 10
    assert all(cell.text == "资料不足" and not cell.evidence for cell in report.cells)
    rendered = service.render_report(report, CHUNKS)
    assert "invented fact" not in rendered
    assert "wrong source" not in rendered


def test_comparability_requires_both_papers_and_evaluation_facts():
    model = draft()
    model.comparability.status = "comparable"
    model.comparability.explanation = "不能仅凭两份分类器说明证明实验可比"
    model.comparability.evidence = [cell.evidence[0] for cell in model.cells]
    report = service.validate_report(request(), model, CHUNKS)
    assert report.comparability.status == "unknown"

    # With documented evaluation facts on both sides, the cited explanation survives.
    model.cells.extend(ComparisonDraft.model_validate({"cells": [
        {"source": SOURCES[0], "dimension": "evaluation", "text": "A 数据集，recall", "evidence": [{"evidence_id": "E1", "quote": "Dataset A uses recall."}]},
        {"source": SOURCES[1], "dimension": "evaluation", "text": "B 数据集，accuracy", "evidence": [{"evidence_id": "E2", "quote": "Dataset B uses accuracy."}]},
    ]}).cells)
    model.comparability.status = "not_comparable"
    model.comparability.explanation = "数据集和指标不同，不能直接按数值排序。"
    report = service.validate_report(request(), model, CHUNKS)
    assert report.comparability.status == "not_comparable"


def test_api_comparison_persists_report_and_evidence(client):
    conversation_id = database.create_conversation()
    with (
        patch.object(service, "retrieve_comparison_evidence", return_value=CHUNKS) as retrieval,
        patch.object(service, "generate_comparison", return_value=draft()) as model,
    ):
        response = client.post("/api/v1/comparisons", json=request(conversation_id).model_dump())
    assert response.status_code == 200
    result = response.json()
    assert len(result["report"]["cells"]) == 10
    assert "第2页; E1" in result["answer"] and "第3页; E2" in result["answer"]
    assert "资料不足" in result["answer"]
    assert retrieval.call_args.args[1] == SOURCES
    model.assert_called_once()
    assert [item["role"] for item in database.get_messages(conversation_id)] == ["user", "assistant"]
    assert database.get_messages(conversation_id)[-1]["content"] == result["answer"]
    assert [item["source"] for item in database.get_citations(result["assistant_message_id"])] == SOURCES
    database.delete_conversation(conversation_id)
    assert not database.get_citations(result["assistant_message_id"])


@pytest.mark.parametrize("sources", [["Alpha.pdf"], ["Alpha.pdf", "Alpha.pdf"], ["A", "B", "C", "D"]])
def test_api_rejects_invalid_selection(client, sources):
    with patch.object(service, "compare") as compare:
        response = client.post("/api/v1/comparisons", json={
            "conversation_id": "test", "sources": sources, "requirements": "some need",
        })
    assert response.status_code == 422
    compare.assert_not_called()


def test_missing_index_and_model_failure_are_reported_without_fake_answer(client):
    conversation_id = database.create_conversation()
    with patch.object(service, "retrieve_comparison_evidence", side_effect=ComparisonIndexError("请构建索引")):
        response = client.post("/api/v1/comparisons", json=request(conversation_id).model_dump())
    assert response.status_code == 409
    assert all(item["role"] == "user" for item in database.get_messages(conversation_id))
    with (
        patch.object(service, "retrieve_comparison_evidence", return_value=CHUNKS),
        patch.object(service, "generate_comparison", side_effect=ValueError("invalid JSON")),
    ):
        response = client.post("/api/v1/comparisons", json=request(conversation_id).model_dump())
    assert response.status_code == 502
    assert all(item["role"] == "user" for item in database.get_messages(conversation_id))
    assert client.post("/api/v1/comparisons", json=request("missing").model_dump()).status_code == 404
