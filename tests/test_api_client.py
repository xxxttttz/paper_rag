"""Tests for the Streamlit-to-FastAPI HTTP client."""

from unittest.mock import patch

import httpx
import pytest

import api_client


def _response(status_code: int, json: dict) -> httpx.Response:
    request = httpx.Request("GET", "http://test/api")
    return httpx.Response(status_code, json=json, request=request)


def test_create_ingestion_job():
    payload = {"job_id": "job-123", "status": "queued"}
    with patch("api_client.httpx.request", return_value=_response(202, payload)) as request:
        result = api_client.create_ingestion_job()

    assert result == payload
    assert request.call_args.args[:2] == (
        "POST",
        f"{api_client.config.API_BASE_URL}/api/v1/ingestion/jobs",
    )


def test_api_error_preserves_conflicting_job_id():
    detail = {"message": "already active", "job_id": "active-job"}
    with patch(
        "api_client.httpx.request",
        return_value=_response(409, {"detail": detail}),
    ):
        with pytest.raises(api_client.PaperRAGAPIError) as captured:
            api_client.create_ingestion_job()

    assert captured.value.status_code == 409
    assert captured.value.detail["job_id"] == "active-job"


def test_api_connection_error_has_actionable_message():
    request = httpx.Request("GET", "http://test/api")
    with patch(
        "api_client.httpx.request",
        side_effect=httpx.ConnectError("unavailable", request=request),
    ):
        with pytest.raises(api_client.PaperRAGAPIError, match="FastAPI"):
            api_client.get_ingestion_job("job-123")
