"""Tests for asynchronous knowledge-base ingestion endpoints."""

from unittest.mock import Mock, patch

from redis.exceptions import ConnectionError as RedisConnectionError
from rq.exceptions import NoSuchJobError

from job_queue import ActiveIngestionJobError


SERIALIZED_JOB = {
    "job_id": "job-123",
    "status": "queued",
    "stage": "queued",
    "result": None,
    "error": None,
    "enqueued_at": None,
    "started_at": None,
    "ended_at": None,
}


def test_create_ingestion_job(client):
    with (
        patch(
            "backend.routers.ingestion.glob.glob",
            return_value=["paper.pdf"],
        ),
        patch(
            "backend.routers.ingestion.enqueue_rebuild",
            return_value=Mock(),
        ),
        patch(
            "backend.routers.ingestion.serialize_job",
            return_value=SERIALIZED_JOB,
        ),
    ):
        response = client.post("/api/v1/ingestion/jobs")

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-123"
    assert response.json()["stage"] == "queued"


def test_create_ingestion_job_requires_pdf(client):
    with patch("backend.routers.ingestion.glob.glob", return_value=[]):
        response = client.post("/api/v1/ingestion/jobs")

    assert response.status_code == 400
    assert response.json()["detail"] == "No PDF files are available for ingestion"


def test_create_ingestion_job_rejects_duplicate(client):
    with (
        patch(
            "backend.routers.ingestion.glob.glob",
            return_value=["paper.pdf"],
        ),
        patch(
            "backend.routers.ingestion.enqueue_rebuild",
            side_effect=ActiveIngestionJobError("active-job"),
        ),
    ):
        response = client.post("/api/v1/ingestion/jobs")

    assert response.status_code == 409
    assert response.json()["detail"]["job_id"] == "active-job"


def test_get_ingestion_job(client):
    with (
        patch("backend.routers.ingestion.get_job", return_value=Mock()),
        patch(
            "backend.routers.ingestion.serialize_job",
            return_value=SERIALIZED_JOB,
        ),
    ):
        response = client.get("/api/v1/ingestion/jobs/job-123")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_get_missing_ingestion_job(client):
    with patch(
        "backend.routers.ingestion.get_job",
        side_effect=NoSuchJobError,
    ):
        response = client.get("/api/v1/ingestion/jobs/missing")

    assert response.status_code == 404


def test_ingestion_returns_503_when_redis_is_unavailable(client):
    with (
        patch(
            "backend.routers.ingestion.glob.glob",
            return_value=["paper.pdf"],
        ),
        patch(
            "backend.routers.ingestion.enqueue_rebuild",
            side_effect=RedisConnectionError("unavailable"),
        ),
    ):
        response = client.post("/api/v1/ingestion/jobs")

    assert response.status_code == 503
