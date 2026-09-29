"""Redis Queue integration for background ingestion jobs."""

from __future__ import annotations

from dataclasses import dataclass

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import StartedJobRegistry
from rq.serializers import JSONSerializer

import config


ENQUEUE_LOCK_NAME = "paper-rag:ingestion:enqueue-lock"
JOB_TYPE = "knowledge-base-rebuild"


@dataclass
class ActiveIngestionJobError(Exception):
    job_id: str


def redis_connection() -> Redis:
    return Redis.from_url(config.REDIS_URL)


def ingestion_queue(connection: Redis | None = None) -> Queue:
    return Queue(
        config.INGESTION_QUEUE_NAME,
        connection=connection or redis_connection(),
        serializer=JSONSerializer,
    )


def _active_job_id(queue: Queue) -> str | None:
    queued_ids = queue.get_job_ids()
    if queued_ids:
        return queued_ids[0]
    registry = StartedJobRegistry(queue=queue)
    started_ids = registry.get_job_ids()
    return started_ids[0] if started_ids else None


def enqueue_rebuild() -> Job:
    connection = redis_connection()
    queue = ingestion_queue(connection)
    lock = connection.lock(
        ENQUEUE_LOCK_NAME,
        timeout=10,
        blocking_timeout=2,
    )
    if not lock.acquire():
        raise RuntimeError("Could not acquire ingestion enqueue lock")
    try:
        active_job_id = _active_job_id(queue)
        if active_job_id:
            raise ActiveIngestionJobError(active_job_id)
        return queue.enqueue(
            "worker_tasks.rebuild_knowledge_base_job",
            job_timeout=config.INGESTION_JOB_TIMEOUT,
            result_ttl=config.INGESTION_RESULT_TTL,
            failure_ttl=config.INGESTION_RESULT_TTL,
            description="Rebuild the Paper RAG knowledge base",
            meta={"type": JOB_TYPE, "stage": "queued"},
        )
    finally:
        lock.release()


def get_job(job_id: str) -> Job:
    return Job.fetch(
        job_id,
        connection=redis_connection(),
        serializer=JSONSerializer,
    )


def serialize_job(job: Job) -> dict:
    status = job.get_status(refresh=True)
    status_value = getattr(status, "value", str(status))
    result = job.return_value(refresh=True) if status_value == "finished" else None
    error = None
    if status_value == "failed" and job.exc_info:
        error = job.exc_info.strip().splitlines()[-1][:500]
    return {
        "job_id": job.id,
        "status": status_value,
        "stage": job.meta.get("stage"),
        "result": result,
        "error": error,
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
    }
