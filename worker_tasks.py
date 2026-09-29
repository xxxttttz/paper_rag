"""Functions executed by RQ workers."""

from __future__ import annotations

from redis.exceptions import LockError
from rq import get_current_job

import config
from ingestion_service import rebuild_knowledge_base
from job_queue import redis_connection


REBUILD_LOCK_NAME = "paper-rag:ingestion:rebuild-lock"


def rebuild_knowledge_base_job() -> dict:
    """Run one mutually exclusive knowledge-base rebuild."""
    connection = redis_connection()
    lock = connection.lock(
        REBUILD_LOCK_NAME,
        timeout=config.INGESTION_JOB_TIMEOUT + 300,
        blocking=False,
    )
    if not lock.acquire(blocking=False):
        raise RuntimeError("Another knowledge-base rebuild is already running")

    job = get_current_job()

    def update_progress(stage: str, details: dict) -> None:
        if job is None:
            return
        job.meta.update({"stage": stage, **details})
        job.save_meta()

    try:
        update_progress("started", {})
        return rebuild_knowledge_base(progress=update_progress)
    finally:
        try:
            lock.release()
        except LockError:
            pass
