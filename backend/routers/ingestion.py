"""Asynchronous knowledge-base ingestion endpoints."""

import glob
import os

from fastapi import APIRouter, HTTPException, status
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError

import config
from backend.schemas import IngestionJobResponse
from job_queue import (
    ActiveIngestionJobError,
    enqueue_rebuild,
    get_job,
    serialize_job,
)


router = APIRouter(prefix="/ingestion/jobs", tags=["ingestion"])


@router.post("", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_ingestion_job() -> dict:
    if not glob.glob(os.path.join(config.PDF_DIR, "*.pdf")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF files are available for ingestion",
        )
    try:
        return serialize_job(enqueue_rebuild())
    except ActiveIngestionJobError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "A knowledge-base rebuild is already active",
                "job_id": error.job_id,
            },
        ) from error
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis queue is unavailable",
        ) from error


@router.get("/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(job_id: str) -> dict:
    try:
        return serialize_job(get_job(job_id))
    except NoSuchJobError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found",
        ) from error
    except RedisError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis queue is unavailable",
        ) from error
