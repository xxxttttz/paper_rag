"""HTTP client used by the Streamlit UI to call the Paper RAG API."""

from __future__ import annotations

from typing import Any

import httpx

import config


class PaperRAGAPIError(RuntimeError):
    """Represent an API response error or a connection failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _request_json(method: str, path: str) -> dict:
    url = f"{config.API_BASE_URL}{path}"
    try:
        response = httpx.request(
            method,
            url,
            timeout=config.API_REQUEST_TIMEOUT,
        )
    except httpx.RequestError as error:
        raise PaperRAGAPIError(
            f"无法连接 FastAPI 服务：{config.API_BASE_URL}"
        ) from error

    if response.is_error:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text or None
        message = detail if isinstance(detail, str) else f"API 请求失败（HTTP {response.status_code}）"
        raise PaperRAGAPIError(
            message,
            status_code=response.status_code,
            detail=detail,
        )

    return response.json()


def create_ingestion_job() -> dict:
    return _request_json("POST", "/api/v1/ingestion/jobs")


def get_ingestion_job(job_id: str) -> dict:
    return _request_json("GET", f"/api/v1/ingestion/jobs/{job_id}")
