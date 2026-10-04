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


def _request_json(
    method: str, path: str, payload: dict | None = None, *, timeout: float | None = None,
) -> dict:
    url = f"{config.API_BASE_URL}{path}"
    try:
        response = httpx.request(
            method,
            url,
            timeout=timeout if timeout is not None else config.API_REQUEST_TIMEOUT,
            **({"json": payload} if payload is not None else {}),
        )
    except httpx.ReadTimeout as error:
        raise PaperRAGAPIError("请求超时，服务可能仍在处理。请稍后刷新会话记录，确认结果后再重试。") from error
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


def create_comparison(conversation_id: str, sources: list[str], requirements: str) -> dict:
    return _request_json("POST", "/api/v1/comparisons", {
        "conversation_id": conversation_id, "sources": sources, "requirements": requirements,
    }, timeout=config.COMPARISON_REQUEST_TIMEOUT)
