"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import database
from backend.routers.chat import router as chat_router
from backend.routers.comparisons import router as comparisons_router
from backend.routers.conversations import router as conversations_router
from backend.routers.ingestion import router as ingestion_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize application resources once when the API starts."""
    database.init_database()
    try:
        yield
    finally:
        database.close_connections()


class HealthResponse(BaseModel):
    """Stable response contract for health checks."""

    status: str
    service: str


app = FastAPI(
    title="Paper RAG API",
    description="Enterprise API for paper ingestion, retrieval, and chat.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(conversations_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(comparisons_router, prefix="/api/v1")
app.include_router(ingestion_router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Report whether the API process is ready to receive requests."""
    return HealthResponse(status="ok", service="paper-rag-api")
