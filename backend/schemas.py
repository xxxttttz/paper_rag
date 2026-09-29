"""Pydantic request and response contracts for the public API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=100)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=10_000)


class TextCitationResponse(BaseModel):
    source: str
    page: int
    text: str
    score: float | None = None
    retrieval_score: float | None = None


class ImageCitationResponse(BaseModel):
    image_path: str
    source: str
    page: int
    score: float | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    citations: list[TextCitationResponse]
    image_citations: list[ImageCitationResponse]
    retrieval_query: str


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    result: dict | None = None
    error: str | None = None
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
