"""RAG chat HTTP endpoint."""

from fastapi import APIRouter, HTTPException, status

import chat_service
import database
from backend.schemas import ChatRequest, ChatResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def ask_question(payload: ChatRequest) -> dict:
    """Run one persisted RAG turn for an existing conversation."""
    if database.get_conversation(payload.conversation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    try:
        return chat_service.ask(payload.conversation_id, payload.question)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAG pipeline failed",
        ) from error
