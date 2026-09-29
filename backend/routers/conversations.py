"""Conversation and message HTTP endpoints."""

from fastapi import APIRouter, HTTPException, status

import database
from backend.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(payload: ConversationCreate) -> dict:
    conversation_id = database.create_conversation(payload.title)
    return database.get_conversation(conversation_id)


@router.get("", response_model=list[ConversationResponse])
def list_conversations() -> list[dict]:
    return database.list_conversations()


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
) -> dict:
    if database.get_conversation(conversation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    try:
        database.rename_conversation(conversation_id, payload.title)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return database.get_conversation(conversation_id)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(conversation_id: str) -> None:
    if database.get_conversation(conversation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    database.delete_conversation(conversation_id)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(conversation_id: str) -> list[dict]:
    if database.get_conversation(conversation_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return database.get_messages(conversation_id)
