"""Application service that coordinates chat persistence and the RAG pipeline."""

from __future__ import annotations

import database
import config
from generator import generate_answer, rewrite_query
from retriever import retrieve


DEFAULT_TITLE = "新会话"
TITLE_MAX_LENGTH = config.CONVERSATION_TITLE_LENGTH


def initialize() -> None:
    """Initialize storage required by the chat application."""
    database.init_database()


def create_conversation() -> str:
    """Create an empty conversation and return its ID."""
    return database.create_conversation(DEFAULT_TITLE)


def _build_title(question: str) -> str:
    """Use the first question as a simple, deterministic conversation title."""
    normalized = " ".join(question.split())
    if len(normalized) <= TITLE_MAX_LENGTH:
        return normalized
    return normalized[:TITLE_MAX_LENGTH].rstrip() + "…"


def ask(conversation_id: str, question: str) -> dict:
    """
    Run one RAG turn and persist both messages and the answer citations.

    The user message is saved before external API calls so an interrupted or
    failed request is not silently lost. Exceptions are allowed to reach the UI,
    which can present a retry action without storing a fabricated answer.
    """
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty")

    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise ValueError("Conversation does not exist")

    existing_messages = database.get_messages(conversation_id)
    recent_history = existing_messages[-config.MAX_HISTORY_MESSAGES :]
    user_message_id = database.add_message(
        conversation_id,
        "user",
        clean_question,
    )

    if not existing_messages and conversation["title"] == DEFAULT_TITLE:
        database.rename_conversation(
            conversation_id,
            _build_title(clean_question),
        )

    try:
        retrieval_query = rewrite_query(clean_question, recent_history)
    except Exception:
        # Query rewriting is an enhancement; a transient failure should not
        # prevent the original question from reaching the normal RAG flow.
        retrieval_query = clean_question

    chunks = retrieve(retrieval_query)
    if not chunks:
        answer = "没有检索到相关论文内容，请确认知识库已构建并包含相关资料。"
    else:
        answer = generate_answer(clean_question, chunks, recent_history)

    assistant_message_id = database.add_message(
        conversation_id,
        "assistant",
        answer,
    )
    database.add_citations(assistant_message_id, chunks)

    return {
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "answer": answer,
        "citations": chunks,
        "retrieval_query": retrieval_query,
    }
