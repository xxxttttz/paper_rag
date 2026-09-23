"""Integration tests for the public HTTP API."""

from unittest.mock import patch

import chat_service
import database


def _create_conversation(client, title="API test") -> str:
    response = client.post("/api/v1/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()["id"]


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "paper-rag-api"}


def test_conversation_lifecycle(client):
    conversation_id = _create_conversation(client, "Initial title")

    listed = client.get("/api/v1/conversations")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == conversation_id

    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "Renamed title"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed title"

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    ).status_code == 404


def test_missing_conversation_returns_404(client):
    assert client.patch(
        "/api/v1/conversations/missing",
        json={"title": "New title"},
    ).status_code == 404
    assert client.delete("/api/v1/conversations/missing").status_code == 404
    assert client.post(
        "/api/v1/chat",
        json={"conversation_id": "missing", "question": "hello"},
    ).status_code == 404


def test_chat_persists_messages_and_citations(client):
    conversation_id = _create_conversation(client)
    text_hits = [{
        "text": "Transformer uses self-attention.",
        "source": "paper.pdf",
        "page": 3,
        "score": 0.93,
        "retrieval_score": 0.81,
    }]
    image_hits = [{
        "image_path": "data/images/paper/page_3.png",
        "source": "paper.pdf",
        "page": 3,
        "score": 0.88,
    }]

    with (
        patch.object(chat_service, "rewrite_query", return_value="attention figure"),
        patch.object(chat_service, "retrieve", return_value=text_hits),
        patch.object(chat_service, "retrieve_images", return_value=image_hits),
        patch.object(
            chat_service,
            "generate_answer",
            return_value="It uses self-attention.",
        ),
    ):
        response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "question": "Explain the attention figure",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It uses self-attention."
    assert body["citations"][0]["source"] == "paper.pdf"
    assert body["image_citations"][0]["page"] == 3

    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    )
    assert [message["role"] for message in messages.json()] == [
        "user",
        "assistant",
    ]
    assert database.get_citations(body["assistant_message_id"])[0]["page"] == 3
    assert database.get_image_citations(
        body["assistant_message_id"]
    )[0]["page"] == 3
