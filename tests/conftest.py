"""Shared fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient

import database
from backend.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Run each test against an isolated SQLite database."""
    monkeypatch.setattr(database, "DATABASE_URL", "")
    monkeypatch.setattr(database, "CHAT_DB_PATH", tmp_path / "chat_history.db")
    with TestClient(app) as test_client:
        yield test_client
