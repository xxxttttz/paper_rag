"""Exercise the actual sidebar recovery behavior without model calls."""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import api_client
import config
import database


@pytest.fixture
def app_test(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_URL", "")
    monkeypatch.setattr(database, "CHAT_DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "test.pdf").touch()
    monkeypatch.setattr(config, "PDF_DIR", str(pdf_dir))
    database.init_database()
    database.create_conversation("UI regression test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["ingestion_job_id"] = "test-job"
    return app


def rebuild_button(app):
    return next(button for button in app.button if "重新构建" in button.label)


def test_missing_job_clears_active_state_and_restores_rebuild_button(app_test):
    with patch(
        "api_client.get_ingestion_job",
        side_effect=api_client.PaperRAGAPIError("not found", status_code=404),
    ):
        app_test.run(timeout=10)

    assert not app_test.exception
    assert "ingestion_job_id" not in app_test.session_state
    assert not rebuild_button(app_test).disabled
    assert any("任务记录已过期" in warning.value for warning in app_test.warning)


@pytest.mark.parametrize("status_code", [None, 503])
def test_transient_failure_retains_active_job_tracking(app_test, status_code):
    with patch(
        "api_client.get_ingestion_job",
        side_effect=api_client.PaperRAGAPIError("unavailable", status_code=status_code),
    ):
        app_test.run(timeout=10)

    assert not app_test.exception
    assert app_test.session_state["ingestion_job_id"] == "test-job"
    assert rebuild_button(app_test).disabled
    assert any("暂时无法获取" in warning.value for warning in app_test.warning)


def test_finished_job_still_shows_result_and_restores_button(app_test):
    job = {
        "job_id": "test-job",
        "status": "finished",
        "result": {"text_chunks": 2, "images": 1, "image_error": None},
    }
    with patch("api_client.get_ingestion_job", return_value=job):
        app_test.run(timeout=10)

    assert not app_test.exception
    assert not rebuild_button(app_test).disabled
    assert any("2 个文本块" in notice.value for notice in app_test.success)


def test_finished_job_reports_graph_build_failure(app_test):
    job = {
        "job_id": "test-job",
        "status": "finished",
        "result": {
            "text_chunks": 2, "images": 1, "image_error": None,
            "graph": None, "graph_error": "model failed",
        },
    }
    with patch("api_client.get_ingestion_job", return_value=job):
        app_test.run(timeout=10)

    assert not app_test.exception
    assert any("图索引未启用" in notice.value for notice in app_test.warning)
