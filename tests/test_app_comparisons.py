"""Exercise the comparison form, HTTP path and history rendering together."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import api_client
import comparison_service
import config
import database


def test_submit_comparison_and_reopen_history(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name in ("Alpha.pdf", "Beta.pdf"):
        (pdf_dir / name).touch()
    monkeypatch.setattr(config, "PDF_DIR", str(pdf_dir))
    conversation_id = database.create_conversation()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=10)
    assert not app.exception
    assert app.multiselect[0].value == ["Alpha.pdf", "Beta.pdf"]

    def call_api(conversation_id, sources, requirements):
        response = client.post("/api/v1/comparisons", json={
            "conversation_id": conversation_id, "sources": sources, "requirements": requirements,
        })
        response.raise_for_status()
        return response.json()

    with (
        patch.object(api_client, "create_comparison", side_effect=call_api) as submit,
        patch.object(comparison_service, "retrieve_comparison_evidence", return_value=[]),
        patch.object(comparison_service, "generate_comparison") as generate,
    ):
        next(button for button in app.button if button.label == "生成方案对比").click().run(timeout=10)
    assert not app.exception
    submit.assert_called_once()
    generate.assert_not_called()  # No evidence means no invented comparison or model request.
    assert any("# 方案对比报告" in item.value for item in app.markdown)
    assert any("资料不足" in item.value for item in app.markdown)
    assert len(database.get_messages(conversation_id)) == 2
    app.run(timeout=10)
    assert not app.exception
    assert len(database.get_messages(conversation_id)) == 2
    assert any("# 方案对比报告" in item.value for item in app.markdown)
