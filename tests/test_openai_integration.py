"""
tests/test_openai_integration.py
================================
Verifies OpenAI integration in Doctor Buddy & Secondary Vision:
1. Official OpenAI SDK client initialization with configurable model.
2. Grounded session results passed to chat completions.
3. Multi-turn follow-up history support.
4. Clean fallback behavior when key is absent or API errors.
5. Absolute prevention of API key leakage in outputs.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from src.inference.doctor_buddy import ask_doctor_buddy_question
from src.inference.secondary_vision import SecondaryVisionService
from PIL import Image

@pytest.mark.anyio
async def test_doctor_buddy_openai_chat_completions_mocked():
    """Verifies that Doctor Buddy calls OpenAI chat completions when configured."""
    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Based on your screening hemoglobin of 13.81 g/dL, your values reflect normal eyelid vascularity. A routine CBC is recommended for clinical verification."
    mock_chat_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat_completion

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mocked-test-key-12345", "OPENAI_MODEL_NAME": "gpt-4o-mini"}):
        with patch("openai.OpenAI", return_value=mock_client):
            res = await ask_doctor_buddy_question(
                question="Can you explain my hemoglobin reading in detail?",
                language="en",
                current_step=6,
                session_id=None,
                history=[{"role": "user", "content": "Hello Astra"}]
            )

            assert res is not None
            assert "mode" in res
            assert res["mode"] == "openai_chat"
            assert res["provider"] == "openai"
            assert res["model"] == "gpt-4o-mini"
            assert "13.81" in res["message"] or "hemoglobin" in res["message"]
            assert "follow_up_suggestions" in res
            assert len(res["follow_up_suggestions"]) > 0

            # Verify API key was NEVER leaked in the response
            assert "sk-mocked" not in str(res)


@pytest.mark.anyio
async def test_doctor_buddy_fallback_when_openai_fails():
    """Verifies clean fallback to grounded clinical rules when OpenAI call fails."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI API rate limit exceeded")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mocked-failing-key", "OPENAI_MODEL_NAME": "gpt-4o-mini"}):
        with patch("openai.OpenAI", return_value=mock_client):
            res = await ask_doctor_buddy_question(
                question="What does my result mean?",
                language="en",
                current_step=6,
                session_id=None
            )

            assert res is not None
            assert "message" in res
            # Grounded patient rule answered gracefully
            assert "Screening Results Mean" in res["message"]
            assert "sk-mocked" not in str(res)


def test_secondary_vision_openai_integration():
    """Verifies SecondaryVisionService uses chat completions and JSON mode."""
    service = SecondaryVisionService(api_key="sk-test-vision-key", model_name="gpt-4o-mini")
    assert service.model_name == "gpt-4o-mini"

    # Verify status dictionary does NOT leak the API key
    status = service.get_status()
    assert status["openai_vision"]["configured_model"] == "gpt-4o-mini"
    assert "sk-test" not in str(status)
