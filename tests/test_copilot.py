"""
tests/test_copilot.py
---------------------
Unit tests for the Gemini Research Copilot integration.

Tests cover:
  1. Input sanitization (PII/path/clinical value redaction)
  2. Medical advice topic blocking
  3. Offline fallback responses
  4. Chat API endpoint contract
  5. Copilot status endpoint
"""

import os
import sys
import pytest

# Ensure workspace root is importable
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.inference.copilot import sanitize_input, is_blocked_topic, _get_offline_response


# ===========================================================================
# 1. Sanitization Tests — Ensure PII, paths, and clinical values are stripped
# ===========================================================================

class TestSanitization:
    """Verifies that sensitive data is redacted before sending to the LLM."""

    def test_windows_path_redacted(self):
        text = r"The image is at C:\Users\sarth\OneDrive\Pictures\eye_scan.jpg"
        result = sanitize_input(text)
        assert r"C:\Users" not in result
        assert "[REDACTED_PATH]" in result

    def test_unix_path_redacted(self):
        text = "File located at /home/user/datasets/anemia/patient_001.png"
        result = sanitize_input(text)
        assert "/home/user" not in result
        assert "[REDACTED_PATH]" in result

    def test_hemoglobin_value_redacted(self):
        text = "The patient has hemoglobin of 12.5 g/dL which seems low"
        result = sanitize_input(text)
        assert "12.5 g/dL" not in result
        assert "[REDACTED_HB_VALUE]" in result

    def test_hemoglobin_variant_redacted(self):
        text = "Hb level is 8.2g/dl"
        result = sanitize_input(text)
        assert "8.2g/dl" not in result
        assert "[REDACTED_HB_VALUE]" in result

    def test_bilirubin_value_redacted(self):
        text = "Total bilirubin was 3.2 mg/dL"
        result = sanitize_input(text)
        assert "3.2 mg/dL" not in result
        assert "[REDACTED_BILIRUBIN_VALUE]" in result

    def test_phone_number_redacted(self):
        text = "Contact the patient at 9019488301"
        result = sanitize_input(text)
        assert "9019488301" not in result
        assert "[REDACTED_PHONE]" in result

    def test_email_redacted(self):
        text = "Send results to patient@hospital.com"
        result = sanitize_input(text)
        assert "patient@hospital.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_base64_image_data_redacted(self):
        text = "Here is the image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAA"
        result = sanitize_input(text)
        assert "base64" not in result
        assert "[REDACTED_IMAGE_DATA]" in result

    def test_patient_id_redacted(self):
        text = "patient_id: PAT-2024-001 needs review"
        result = sanitize_input(text)
        assert "PAT-2024-001" not in result
        assert "[REDACTED_PATIENT_ID]" in result

    def test_clean_research_question_unchanged(self):
        text = "What CNN architecture is used for the anemia model?"
        result = sanitize_input(text)
        assert result == text

    def test_mixed_content_partial_redaction(self):
        text = "The ResNet-50 model at C:\\Models\\best.pth has MAE of 1.5"
        result = sanitize_input(text)
        assert "ResNet-50" in result
        assert r"C:\Models" not in result


# ===========================================================================
# 2. Topic Blocking Tests — Medical advice queries must be rejected
# ===========================================================================

class TestTopicBlocking:
    """Ensures medical advice queries are properly identified and blocked."""

    def test_diagnosis_blocked(self):
        assert is_blocked_topic("Can you diagnose my patient's condition?")

    def test_treatment_blocked(self):
        assert is_blocked_topic("Should the patient take iron supplements?")

    def test_medication_blocked(self):
        assert is_blocked_topic("What medication should they use for the patient?")

    def test_research_question_not_blocked(self):
        assert not is_blocked_topic("What is the ResNet-50 architecture?")

    def test_dataset_question_not_blocked(self):
        assert not is_blocked_topic("How many images are in the anemia dataset?")

    def test_model_metrics_not_blocked(self):
        assert not is_blocked_topic("What is the test MAE for the anemia regressor?")

    def test_code_help_not_blocked(self):
        assert not is_blocked_topic("How do I fix this PyTorch DataLoader error?")


# ===========================================================================
# 3. Offline Fallback Tests — Keyword-based responses
# ===========================================================================

class TestOfflineFallback:
    """Verifies offline fallback responses are properly categorized."""

    def test_architecture_keyword_match(self):
        result = _get_offline_response("What CNN architecture is used?")
        assert "ResNet-50" in result
        assert "MobileNetV3" in result

    def test_metrics_keyword_match(self):
        result = _get_offline_response("What is the test MAE?")
        assert "MAE" in result
        assert "1.50" in result

    def test_dataset_keyword_match(self):
        result = _get_offline_response("Tell me about the dataset structure")
        assert "India.xlsx" in result or "Dataset" in result

    def test_default_fallback(self):
        result = _get_offline_response("random unrelated question xyz123")
        assert "offline mode" in result.lower()

    def test_resnet_keyword(self):
        result = _get_offline_response("How does ResNet work here?")
        assert "ResNet-50" in result

    def test_mobilenet_keyword(self):
        result = _get_offline_response("Tell me about MobileNet")
        assert "MobileNetV3" in result


# ===========================================================================
# 4. API Endpoint Tests — FastAPI contract validation
# ===========================================================================

class TestChatEndpoint:
    """Tests the /api/chat and /api/copilot-status endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app import app
        return TestClient(app)

    def test_copilot_status_returns_200(self, client):
        res = client.get("/api/copilot-status")
        assert res.status_code == 200
        data = res.json()
        assert "available" in data
        assert "mode" in data
        assert "safety_gates" in data
        assert isinstance(data["safety_gates"], list)
        assert len(data["safety_gates"]) > 0

    def test_chat_requires_message(self, client):
        res = client.post("/api/chat", json={})
        assert res.status_code == 422  # Pydantic validation error

    def test_chat_rejects_empty_message(self, client):
        res = client.post("/api/chat", json={"message": ""})
        assert res.status_code == 422  # min_length=1 constraint

    def test_chat_returns_structured_response(self, client):
        res = client.post("/api/chat", json={"message": "What model architectures are used?"})
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert "mode" in data
        assert "sanitized" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_blocks_medical_advice(self, client):
        res = client.post("/api/chat", json={"message": "Can you diagnose my patient's condition?"})
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "blocked"
        assert "cannot provide medical" in data["response"].lower() or "cannot" in data["response"].lower()

    def test_chat_sanitizes_sensitive_input(self, client):
        msg = r"Tell me about the model at C:\Users\sarth\models\best.pth with Hb 12.5 g/dL"
        res = client.post("/api/chat", json={"message": msg})
        assert res.status_code == 200
        data = res.json()
        assert data["sanitized"] is True

    def test_chat_with_history(self, client):
        history = [
            {"role": "user", "content": "What is ResNet?"},
            {"role": "model", "content": "ResNet is a deep residual network."}
        ]
        res = client.post("/api/chat", json={
            "message": "How is it used in this project?",
            "history": history
        })
        assert res.status_code == 200
        data = res.json()
        assert "response" in data

    def test_chat_rejects_oversized_message(self, client):
        long_msg = "x" * 2001  # Over 2000 char limit
        res = client.post("/api/chat", json={"message": long_msg})
        assert res.status_code == 422  # max_length=2000 constraint
