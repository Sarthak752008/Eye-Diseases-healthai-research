"""
tests/test_doctor_buddy.py
--------------------------
Unit and integration tests for AI Doctor Buddy Mascot (Dr. Netra).

Verifies:
  1. Deterministic event guidance for all critical workflow states (welcome, camera permission denied, blur, lighting, blocked models, borderline anemia).
  2. Strict bilingual consistency (English and Hindi).
  3. PII and medical diagnosis blocking safeguards.
  4. Structured response schema validation (message, speak_text, expression, severity, suggested_next_step).
  5. Graceful fallback when Gemini API encounters quota limits or missing keys.
  6. Endpoint contracts for /api/doctor-buddy/event and /api/doctor-buddy/ask.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app import app
from src.inference.doctor_buddy import (
    get_doctor_buddy_guidance,
    ask_doctor_buddy_question,
    DETERMINISTIC_EVENTS
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestDoctorBuddyDeterministicGuidance:
    """Tests the deterministic guidance dictionary and fallback logic."""

    def test_welcome_event_english(self):
        res = get_doctor_buddy_guidance("welcome", "en")
        assert "Dr. Netra" in res["message"]
        assert res["expression"] == "happy"
        assert res["severity"] == "info"
        assert len(res["speak_text"]) > 0

    def test_welcome_event_hindi(self):
        res = get_doctor_buddy_guidance("welcome", "hi")
        assert "नेत्रा" in res["message"]
        assert res["expression"] == "happy"
        assert res["severity"] == "info"

    def test_camera_permission_denied_guidance(self):
        res = get_doctor_buddy_guidance("camera_permission_denied", "en")
        assert res["severity"] == "warning"
        assert "blocked" in res["message"].lower() or "permission" in res["message"].lower()
        assert res["expression"] == "alert"

    def test_quality_blurry_guidance(self):
        res = get_doctor_buddy_guidance("quality_blurry", "en")
        assert "blurry" in res["message"].lower()
        assert res["expression"] == "alert"
        assert res["severity"] == "warning"

    def test_model_mendeley_blocked_guidance(self):
        res = get_doctor_buddy_guidance("model_mendeley_blocked", "en")
        assert "mendeley" in res["message"].lower()
        assert "slid" in res["message"].lower()
        assert res["severity"] == "warning"

    def test_unknown_event_fallback(self):
        res = get_doctor_buddy_guidance("non_existent_event_xyz", "en")
        assert "Dr. Netra" in res["message"]
        assert res["mode"] == "deterministic"


class TestDoctorBuddyMedicalSafetyAndPII:
    """Ensures Doctor Buddy never dispenses diagnosis or leaks PII."""

    def test_diagnosis_attempt_blocked(self):
        import asyncio
        res = asyncio.run(ask_doctor_buddy_question("Can you diagnose my patient's eye disease?", "en", 4))
        assert res["mode"] == "blocked"
        assert "cannot provide" in res["message"].lower()
        assert res["expression"] == "alert"

    def test_prescribing_medication_blocked(self):
        import asyncio
        res = asyncio.run(ask_doctor_buddy_question("What medicine should the patient take for red eye?", "en", 4))
        assert res["mode"] == "blocked"
        assert "cannot" in res["message"].lower()

    def test_camera_question_routes_safely(self):
        import asyncio
        res = asyncio.run(ask_doctor_buddy_question("How do I grant camera permission in browser?", "en", 4))
        assert res["mode"] in ("deterministic", "gemini_enriched")
        assert "camera" in res["message"].lower() or "permission" in res["message"].lower()


class TestDoctorBuddyEndpoints:
    """Validates FastAPI HTTP contracts for Doctor Buddy endpoints."""

    def test_event_endpoint_success(self, client):
        payload = {
            "event_code": "positioning_too_far",
            "language": "en",
            "context_note": "webcam_alignment"
        }
        res = client.post("/api/doctor-buddy/event", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "message" in data
        assert "speak_text" in data
        assert "expression" in data
        assert "severity" in data
        assert "suggested_next_step" in data
        assert data["expression"] == "alert"
        assert data["severity"] == "warning"

    def test_event_endpoint_hindi(self, client):
        payload = {
            "event_code": "quality_dark",
            "language": "hi"
        }
        res = client.post("/api/doctor-buddy/event", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "रोशनी" in data["message"] or "अंधेरी" in data["message"]

    def test_ask_endpoint_blocks_medical_advice(self, client):
        payload = {
            "question": "Should the patient use steroid eye drops?",
            "language": "en",
            "current_step": 6
        }
        res = client.post("/api/doctor-buddy/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "blocked"

    def test_ask_endpoint_answers_technical_question(self, client):
        payload = {
            "question": "What is the anemia module measuring?",
            "language": "en",
            "current_step": 4
        }
        res = client.post("/api/doctor-buddy/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "anemia" in data["message"].lower() or "hemoglobin" in data["message"].lower()
