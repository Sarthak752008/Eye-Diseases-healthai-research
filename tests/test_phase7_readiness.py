"""
tests/test_phase7_readiness.py
------------------------------
Phase 7 End-to-End Verification and Deploy-Readiness Test Suite.
Covers:
- Dual health endpoints (/health and /api/health)
- Security headers and upload size/format enforcement
- Corrupt/oversized/invalid image rejection
- Session reset and race-condition guards in frontend
- End-to-end integration workflow (upload -> quality-check -> inference -> export)
- System resilience and configuration integrity
"""

import os
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.config import settings
from src.inference.session_export import validate_session_export_schema


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_ocular_image_bytes():
    """Real sample image from dataset."""
    img_path = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg')
    with open(img_path, 'rb') as f:
        return f.read()


def test_health_endpoints_dual_route(client):
    """Verifies that both /health and /api/health report identical operational status."""
    res1 = client.get("/health")
    assert res1.status_code == 200
    data1 = res1.json()

    res2 = client.get("/api/health")
    assert res2.status_code == 200
    data2 = res2.json()

    assert data1 == data2
    assert data1["status"] == "healthy"
    assert "EyeHealthAI" in data1["service"]
    assert data1["modules"]["anemia"]["operational_status"] == "available_research_only"
    assert data1["modules"]["conjunctivitis"]["operational_status"] == "blocked_validation"
    assert data1["modules"]["jaundice"]["operational_status"] == "unavailable_missing_data"
    assert "RESEARCH USE ONLY" in data1["disclaimer"]


def test_security_headers_present(client):
    """Verifies that all required HTTP security headers are attached to responses."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("x-content-type-options") == "nosniff"
    assert res.headers.get("x-frame-options") == "DENY"
    assert res.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "camera=(self)" in res.headers.get("permissions-policy", "")


def test_oversized_image_rejection(client):
    """Verifies that image uploads exceeding MAX_UPLOAD_SIZE_MB trigger HTTP 413."""
    # Create fake oversized payload (> 10MB)
    fake_oversized = b"0" * (11 * 1024 * 1024)
    files = {"image": ("huge_file.jpg", fake_oversized, "image/jpeg")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 413
    assert "exceeds maximum allowed upload size" in res.json()["detail"]


def test_unsupported_file_extension_rejection(client):
    """Verifies that forbidden file extensions are rejected with HTTP 400."""
    fake_file = b"executable code or script content"
    files = {"image": ("malicious.exe", fake_file, "application/octet-stream")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]


def test_unsupported_mime_type_rejection(client):
    """Verifies that disallowed MIME types are rejected with HTTP 400."""
    files = {"image": ("data.txt", b"plain text", "text/plain")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 400
    assert "Unsupported" in res.json()["detail"]


def test_frontend_session_invalidation_and_reset_integrity():
    """Verifies that dashboard.js implements session reset, invalidation, and race-condition guards."""
    js_path = os.path.join(WORKSPACE_ROOT, "static", "js", "dashboard.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js_code = f.read()

    assert "resetSession" in js_code
    assert "invalidatePreviousResults" in js_code
    assert "currentSessionRequestId" in js_code
    assert "btnResetSession" in js_code
    assert "Not available — one view" in js_code
    assert "Dimensions: Unavailable" in js_code
    assert "undefinedxundefined" not in js_code


def test_html_dashboard_reset_button_exists(client):
    """Verifies that the New Session button exists in index.html."""
    res = client.get("/")
    assert res.status_code == 200
    assert 'id="btnResetSession"' in res.text
    assert "New Session" in res.text


def test_configuration_integrity():
    """Verifies that central configuration loads properly with secure defaults."""
    assert settings.MAX_UPLOAD_SIZE_MB >= 1
    assert settings.MAX_UPLOAD_SIZE_BYTES == settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    assert ".jpg" in settings.ALLOWED_IMAGE_EXTENSIONS
    assert ".png" in settings.ALLOWED_IMAGE_EXTENSIONS
    assert "image/jpeg" in settings.ALLOWED_MIME_TYPES
    assert "image/png" in settings.ALLOWED_MIME_TYPES


def test_end_to_end_research_workflow(client, sample_ocular_image_bytes):
    """
    End-to-end integration test:
    1. Pre-screen image quality (/api/quality-check)
    2. Execute experimental Anemia inference (/api/analyze)
    3. Export verified session report (/api/export)
    """
    # Step 1: Pre-screen image quality
    q_files = {"image": ("session_crop.jpg", sample_ocular_image_bytes, "image/jpeg")}
    q_res = client.post("/api/quality-check", files=q_files)
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["is_valid"] is True
    assert "dimensions" in q_data
    assert q_data["dimensions"]["width"] > 0
    assert q_data["dimensions"]["height"] > 0

    # Step 2: Execute inference
    infer_files = [("images", ("session_crop.jpg", sample_ocular_image_bytes, "image/jpeg"))]
    infer_data = {
        "module": "anemia",
        "view_types": ["raw_eye"],
        "capture_source": "upload"
    }
    infer_res = client.post("/api/analyze", files=infer_files, data=infer_data)
    assert infer_res.status_code == 200
    session_payload = infer_res.json()
    assert session_payload["operational_status"] == "available_research_only"
    assert session_payload["view_count"] == 1
    assert session_payload["view_mean"] > 0

    # Step 3: Export session report
    export_res = client.post("/api/export", json=session_payload)
    assert export_res.status_code == 200
    assert "application/json" in export_res.headers["content-type"]
    exported_report = export_res.json()
    assert validate_session_export_schema(exported_report) is True
    assert exported_report["session_id"] == session_payload["session_id"]
    assert exported_report["selected_module"] == "anemia"


def test_screening_report_page_served(client):
    """Verifies that the Ocular Health Screening Report page is served at /report."""
    res = client.get("/report")
    assert res.status_code == 200
    html = res.text
    assert "Ocular Health Scorecard" in html
    assert "AI SCREENING REPORT:" in html
    assert "TREATMENT &amp; SCREENING RECOMMENDATIONS:" in html
    assert "Microvascular Pallor" in html
    assert "rptOverallScore" in html
    assert "Print / Save as PDF" in html
