"""
tests/test_layer6_webcam.py
---------------------------
Layer 6 test suite: Live Laptop Webcam Capture, Quality Guidance, and Camera-Domain Validation Gates.
"""

import os
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.inference.session_export import (
    create_session_export,
    validate_session_export_schema
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_webcam_frame_bytes():
    """Generates an in-memory sample 1280x720 simulated webcam frame."""
    img = Image.new('RGB', (1280, 720), color=(140, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def sample_real_image_bytes():
    p = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg')
    with open(p, 'rb') as f:
        return f.read()


def test_webcam_ui_rendered_in_dashboard(client):
    """Verifies that all Layer 6 webcam elements exist in the served HTML."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "btnModeWebcam" in html
    assert "webcamSection" in html
    assert "webcamVideo" in html
    assert "btnStartCamera" in html
    assert "btnCaptureStill" in html
    assert "btnRecapture" in html
    assert "chkWebcamOverride" in html
    assert "Camera-Domain Validation Gate Active" in html


def test_quality_check_returns_guidance_and_dimensions(client, sample_webcam_frame_bytes):
    """Tests that /api/quality-check provides dimensions, usability, and guidance."""
    files = {"image": ("webcam_frame.jpg", sample_webcam_frame_bytes, "image/jpeg")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["is_valid"] is True
    assert data["dimensions"]["width"] == 1280
    assert data["dimensions"]["height"] == 720
    assert "usability_status" in data
    assert len(data["actionable_guidance"]) >= 1
    assert "anatomical_disclaimer" in data
    assert "Image-quality check verifies pixel sharpness" in data["anatomical_disclaimer"]


def test_webcam_default_domain_validation_gate(client, sample_webcam_frame_bytes):
    """Ensures that webcam captures are blocked by default from model inference."""
    files = [("images", ("webcam_capture.jpg", sample_webcam_frame_bytes, "image/jpeg"))]
    data = {
        "module": "anemia",
        "capture_source": "webcam",
        "experimental_override": "false"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "anemia"
    assert result["capture_source"] == "webcam"
    assert result["operational_status"] == "blocked_domain_validation"
    assert result["inference_permitted"] is False
    assert result["model_output"] is None
    assert any("camera-domain validation gate" in w.lower() for w in result["reliability_warnings"])


def test_webcam_experimental_override(client, sample_webcam_frame_bytes):
    """Verifies that explicit researcher override allows unvalidated testing with prominent warnings."""
    files = [("images", ("webcam_capture.jpg", sample_webcam_frame_bytes, "image/jpeg"))]
    data = {
        "module": "anemia",
        "capture_source": "webcam",
        "experimental_override": "true"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "anemia"
    assert result["capture_source"] == "webcam"
    assert result["operational_status"] == "available_research_only"
    assert result["inference_permitted"] is True
    assert result["unit"] == "g/dL"
    assert len(result["model_output"]) == 1
    # Check that prominent unvalidated warnings exist
    assert any("unvalidated webcam" in w.lower() for w in result["reliability_warnings"])
    assert any("override active" in w.lower() for w in result["reliability_warnings"])


def test_conjunctivitis_webcam_gate_preserved(client, sample_webcam_frame_bytes):
    """Ensures conjunctivitis remains blocked even when webcam mode is active."""
    files = [("images", ("webcam_capture.jpg", sample_webcam_frame_bytes, "image/jpeg"))]
    data = {
        "module": "conjunctivitis",
        "capture_source": "webcam",
        "experimental_override": "true"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "conjunctivitis"
    assert result["operational_status"] == "blocked_validation"
    assert result["inference_permitted"] is False
    assert result["model_output"] is None


def test_jaundice_webcam_gate_preserved(client, sample_webcam_frame_bytes):
    """Ensures jaundice remains unavailable even when webcam mode is active."""
    data = {
        "module": "jaundice",
        "capture_source": "webcam"
    }

    res = client.post("/api/analyze", data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "jaundice"
    assert result["operational_status"] == "unavailable_missing_data"
    assert result["inference_permitted"] is False
    assert result["model_output"] is None


def test_session_export_includes_capture_source(client):
    """Verifies that session export includes capture_source and passes schema validation."""
    mock_webcam_session = create_session_export(
        module="anemia",
        inference_result={
            "status": "blocked_domain_validation",
            "prediction": None,
            "warnings": ["Webcam domain gate active"],
            "limitations": ["Requires prospective validation"],
            "disclaimer": "RESEARCH USE ONLY"
        },
        filenames=["webcam_still.jpg"],
        capture_source="webcam"
    )

    assert mock_webcam_session["capture_source"] == "webcam"
    assert validate_session_export_schema(mock_webcam_session) is True

    res = client.post("/api/export", json=mock_webcam_session)
    assert res.status_code == 200
    exported = res.json()
    assert exported["capture_source"] == "webcam"
    assert exported["operational_status"] == "blocked_domain_validation"


def test_existing_upload_inference_remains_functional(client, sample_real_image_bytes):
    """Regression check: Ensures existing file-upload inference continues working identically."""
    files = [("images", ("real_eye.jpg", sample_real_image_bytes, "image/jpeg"))]
    data = {
        "module": "anemia",
        "view_types": ["raw_eye"],
        "capture_source": "upload"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["capture_source"] == "upload"
    assert result["operational_status"] == "available_research_only"
    assert result["unit"] == "g/dL"
    assert result["view_count"] == 1
    assert result["view_mean"] > 0


def test_eye_guide_rendering_and_manual_instructions(client):
    """Verifies that the eye-guide reticle and explicit manual positioning instructions are rendered."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    assert "eye-guide-reticle" in html
    assert "eye-guide-oval" in html
    assert "Manual Macro Framing Target" in html
    assert "Positioning aid only; not automated eye detection" in html
    assert "MACRO EYE FRAMING REQUIRED — NOT A FULL FACE" in html


def test_dimension_metadata_integrity(client, sample_webcam_frame_bytes):
    """Verifies that dimension metadata is intact and frontend script does not render undefinedxundefined."""
    files = {"image": ("test_frame.jpg", sample_webcam_frame_bytes, "image/jpeg")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 200
    data = res.json()

    assert "dimensions" in data
    assert isinstance(data["dimensions"]["width"], int)
    assert isinstance(data["dimensions"]["height"], int)
    assert data["dimensions"]["width"] == 1280
    assert data["dimensions"]["height"] == 720

    # Verify frontend JavaScript code handles fallback and avoids undefinedxundefined
    js_path = os.path.join(WORKSPACE_ROOT, "static", "js", "dashboard.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js_code = f.read()
    assert "undefinedxundefined" not in js_code
    assert "Dimensions: Unavailable" in js_code


def test_webcam_override_warning_medical_decision_disclaimer(client, sample_webcam_frame_bytes):
    """Verifies that researcher override explicitly warns that outputs are unsuitable for medical decisions."""
    files = [("images", ("webcam_capture.jpg", sample_webcam_frame_bytes, "image/jpeg"))]
    data = {
        "module": "anemia",
        "capture_source": "webcam",
        "experimental_override": "true"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    warnings_text = " ".join(result["reliability_warnings"]).lower()
    limitations_text = " ".join(result["documented_limitations"]).lower()

    assert "unsuitable for medical decisions" in (warnings_text + " " + limitations_text)
    assert "unvalidated webcam" in warnings_text


def test_invalid_full_face_input_rejection(client, sample_webcam_frame_bytes):
    """Verifies that full-face inputs or incompatible face view tags are rejected before model inference."""
    # 1. API level rejection of incompatible 'full_face' anatomical view
    files = [("images", ("face.jpg", sample_webcam_frame_bytes, "image/jpeg"))]
    data = {
        "module": "anemia",
        "view_types": ["full_face"],
        "capture_source": "upload"
    }

    res = client.post("/api/analyze", files=files, data=data)
    # The endpoint or service catches incompatible anatomical view
    assert res.status_code == 400 or (res.status_code == 200 and res.json().get("operational_status") == "error")

    # 2. Framing validator unit verification
    from src.preprocessing.image_quality import verify_ocular_framing_or_detect_face
    img = Image.new("RGB", (640, 480), color=(180, 140, 120))
    
    # Incompatible face view
    check_face_view = verify_ocular_framing_or_detect_face(img, view_type="full_face")
    assert check_face_view["is_valid"] is False
    assert check_face_view["is_full_face"] is True

    # Compatible ocular view
    check_palpebral = verify_ocular_framing_or_detect_face(img, view_type="palpebral")
    assert check_palpebral["is_valid"] is True


def test_single_view_variation_display_logic():
    """Verifies that single-view analysis displays 'Not available — one view' rather than ±0 g/dL."""
    js_path = os.path.join(WORKSPACE_ROOT, "static", "js", "dashboard.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js_code = f.read()

    assert "Not available — one view" in js_code
    assert "viewCount > 1" in js_code

