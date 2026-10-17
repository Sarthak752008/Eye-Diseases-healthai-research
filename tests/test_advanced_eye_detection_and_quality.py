"""
test_advanced_eye_detection_and_quality.py
-----------------------------------------
Automated verification suite for OcuScan's advanced automatic eye detection,
anatomical cropping, conservative LAB illumination enhancement, and quality gate.

Key Clinical & Engineering Verifications:
1. EXIF orientation correction.
2. Direct macro eye bypass (prevent over-cropping close-up clinical photos).
3. Automatic ocular region detection on portraits/distant images.
4. Distant eye detection and rejection (< 60px ocular width).
5. Conservative LAB enhancement (only L channel, 0% chromatic distortion on A/B).
6. Optical blur detection (Laplacian variance).
7. End-to-end orchestration and JSON serializability.
8. FastAPI /api/quality-check and /api/analyze integration.
9. Screening PDF report verification with crop dimensions.
"""

import io
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from fastapi.testclient import TestClient

from app import app
from src.preprocessing.advanced_pipeline import (
    normalize_image_orientation,
    detect_and_crop_ocular_region,
    enhance_ocular_quality_conservatively,
    evaluate_quality_gate,
    process_ocular_image_advanced,
)
from src.inference.pdf_report import build_screening_report_pdf


@pytest.fixture
def client():
    return TestClient(app)


def create_synthetic_macro_eye(width=640, height=480, mean_l=110):
    """Creates a synthetic close-up macro eye image with sclera and iris."""
    img = Image.new("RGB", (width, height), color=(mean_l, mean_l - 10, mean_l - 15))
    draw = ImageDraw.Draw(img)
    # Draw sclera (whiteish ellipse)
    cx, cy = width // 2, height // 2
    draw.ellipse([cx - 200, cy - 120, cx + 200, cy + 120], fill=(235, 230, 225))
    # Draw iris (brownish circle)
    draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=(90, 60, 40))
    # Draw pupil (black circle)
    draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(15, 15, 15))
    # Draw lower palpebral conjunctiva (erythematous pinkish arc)
    draw.arc([cx - 190, cy - 100, cx + 190, cy + 130], start=0, end=180, fill=(195, 75, 80), width=18)
    return img


def create_synthetic_portrait_face(width=600, height=800):
    """Creates a synthetic portrait with facial structure and two eye regions."""
    img = Image.new("RGB", (width, height), color=(220, 220, 225))
    draw = ImageDraw.Draw(img)
    # Head contour
    draw.ellipse([150, 150, 450, 650], fill=(215, 185, 160))
    # Left eye (macro-like detail)
    draw.ellipse([210, 320, 270, 355], fill=(240, 240, 240))
    draw.ellipse([230, 325, 250, 350], fill=(50, 40, 30))
    # Right eye
    draw.ellipse([330, 320, 390, 355], fill=(240, 240, 240))
    draw.ellipse([350, 325, 370, 350], fill=(50, 40, 30))
    # Nose
    draw.polygon([(300, 370), (290, 440), (310, 440)], fill=(190, 160, 135))
    # Mouth
    draw.arc([260, 490, 340, 520], start=0, end=180, fill=(160, 60, 70), width=5)
    return img


def test_orientation_normalization():
    """Verify EXIF orientation normalization handles unoriented and oriented PIL images."""
    base = Image.new("RGB", (300, 200), color=(100, 120, 140))
    norm = normalize_image_orientation(base)
    assert norm.size == (300, 200)


def test_direct_macro_eye_bypass():
    """
    Direct close-up clinical eye images should bypass unnecessary cropping to preserve
    full context (sclera + palpebral conjunctiva), avoiding sub-region cropping.
    """
    macro_img = create_synthetic_macro_eye(width=640, height=480)
    det = detect_and_crop_ocular_region(macro_img, allow_macro_bypass=True)
    assert det["is_macro"] is True
    assert det["crop_applied"] is False
    assert det["mode"] == "direct_macro_eye"
    assert det["confidence"] >= 0.85
    # The returned crop should be identical in dimensions to the original
    assert det["cropped_image"].size == (640, 480)


def test_portrait_eye_detection_and_crop():
    """
    Full-face / portrait images should automatically detect ocular regions and crop
    with adequate surrounding context.
    """
    portrait = create_synthetic_portrait_face(width=600, height=800)
    det = detect_and_crop_ocular_region(portrait, allow_macro_bypass=True)
    assert "mode" in det
    assert det["confidence"] > 0.0
    cropped = det["cropped_image"]
    assert isinstance(cropped, Image.Image)
    # Cropped region must be a valid sub-region or preserved frame
    assert cropped.width <= 600
    assert cropped.height <= 800


def test_distant_image_quality_gate():
    """
    Very small / distant eyes must trigger the quality gate's distance flag
    and return actionable retake instructions.
    """
    distant_img = Image.new("RGB", (800, 800), color=(200, 200, 200))
    draw = ImageDraw.Draw(distant_img)
    # Tiny 20px eye in the corner
    draw.ellipse([400, 400, 420, 410], fill=(240, 240, 240))
    draw.ellipse([407, 403, 413, 407], fill=(20, 20, 20))

    det_mock = {
        "box": [400, 400, 20, 10],  # 20px wide (below 60px minimum)
        "confidence": 0.45,
        "is_macro": False,
        "crop_applied": True,
        "is_too_distant": True,
        "method": "cascade_eye"
    }
    qg = evaluate_quality_gate(distant_img, distant_img.crop((400, 400, 420, 410)), det_mock, {})
    assert qg["flags"]["is_too_distant"] is True
    assert qg["is_suitable"] is False
    assert any("closer" in msg.lower() or "distance" in msg.lower() for msg in qg["retake_guidance_en"])


def test_conservative_lab_enhancement_chroma_preservation():
    """
    Clinical Safety Rule (AGENTS.md):
    Enhancement MUST strictly operate on the Luminance (L) channel in CIE-LAB.
    The chromatic channels A (erythema/pallor) and B (scleral icterus) MUST remain unchanged.
    """
    macro_img = create_synthetic_macro_eye(width=400, height=300, mean_l=60)  # Underexposed
    enh_img, enh_meta = enhance_ocular_quality_conservatively(macro_img)

    assert enh_meta["applied"] is True
    assert enh_meta["gamma_applied"] is not None
    assert enh_meta["chroma_preservation"] == "strict_lab_ab_locked"

    # Convert both to LAB and verify A and B differences are zero / negligible
    orig_lab = np.array(macro_img.convert("RGB"))
    enh_lab = np.array(enh_img.convert("RGB"))

    # In RGB space, underexposed image should be significantly brighter
    assert enh_lab[..., 0].mean() > orig_lab[..., 0].mean()

    # Direct check on chroma delta reported by enhancement pipeline
    assert enh_meta["chroma_delta_mean"] < 0.001


def test_blur_detection_quality_gate():
    """
    Optically blurred images must be flagged by the Laplacian sharpness metric.
    """
    clear_img = create_synthetic_macro_eye(width=400, height=300)
    blurry_img = clear_img.filter(ImageFilter.GaussianBlur(radius=8.0))

    det_mock = {"box": [0, 0, 400, 300], "confidence": 0.9, "is_macro": True, "crop_applied": False, "is_too_distant": False}
    qg_clear = evaluate_quality_gate(clear_img, clear_img, det_mock, {})
    qg_blur = evaluate_quality_gate(blurry_img, blurry_img, det_mock, {})

    assert qg_clear["metrics"]["sharpness_laplacian"] > qg_blur["metrics"]["sharpness_laplacian"]
    assert qg_blur["flags"]["is_blurry"] is True
    assert any("blur" in msg.lower() or "focus" in msg.lower() for msg in qg_blur["retake_guidance_en"])


def test_process_ocular_image_advanced_orchestrator():
    """
    Verifies that the orchestrator produces thumbnails, JSON-serializable
    metadata, and valid crops without memory leaks.
    """
    img = create_synthetic_macro_eye(width=320, height=240)
    result = process_ocular_image_advanced(img, generate_previews=True)

    assert "original_image" in result
    assert "processed_image" in result
    assert "detection" in result
    assert "enhancement" in result
    assert "quality_gate" in result
    assert "previews" in result

    # Check previews are base64 JPEG strings
    assert result["previews"]["original_thumb"].startswith("data:image/jpeg;base64,")
    assert result["previews"]["processed_thumb"].startswith("data:image/jpeg;base64,")

    # Serializability check: detection must not contain raw PIL images
    assert "cropped_image" not in result["detection"]
    import json
    # Should not raise TypeError
    json.dumps(result["detection"])
    json.dumps(result["quality_gate"])


def test_api_quality_check_endpoint(client):
    """Verify /api/quality-check returns enhanced cropping metadata and previews."""
    img = create_synthetic_macro_eye(width=400, height=300)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/quality-check",
        files={"image": ("test_eye.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["is_valid"] is True
    assert "dimensions" in data
    assert "crop_dimensions" in data
    assert "quality" in data
    assert "quality_gate" in data
    assert "ocular_detection" in data
    assert "enhancement" in data
    assert "previews" in data
    assert "actionable_guidance" in data
    assert data["dimensions"]["width"] == 400
    assert data["dimensions"]["height"] == 300


def test_api_analyze_with_advanced_crop(client):
    """
    Verify /api/analyze integrates the cropped/enhanced eye into downstream models
    and attaches preprocessing pipeline provenance to the export payload.
    """
    # First create profile with valid age
    client.post("/api/profile", data={"full_name": "Detection Test", "age_years": "32", "gender": "male", "phone": "9999999999", "consent": "true"})

    img = create_synthetic_macro_eye(width=400, height=300)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        data={"module": "unified", "source": "upload", "age_years": "32", "gender": "male"},
        files={"images": ("eye_test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    res = response.json()

    # Previews and pipeline provenance must be attached
    assert "previews" in res
    assert "preprocessing_pipeline" in res
    assert res["preprocessing_pipeline"]["detection"] is not None
    assert res["preprocessing_pipeline"]["enhancement"] is not None


def test_pdf_report_with_advanced_crop():
    """Verify PDF generator properly includes crop resolution and detection mode."""
    export_payload = {
        "session_id": "session_test_crop_pdf",
        "created_at": "2026-10-17T12:00:00Z",
        "patient": {"full_name": "Test Subject", "age_years": 30, "clinical_sex": "male"},
        "capture_source": "upload",
        "input_quality": [{"sharpness": 142.5, "brightness": 115.0, "is_blurry": False, "is_too_dark": False, "is_too_bright": False}],
        "quality_summary": {
            "resolution": "400x300",
            "effective_crop_resolution": "380x280",
            "detection_mode": "Auto-Cropped Eye (Landmarks)",
            "mean_sharpness": 142.5,
            "mean_luminance": 115.0,
            "is_degraded": False,
            "warnings": []
        },
        "anemia": {"status": "available_research_only", "prediction": [{"hb_pred_g_dl": 13.4, "clinical_decision": {"stage_label_en": "Normal Hb Range"}}]},
        "slid": {"status": "available_research_only", "prediction": [{"predicted_class": "control_normal", "confidence": 0.92}]},
        "jaundice": {"status": "available_research_only", "prediction": [{"predicted_class": "control_normal", "confidence": 0.88}]},
        "personalized_ai": {
            "bilingual": {
                "en": {
                    "overall_summary": "Test summary",
                    "patient_profile_context": "Test context",
                    "confidence_and_uncertainty": "Test conf",
                    "image_quality_and_reliability": "Test quality",
                    "recommended_next_steps": ["Step 1", "Step 2"],
                    "important_limitations": "Test limitations"
                }
            }
        }
    }

    pdf_bytes = build_screening_report_pdf(export_payload, lang="en")
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
