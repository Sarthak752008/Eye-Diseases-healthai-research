"""
tests/test_cataract_and_stye_integration.py
-------------------------------------------
Comprehensive integration test suite for 5-Target OcuScan Architecture:
Target A: Anemia / Hemoglobin (ResNet-50)
Target B: Conjunctival Injection / Hyperemia (SLID MobileNetV3-Small)
Target C: Scleral Icterus / Jaundice (MobileNetV3-Small)
Target D: Nuclear Cataract (MobileNetV3-Small LOCS III Staging)
Target E: Stye / Hordeolum (MobileNetV3-Small Acute Lesion Detection)

Tests:
1. Model Architecture & Checkpoint Loading (Cataract & Stye)
2. Slit-Lamp Biomicroscopy Domain Validation Gate for Cataract
3. Stye Anterior Photography Screening
4. Full Unified Multi-Target Inference Pass (All 5 Targets Evaluated)
5. Grad-CAM Feature Attribution for Cataract & Stye
6. Health and Module Alias Routing via app.py
7. Two-Page PDF Report Generation Incorporating Cataract & Stye
"""

import sys
import os
import io
import json
import pytest
from PIL import Image
import torch
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app import app
from src.models.cataract_model import CataractClassifier, load_cataract_checkpoint
from src.models.stye_model import StyeClassifier, load_stye_checkpoint
from src.inference.unified_service import UnifiedInferenceService, InferenceResult
from src.inference.pdf_report import build_screening_report_pdf, _extract_clinical_findings
from src.config import CATARACT_CHECKPOINT_PATH, STYE_CHECKPOINT_PATH


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_eye_image(tmp_path):
    img_path = str(tmp_path / "test_eye.jpg")
    img = Image.new("RGB", (300, 300), color=(180, 110, 110))
    img.save(img_path, format="JPEG")
    return img_path


def test_cataract_checkpoint_and_architecture():
    """Verify CataractClassifier initializes and loads checkpoint cleanly."""
    model = CataractClassifier(pretrained=False, num_classes=2)
    assert model is not None
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 2)

    if os.path.exists(CATARACT_CHECKPOINT_PATH):
        loaded = load_cataract_checkpoint(CATARACT_CHECKPOINT_PATH, device="cpu")
        assert loaded is not None
        out_loaded = loaded(dummy)
        assert out_loaded.shape == (1, 2)


def test_stye_checkpoint_and_architecture():
    """Verify StyeClassifier initializes and loads checkpoint cleanly."""
    model = StyeClassifier(pretrained=False, num_classes=2)
    assert model is not None
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 2)

    if os.path.exists(STYE_CHECKPOINT_PATH):
        loaded = load_stye_checkpoint(STYE_CHECKPOINT_PATH, device="cpu")
        assert loaded is not None
        out_loaded = loaded(dummy)
        assert out_loaded.shape == (1, 2)


def test_cataract_domain_gate(sample_eye_image):
    """Verify Cataract triggers blocked_domain_validation on webcam without experimental override."""
    service = UnifiedInferenceService()
    # Webcam capture without override should be domain-gated
    res_gated = service.predict_cataract(sample_eye_image, capture_source="webcam", experimental_override=False)
    assert res_gated.status == "blocked_domain_validation"
    assert "CAMERA-DOMAIN VALIDATION GATE" in res_gated.warnings[0]

    # With experimental override, inference is executed
    res_override = service.predict_cataract(sample_eye_image, capture_source="webcam", experimental_override=True)
    assert res_override.status in ("available_research_only", "success")
    assert res_override.prediction is not None
    assert len(res_override.prediction) > 0
    pred = res_override.prediction[0]
    assert "predicted_class" in pred
    assert "locs_grade_estimate" in pred


def test_cataract_upload_inference(sample_eye_image):
    """Verify Cataract upload execution and clinical decision formatting."""
    service = UnifiedInferenceService()
    res = service.predict_cataract(sample_eye_image, capture_source="upload")
    assert res.status in ("available_research_only", "success")
    assert res.prediction is not None
    pred = res.prediction[0]
    assert pred["predicted_class"] in ("normal_clear_lens", "nuclear_cataract")
    assert 1 <= pred["locs_grade_estimate"] <= 6
    assert "clinical_decision" in pred
    assert "bilingual_advice" in pred["clinical_decision"]


def test_stye_upload_inference(sample_eye_image):
    """Verify Stye inference classifies lesion status and provides clinical advice."""
    service = UnifiedInferenceService()
    res = service.predict_stye(sample_eye_image, capture_source="upload")
    assert res.status in ("available_research_only", "success")
    assert res.prediction is not None
    pred = res.prediction[0]
    assert pred["predicted_class"] in ("normal", "stye_hordeolum")
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "clinical_decision" in pred
    assert "bilingual_advice" in pred["clinical_decision"]


def test_unified_service_five_targets(sample_eye_image):
    """Verify UnifiedInferenceService.predict_unified runs all 5 targets in consolidated pass."""
    service = UnifiedInferenceService()
    unified = service.predict_unified([sample_eye_image], capture_source="upload")

    assert "session_id" in unified
    assert "anemia" in unified
    assert "slid" in unified
    assert "jaundice" in unified
    assert "cataract" in unified
    assert "stye" in unified

    # Check that module_execution_audit lists all 5 targets
    audit = unified.get("module_execution_audit", {})
    assert "anemia" in audit
    assert "slid" in audit
    assert "jaundice" in audit
    assert "cataract" in audit
    assert "stye" in audit

    # Check that image_sequence contains stage 03 & 04 with Grad-CAM overlays
    seq = unified.get("image_sequence", {})
    assert "stage_01_upload" in seq
    assert "stage_04_visualization" in seq
    stage_04 = seq["stage_04_visualization"]
    assert "overlays" in stage_04
    overlays = stage_04["overlays"]
    assert "cataract" in overlays
    assert "stye" in overlays


def test_fastapi_analyze_endpoint_single_modules(test_client, sample_eye_image):
    """Verify /api/analyze router handles single module requests for cataract and stye."""
    with open(sample_eye_image, "rb") as f:
        # Test cataract single target
        res_cat = test_client.post(
            "/api/analyze",
            data={"module": "cataract", "capture_source": "upload"},
            files={"images": ("test_eye.jpg", f, "image/jpeg")}
        )
    assert res_cat.status_code == 200
    cat_json = res_cat.json()
    assert cat_json["status"] in ("available_research_only", "success")

    with open(sample_eye_image, "rb") as f:
        # Test stye single target
        res_stye = test_client.post(
            "/api/analyze",
            data={"module": "stye", "capture_source": "upload"},
            files={"images": ("test_eye.jpg", f, "image/jpeg")}
        )
    assert res_stye.status_code == 200
    stye_json = res_stye.json()
    assert stye_json["status"] in ("available_research_only", "success")


def test_pdf_report_five_targets(sample_eye_image):
    """Verify build_screening_report_pdf builds a valid 2-page PDF with 5 targets."""
    service = UnifiedInferenceService()
    unified = service.predict_unified([sample_eye_image], capture_source="upload")

    # English PDF
    pdf_bytes_en = build_screening_report_pdf(unified, language="en")
    assert isinstance(pdf_bytes_en, (bytes, bytearray))
    assert len(pdf_bytes_en) > 1000
    assert pdf_bytes_en.startswith(b"%PDF")

    # Hindi PDF
    pdf_bytes_hi = build_screening_report_pdf(unified, language="hi")
    assert isinstance(pdf_bytes_hi, (bytes, bytearray))
    assert len(pdf_bytes_hi) > 1000
    assert pdf_bytes_hi.startswith(b"%PDF")
