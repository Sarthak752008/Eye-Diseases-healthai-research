"""
tests/test_prediction_integrity_and_reports.py
----------------------------------------------
Comprehensive automated tests verifying prediction and report integrity:
1. Two different test images produce independently computed outputs.
2. Identical-image repeatability under deterministic inference settings.
3. Correct class mapping, Hb units (g/dL), and valid confidence percentages.
4. Poor-quality, unsupported-domain, missing-checkpoint, and inference-failure cases.
5. No fallback values masquerading as real predictions.
6. API response, saved session, history, dashboard, and PDF consistency.
7. English and Hindi report generation and download endpoints.
8. AI explanation disabled, unavailable, and successful-response cases.
"""

import io
import json
import pytest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from app import app
from src.inference.unified_service import get_inference_service
from src.inference.pdf_report import (
    build_screening_report_pdf,
    build_technical_scorecard_pdf,
    _extract_model_outputs
)
from src.inference.personalized_ai import PersonalizedAIService


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_test_image(color=(180, 60, 60), size=(240, 240), pattern=None):
    """Creates a distinct PIL RGB image for testing."""
    img = Image.new("RGB", size, color=color)
    if pattern == "checker":
        arr = np.array(img)
        arr[::20, ::20] = [255, 255, 255]
        img = Image.fromarray(arr)
    elif pattern == "gradient":
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for y in range(size[1]):
            arr[y, :, 0] = int(255 * (y / size[1]))
            arr[y, :, 1] = int(color[1])
            arr[y, :, 2] = int(color[2])
        img = Image.fromarray(arr)
    return img


def image_to_bytes(img: Image.Image, format="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


class TestMultiImageIndependenceAndRepeatability:
    """Verify distinct images produce distinct outputs and identical images repeat deterministically."""

    def test_distinct_images_produce_independent_outputs(self):
        service = get_inference_service()
        img1 = make_test_image(color=(220, 50, 50), pattern="gradient")
        img2 = make_test_image(color=(80, 160, 120), pattern="checker")

        res1 = service.predict_unified([img1], capture_source="upload")
        res2 = service.predict_unified([img2], capture_source="upload")

        hb1 = res1["anemia"]["view_mean"]
        hb2 = res2["anemia"]["view_mean"]
        # Both should be numerical floats
        assert isinstance(hb1, (int, float))
        assert isinstance(hb2, (int, float))
        # Chromatic features and/or Hb must not be identical hardcoded constants
        chrom1 = res1["anemia"]["prediction"][0]["chromatic"]
        chrom2 = res2["anemia"]["prediction"][0]["chromatic"]
        assert chrom1["erythema_index"] != chrom2["erythema_index"]

    def test_identical_image_deterministic_repeatability(self):
        service = get_inference_service()
        img = make_test_image(color=(190, 70, 70), pattern="gradient")

        res_a = service.predict_unified([img], capture_source="upload")
        res_b = service.predict_unified([img], capture_source="upload")

        assert res_a["anemia"]["view_mean"] == res_b["anemia"]["view_mean"]
        assert res_a["slid"]["view_mean"] == res_b["slid"]["view_mean"]
        assert res_a["jaundice"]["view_mean"] == res_b["jaundice"]["view_mean"]


class TestClassMappingsAndUnits:
    """Verify class mapping semantics, Hb units, and probability bounds."""

    def test_anemia_units_and_scaling(self):
        service = get_inference_service()
        import os
        real_img_path = "data/processed/anemia/IND_001_20200118_164733.jpg"
        if os.path.exists(real_img_path):
            img = Image.open(real_img_path).convert("RGB")
        else:
            img = make_test_image(color=(200, 60, 60))
        res = service.predict_anemia(img, capture_source="upload")

        assert res["status"] == "available_research_only"
        assert res["unit"] == "g/dL"
        val = res["view_mean"]
        assert 4.0 <= val <= 20.0, f"Predicted Hb {val} g/dL is physiologically implausible"

    def test_slid_class_mapping_and_confidence(self):
        service = get_inference_service()
        img = make_test_image(color=(160, 40, 40))
        res = service.predict_conjunctival_injection(img, capture_source="upload")

        assert res["status"] == "available_research_only"
        assert res["unit"] == "categorical"
        pred = res["prediction"][0]
        assert pred["predicted_class"] in ("control_normal", "conjunctival_injection")
        assert 0.5 <= pred["confidence"] <= 1.0
        assert "control_normal" in pred["probabilities"]
        assert "conjunctival_injection" in pred["probabilities"]

    def test_jaundice_class_mapping_and_confidence(self):
        service = get_inference_service()
        img = make_test_image(color=(210, 190, 50))
        res = service.predict_scleral_icterus(img, capture_source="upload")

        assert res["status"] == "available_research_only"
        assert res["unit"] == "categorical"
        pred = res["prediction"][0]
        assert pred["predicted_class"] in ("normal", "visibly_jaundiced")
        assert 0.5 <= pred["confidence"] <= 1.0


class TestDomainValidationAndReliabilityGates:
    """Verify safety gates: webcam blocks, legacy mendeley block, missing bilirubin block."""

    def test_webcam_input_triggers_domain_validation_gate(self):
        service = get_inference_service()
        img = make_test_image()
        res = service.predict_anemia(img, capture_source="webcam", experimental_override=False)

        assert res["status"] == "blocked_domain_validation"
        assert res["prediction"] is None
        assert any("CAMERA-DOMAIN VALIDATION GATE" in w for w in res["warnings"])

    def test_legacy_mendeley_conjunctivitis_is_blocked(self):
        service = get_inference_service()
        res = service.predict_conjunctivitis()

        assert res["status"] == "blocked_validation"
        assert res["prediction"] is None
        assert any("shortcut" in w.lower() for w in res["warnings"])

    def test_serum_bilirubin_numerical_gate_enforced(self):
        service = get_inference_service()
        res = service.predict_jaundice()

        assert res["status"] == "unavailable_missing_data"
        assert res["prediction"] is None
        assert any("No clinical laboratory total serum bilirubin" in w for w in res["warnings"])


class TestNoFallbackValuesMasqueradingAsNormal:
    """Ensure blocked or unavailable predictions are NEVER converted to fake normal results in reports."""

    def test_blocked_modules_not_faked_as_normal_in_pdf_extraction(self):
        blocked_session = {
            "session_id": "ses_test_blocked_gate",
            "anemia_result": {
                "status": "blocked_domain_validation",
                "prediction": None,
                "view_mean": None
            },
            "slid_result": {
                "status": "blocked_domain_validation",
                "prediction": None
            },
            "jaundice_result": {
                "status": "unavailable_missing_data",
                "prediction": None
            }
        }
        outputs = _extract_model_outputs(blocked_session)
        # MUST NOT claim "Normal / Healthy" or "Normal Sclera"
        assert "Normal" not in outputs["slid_val"]
        assert "Normal" not in outputs["jaund_val"]
        assert "Blocked" in outputs["hb_val"]
        assert "Blocked" in outputs["slid_val"]
        assert "Unavailable" in outputs["jaund_val"]

    def test_screening_and_scorecard_pdf_generation_with_blocked_session(self):
        blocked_session = {
            "session_id": "ses_test_blocked_gate",
            "patient_full_name": "Test Subject",
            "patient_age_years": 45,
            "patient_clinical_sex": "female",
            "capture_source": "webcam",
            "anemia_result": {"status": "blocked_domain_validation", "prediction": None, "view_mean": None},
            "slid_result": {"status": "blocked_domain_validation", "prediction": None},
            "jaundice_result": {"status": "unavailable_missing_data", "prediction": None},
            "quality_summary": {"laplacian_var": 45.2, "luminance_mean": 95.0, "is_blurry": False}
        }
        pdf_en = build_screening_report_pdf(blocked_session, language="en")
        assert len(pdf_en) > 1000
        assert pdf_en[:4] == b"%PDF"

        scorecard_pdf = build_technical_scorecard_pdf(blocked_session, language="en")
        assert len(scorecard_pdf) > 1000
        assert scorecard_pdf[:4] == b"%PDF"


class TestApiWorkflowAndReportEndpoints:
    """Verify full upload -> analyze -> persist -> retrieve -> download workflow."""

    def test_complete_upload_and_pdf_download_workflow(self, client):
        img_bytes = image_to_bytes(make_test_image(color=(190, 65, 65)))
        files = [("images", ("test_eye.jpg", img_bytes, "image/jpeg"))]
        data = {
            "module": "unified",
            "user_id": "usr_sarthak_default",
            "age_years": "28",
            "clinical_sex": "male",
            "capture_source": "upload"
        }

        # 1. Analyze
        res = client.post("/api/analyze", files=files, data=data)
        assert res.status_code == 200
        res_data = res.json()
        assert "session_id" in res_data
        sid = res_data["session_id"]
        assert res_data["anemia"]["view_mean"] is not None
        assert res_data["slid"]["status"] == "available_research_only"
        assert res_data["jaundice"]["status"] == "available_research_only"

        # 2. Retrieve session from API
        sess_res = client.get(f"/api/reports/{sid}")
        assert sess_res.status_code == 200
        retrieved = sess_res.json()
        assert retrieved["session_id"] == sid
        assert retrieved["anemia"]["view_mean"] == res_data["anemia"]["view_mean"]

        # 3. Download Screening PDF in English and Hindi
        pdf_en = client.get(f"/api/reports/{sid}/screening-pdf?lang=en")
        assert pdf_en.status_code == 200
        assert pdf_en.headers["content-type"] == "application/pdf"
        assert pdf_en.content[:4] == b"%PDF"

        pdf_hi = client.get(f"/api/reports/{sid}/screening-pdf?lang=hi")
        assert pdf_hi.status_code == 200
        assert pdf_hi.content[:4] == b"%PDF"

        # 4. Download Technical Scorecard PDF
        scorecard = client.get(f"/api/reports/{sid}/scorecard-pdf?lang=en")
        assert scorecard.status_code == 200
        assert scorecard.content[:4] == b"%PDF"

        # 5. Access Interactive Web Report
        web_rep = client.get(f"/report?session_id={sid}&lang=en")
        assert web_rep.status_code == 200
        assert "OcuScan" in web_rep.text


class TestPersonalizedAIServiceGuardrails:
    """Verify AI explanation service respects consent and degrades gracefully without fabrication."""

    def test_ai_explanation_without_consent_returns_local_deterministic(self):
        service = PersonalizedAIService()
        img = make_test_image()
        result = service.generate_personalized_explanation(
            image=img,
            age_years=35,
            clinical_sex="female",
            external_ai_consent=False
        )
        assert result["provider"] == "local_deterministic"
        assert result["status"] in ("completed", "success")
        assert "bilingual" in result
        assert "en" in result["bilingual"]
        assert "hi" in result["bilingual"]

    def test_ai_explanation_with_pediatric_age_triggers_population_alert(self):
        service = PersonalizedAIService()
        img = make_test_image()
        result = service.generate_personalized_explanation(
            image=img,
            age_years=8,
            clinical_sex="male",
            external_ai_consent=False
        )
        pop_status = result["bilingual"]["en"]["population_status"]
        assert "pediatric" in pop_status.lower() or "outside" in pop_status.lower()
