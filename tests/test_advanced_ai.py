"""
tests/test_advanced_ai.py
========================
Verification suite for EyeHealthAI Advanced AI integrations:
1. Ocular segmentation (palpebral conjunctiva, bulbar sclera, iris, glare)
2. Secondary vision analytics (GPT-5.6-terra response contract + physiological fallback)
3. MedSigLIP-style ocular embedding & reference case retrieval
4. Calibrated ensemble fusion & concordance evaluation
5. Grounded session consultation in Doctor Buddy
6. End-to-end unified inference integration
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.segmentation.clinical_segmenter import ClinicalOcularSegmenter
from src.inference.secondary_vision import get_secondary_vision_service
from src.inference.reference_retrieval import get_reference_retrieval_service, OcularEmbeddingExtractor
from src.inference.calibrated_ensemble import CalibratedEnsembleFusion, get_calibrated_ensemble
from src.inference.doctor_buddy import ask_doctor_buddy_question
from src.inference.unified_service import get_inference_service


@pytest.fixture
def synthetic_eye_rgb(tmp_path):
    """Generates a synthetic eye-like RGB image."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Sclera / eye background (off-white)
    arr[60:190, 40:220] = [230, 230, 235]
    # Iris (brownish center)
    arr[90:160, 95:165] = [80, 50, 30]
    # Conjunctiva (lower vascularized pinkish-red)
    arr[180:230, 60:200] = [200, 70, 75]
    # Glare highlight
    arr[105:115, 110:120] = [255, 255, 255]
    
    img = Image.fromarray(arr)
    file_path = tmp_path / "test_synthetic_eye.png"
    img.save(file_path)
    return str(file_path), img


def test_clinical_ocular_segmentation(synthetic_eye_rgb):
    _, pil_img = synthetic_eye_rgb
    segmenter = ClinicalOcularSegmenter()
    result = segmenter.segment(pil_img)
    
    assert result.status == "success"
    assert result.coverage_pct["conjunctiva"] > 0
    assert result.coverage_pct["sclera"] > 0
    assert result.overlay_base64.startswith("data:image/")
    assert result.conjunctiva_bbox is not None
    assert result.sclera_bbox is not None


def test_secondary_vision_deterministic_fallback(synthetic_eye_rgb):
    _, pil_img = synthetic_eye_rgb
    service = get_secondary_vision_service()
    # Offline fallback run
    result = service._run_deterministic_fallback(pil_img)
    
    assert result["status"] == "deterministic_fallback"
    assert "palpebral_conjunctiva" in result["ocular_observations"]
    assert "bulbar_sclera" in result["ocular_observations"]
    assert result["uncertainty_score"] >= 0.0
    assert len(result["clinical_limitations"]) > 0


def test_ocular_reference_retrieval(synthetic_eye_rgb):
    _, pil_img = synthetic_eye_rgb
    extractor = OcularEmbeddingExtractor(device="cpu")
    emb = extractor.extract_embedding(pil_img)
    
    assert emb.shape[0] in [2048, 96]
    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)
    
    service = get_reference_retrieval_service(device="cpu")
    retrieval = service.query(pil_img, top_k=3)
    
    assert retrieval["status"] == "success"
    assert retrieval["match_count"] == 3
    assert len(retrieval["top_matches"]) == 3
    for match in retrieval["top_matches"]:
        assert 0.0 <= match["similarity_score"] <= 100.0
        assert "condition" in match
        assert "verified_lab_value" in match


def test_calibrated_ensemble_fusion():
    fuser = CalibratedEnsembleFusion()
    
    anemia_pred = {
        "status": "COMPLETED",
        "predicted_hb": 9.2,
        "anemia_detected": True,
        "risk_level": "Moderate Anemia",
        "confidence": 0.88,
    }
    slid_pred = {
        "status": "COMPLETED",
        "predicted_label": "injection",
        "probability": 0.82,
        "is_normal": False,
        "risk_tier": "CLINICAL_REVIEW_RECOMMENDED",
    }
    jaundice_pred = {
        "status": "COMPLETED",
        "predicted_label": "normal",
        "probability": 0.15,
        "is_normal": True,
        "risk_tier": "NORMAL_OR_LOW_RISK",
    }
    chromatic = {
        "erythema_index": 22.5,
        "scleral_yellowness_index": 1.05,
        "cie_a_star": 15.0,
        "cie_b_star": 4.0,
    }
    segmentation = {
        "coverage_pct": {"conjunctiva": 12.0, "sclera": 18.0},
        "quality_flags": {"glare_contamination": False},
    }
    
    fusion = fuser.evaluate_fusion(
        anemia_result=anemia_pred,
        slid_result=slid_pred,
        jaundice_result=jaundice_pred,
        segmentation_result=segmentation,
        chromatic_features=chromatic,
    )
    
    assert "calibrated_probabilities" in fusion
    assert "concordance_score" in fusion
    assert fusion["screening_tier"] in ["high_risk_flag", "borderline_review", "low_risk"]
    assert fusion["should_abstain"] is False


def test_doctor_buddy_grounded_explanations():
    import asyncio
    # Test session-grounded consultation without database session (clean fallback)
    resp = asyncio.run(ask_doctor_buddy_question(
        question="What did each model observe and what lab test should I take?",
        language="en",
        current_step=6,
        session_id="non_existent_mock_session",
    ))
    
    assert "message" in resp
    assert "AI Doctor Buddy" in resp["message"] or "screening" in resp["message"].lower()
    assert "speak_text" in resp


def test_unified_service_end_to_end(synthetic_eye_rgb):
    file_path, _ = synthetic_eye_rgb
    service = get_inference_service()
    result = service.predict_unified([file_path])
    
    assert result["module"] == "unified"
    assert result["status"] in ["available_research_only", "completed", "warning", "error", "abstain"]
    
    # Verify advanced AI integrations exist in the unified payload
    assert result["segmentation"] is not None
    assert result["segmentation"]["status"] == "success"
    assert "conjunctiva" in result["segmentation"]["coverage_pct"]
    
    assert result["secondary_vision"] is not None
    assert "ocular_observations" in result["secondary_vision"]
    
    assert result["reference_retrieval"] is not None
    assert len(result["reference_retrieval"]["top_matches"]) == 3
    
    assert result["calibrated_ensemble"] is not None
    assert "screening_tier" in result["calibrated_ensemble"]
