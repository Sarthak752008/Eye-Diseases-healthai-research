"""
tests/test_profile_aware_ai_and_age.py
---------------------------------------
Comprehensive test suite for:
  1. Mandatory integer age (0-120) and profile validation in UserRegisterRequest & UserUpdateAgeRequest
  2. WHO 2024 age/sex/pregnancy clinical reference interpretation logic
  3. Population validation boundaries (pediatric <18, geriatric >88)
  4. OpenAI Personalized AI service data minimization & local deterministic fallback
  5. Profile-aware database models, migrations, and session persistence
  6. Two-page PDF report generation with age and clinical reference details
"""

import os
import io
import pytest
from pydantic import ValidationError
from PIL import Image
from pypdf import PdfReader

from app import UserRegisterRequest, UserUpdateAgeRequest
from src.inference.clinical_analytics import compute_anemia_clinical_decision
from src.inference.personalized_ai import PersonalizedAIService
from src.inference.pdf_report import build_two_page_pdf
from src.database import (
    init_db,
    register_user,
    update_user_profile_age,
    get_user_by_id,
    create_session,
    SessionLocal
)


# ===========================================================================
# 1. Mandatory Integer Age & Profile Validation Tests
# ===========================================================================

def test_user_register_request_valid_age():
    """Valid integer ages must be accepted across the 0-120 boundary."""
    for valid_age in [0, 1, 18, 42, 88, 120]:
        req = UserRegisterRequest(
            full_name="Valid Patient",
            phone="9876543210",
            email="patient@example.com",
            age_years=valid_age,
            clinical_sex="male"
        )
        assert req.age_years == valid_age
        assert req.clinical_sex == "male"
        assert req.date_of_birth is None  # Optional when age is provided


def test_user_register_request_rejects_empty_and_none_age():
    """Missing or None age must raise validation error."""
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            full_name="Patient No Age",
            phone="9876543210",
            email="patient@example.com",
            age_years=None
        )


def test_user_register_request_rejects_negative_age():
    """Negative ages must be strictly rejected."""
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            full_name="Negative Age",
            phone="9876543210",
            email="patient@example.com",
            age_years=-1
        )


def test_user_register_request_rejects_out_of_range_age():
    """Ages above 120 must be strictly rejected."""
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            full_name="Out of Range",
            phone="9876543210",
            email="patient@example.com",
            age_years=121
        )


def test_user_register_request_rejects_boolean_and_decimals():
    """Booleans (True/False) and float decimals must not silently pass as integer age."""
    with pytest.raises(ValidationError):
        UserRegisterRequest(
            full_name="Bool Age",
            phone="9876543210",
            email="patient@example.com",
            age_years=True
        )

    with pytest.raises(ValidationError):
        UserRegisterRequest(
            full_name="Decimal Age",
            phone="9876543210",
            email="patient@example.com",
            age_years=28.7
        )


def test_user_update_age_request_validation():
    """UserUpdateAgeRequest must enforce identical integer age constraints."""
    req = UserUpdateAgeRequest(age_years=35, clinical_sex="female", is_pregnant=True, pregnancy_trimester=2)
    assert req.age_years == 35
    assert req.clinical_sex == "female"
    assert req.pregnancy_trimester == 2

    with pytest.raises(ValidationError):
        UserUpdateAgeRequest(age_years=-5)

    with pytest.raises(ValidationError):
        UserUpdateAgeRequest(age_years=150)


# ===========================================================================
# 2. WHO 2024 Hemoglobin Interpretation & Population Boundaries
# ===========================================================================

def test_who_2024_adult_female_threshold():
    """Adult non-pregnant female (age >= 15) cutoff is 12.0 g/dL."""
    decision = compute_anemia_clinical_decision(
        hb_val=11.4,
        age_years=30,
        clinical_sex="female",
        is_pregnant=False
    )
    assert decision["reference_threshold_hb"] == 12.0
    assert "Mild Anemia" in decision["clinical_category"]
    assert decision["population_validation"]["is_validated_population"] is True


def test_who_2024_pregnant_female_thresholds():
    """Pregnant female cutoff is 11.0 g/dL (Trimesters 1 & 3) and 10.5 g/dL (Trimester 2)."""
    # 2nd trimester: cutoff 10.5 g/dL -> 10.8 g/dL is Non-Anemic
    dec_t2 = compute_anemia_clinical_decision(
        hb_val=10.8,
        age_years=26,
        clinical_sex="female",
        is_pregnant=True,
        pregnancy_trimester=2
    )
    assert dec_t2["reference_threshold_hb"] == 10.5
    assert "Non-Anemic Baseline" in dec_t2["clinical_category"]

    # 1st trimester: cutoff 11.0 g/dL -> 10.8 g/dL is Mild Anemia
    dec_t1 = compute_anemia_clinical_decision(
        hb_val=10.8,
        age_years=26,
        clinical_sex="female",
        is_pregnant=True,
        pregnancy_trimester=1
    )
    assert dec_t1["reference_threshold_hb"] == 11.0
    assert "Mild Anemia" in dec_t1["clinical_category"]


def test_who_2024_adult_male_threshold():
    """Adult male (age >= 15) cutoff is 13.0 g/dL."""
    dec_male = compute_anemia_clinical_decision(
        hb_val=12.5,
        age_years=40,
        clinical_sex="male"
    )
    assert dec_male["reference_threshold_hb"] == 13.0
    assert "Mild Anemia" in dec_male["clinical_category"]


def test_pediatric_population_validation_boundary():
    """Age < 18 must be flagged as pediatric and outside validated population."""
    decision = compute_anemia_clinical_decision(
        hb_val=11.2,
        age_years=12,
        clinical_sex="not_disclosed"
    )
    assert decision["population_validation"]["is_validated_population"] is False
    assert decision["population_validation"]["reason"] == "pediatric_under_18"
    assert "Pediatric" in decision["population_validation"]["clinical_notice_en"]


def test_geriatric_population_validation_boundary():
    """Age > 88 must be flagged as geriatric outside validated population."""
    decision = compute_anemia_clinical_decision(
        hb_val=12.2,
        age_years=92,
        clinical_sex="male"
    )
    assert decision["population_validation"]["is_validated_population"] is False
    assert decision["population_validation"]["reason"] == "geriatric_over_88"


# ===========================================================================
# 3. Personalized AI Service & Data Minimization Tests
# ===========================================================================

def test_personalized_ai_deterministic_fallback_when_consent_withheld():
    """Without external_ai_consent, service must return local deterministic bilingual analysis."""
    service = PersonalizedAIService()
    dummy_img = Image.new("RGB", (224, 224), color=(180, 80, 80))
    anemia_res = {
        "status": "available_research_only",
        "predicted_hb_g_dl": 11.5,
        "view_mean": 11.5,
        "confidence_interval_95": [9.45, 13.55]
    }
    clin_decision = compute_anemia_clinical_decision(11.5, age_years=35, clinical_sex="female")

    res = service.generate_personalized_explanation(
        image=dummy_img,
        age_years=35,
        clinical_sex="female",
        anemia_result=anemia_res,
        clinical_decision=clin_decision,
        external_ai_consent=False
    )

    assert res["status"] == "completed"
    assert res["provider"] == "local_deterministic"
    assert res["external_ai_consent"] is False
    assert "bilingual" in res
    assert "en" in res["bilingual"]
    assert "hi" in res["bilingual"]
    assert "visual_observations" in res["bilingual"]["en"]
    assert "reference_context" in res["bilingual"]["en"]
    assert "actionable_next_steps" in res["bilingual"]["en"]
    assert len(res["bilingual"]["en"]["actionable_next_steps"]) >= 2


def test_personalized_ai_payload_sanitization_guarantees_no_pii():
    """Payload construction must never contain patient name, phone, email, or DOB."""
    service = PersonalizedAIService()
    payload = service._build_sanitized_payload(
        age_years=45,
        clinical_sex="male",
        is_pregnant=False,
        pregnancy_trimester=None,
        anemia_result={"predicted_hb_g_dl": 13.2},
        slid_result={"status": "success", "prediction": "Normal"},
        jaundice_result={"status": "success", "prediction": "Normal"},
        quality_result={"is_acceptable": True},
        segmentation_result={"tissue_coverage_pct": 32.5},
        clinical_decision={"reference_threshold_hb": 13.0}
    )

    # Convert to string and assert no PII keys exist
    payload_str = str(payload).lower()
    for forbidden in ["full_name", "phone", "email", "date_of_birth", "dob", "address", "usr_"]:
        assert forbidden not in payload_str


# ===========================================================================
# 4. Database Persistence & Legacy Update Tests
# ===========================================================================

def test_database_user_profile_and_age_update():
    """UserProfile registration with age and subsequent update-age operation."""
    init_db()
    with SessionLocal() as db:
        test_email = f"test_{os.urandom(4).hex()}@research.org"
        user = register_user(
            db=db,
            full_name="Dr. Test User",
            phone="9123456789",
            email=test_email,
            age_years=29,
            clinical_sex="female",
            is_pregnant=False,
            external_ai_consent=True
        )
        assert user.age_years == 29
        assert user.clinical_sex == "female"
        assert user.external_ai_consent is True

        # Update age
        updated = update_user_profile_age(
            db=db,
            user_id=user.id,
            age_years=30,
            clinical_sex="female",
            is_pregnant=True,
            pregnancy_trimester=1
        )
        assert updated.age_years == 30
        assert updated.is_pregnant is True
        assert updated.pregnancy_trimester == 1


def test_database_screening_session_persists_demographics_and_ai():
    """ScreeningSession must persist patient_age_years, clinical_sex, and ai_explanation."""
    init_db()
    with SessionLocal() as db:
        user = register_user(
            db=db,
            full_name="Session Patient",
            phone="9000000000",
            email=f"sess_{os.urandom(4).hex()}@test.org",
            age_years=52,
            clinical_sex="male"
        )
        ai_exp = {
            "status": "completed",
            "provider": "local_deterministic",
            "bilingual": {"en": {"summary": "Screening summary test."}}
        }
        sess = create_session(
            db=db,
            selected_module="unified",
            user_id=user.id,
            patient_age_years=52,
            patient_clinical_sex="male",
            ai_explanation=ai_exp
        )
        assert sess.patient_age_years == 52
        assert sess.patient_clinical_sex == "male"
        assert sess.ai_explanation["provider"] == "local_deterministic"

        # Verify to_dict output
        sess_dict = sess.to_dict()
        assert sess_dict["patient_age_years"] == 52
        assert sess_dict["patient_clinical_sex"] == "male"
        assert sess_dict["ai_explanation"] is not None


# ===========================================================================
# 5. Strict 2-Page PDF Report Generation Tests
# ===========================================================================

def test_pdf_report_displays_age_and_strictly_two_pages():
    """PDF report must contain age and remain strictly 2 pages for both EN and HI."""
    session_data = {
        "session_id": "test_sess_001",
        "capture_source": "upload",
        "timestamp_utc": "2026-10-14T20:00:00Z",
        "patient_age_years": 34,
        "patient_clinical_sex": "female",
        "anemia_result": {
            "view_mean": 11.2,
            "prediction_value": 11.2,
            "clinical_decision": compute_anemia_clinical_decision(11.2, age_years=34, clinical_sex="female")
        },
        "slid_result": {
            "prediction": [{"predicted_class": "control_normal", "confidence": 0.98}]
        },
        "jaundice_result": {
            "prediction": [{"predicted_class": "normal", "confidence": 0.95}]
        },
        "personalized_ai": {
            "status": "completed",
            "provider": "local_deterministic",
            "model_used": "Deterministic WHO 2024",
            "bilingual": {
                "en": {
                    "summary": "Mild eyelid mucosal pallor observed.",
                    "reference_context": "Evaluated against WHO 2024 female cutoff of 12.0 g/dL.",
                    "population_status": "Within validated age cohort (18-88)."
                },
                "hi": {
                    "summary": "पलक के अंदरूनी हिस्से में हल्का पीलापन देखा गया।",
                    "reference_context": "WHO 2024 महिला मानक 12.0 g/dL के अनुसार मूल्यांकन।",
                    "population_status": "मान्य आयु वर्ग (18-88) के भीतर।"
                }
            }
        }
    }
    user_data = {
        "full_name": "Anita Verma",
        "phone": "+91 9876543210",
        "email": "anita@example.com",
        "age_years": 34,
        "clinical_sex": "female"
    }

    # Test English Report
    pdf_bytes_en = build_two_page_pdf(session_data, user_data=user_data, language="en")
    reader_en = PdfReader(io.BytesIO(pdf_bytes_en))
    assert len(reader_en.pages) == 2, f"English PDF must be exactly 2 pages, got {len(reader_en.pages)}"
    text_p1_en = reader_en.pages[0].extract_text()
    assert "34 Yrs" in text_p1_en
    assert "Anita Verma" in text_p1_en

    # Test Hindi Report
    pdf_bytes_hi = build_two_page_pdf(session_data, user_data=user_data, language="hi")
    reader_hi = PdfReader(io.BytesIO(pdf_bytes_hi))
    assert len(reader_hi.pages) == 2, f"Hindi PDF must be exactly 2 pages, got {len(reader_hi.pages)}"
    text_p1_hi = reader_hi.pages[0].extract_text()
    assert "34 Yrs" in text_p1_hi
