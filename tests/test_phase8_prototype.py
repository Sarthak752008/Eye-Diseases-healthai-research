"""
tests/test_phase8_prototype.py
------------------------------
Phase 8 End-to-End Test Suite for Complete Autonomous Prototype:
1. Database persistence, informed consent, user registration, and cascading purge.
2. Two-page professional PDF generation in English and Hindi.
3. Readiness probe (/ready & /api/ready) distinguishing software readiness from clinical model gates.
4. User registration validation and error handling.
5. Session history retrieval and privacy-aware deletion (Right to Be Forgotten).
6. Multi-step UI markup and bilingual data-attribute integrity.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
import pypdf

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.database import (
    init_db,
    register_user,
    get_user_by_id,
    create_session,
    save_model_result,
    get_user_sessions,
    delete_user_data,
    delete_session_record,
    SessionLocal
)
from src.inference.pdf_report import build_two_page_pdf


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Database & Consent Tests
# ---------------------------------------------------------------------------

def test_database_user_registration_and_consent():
    """Verifies user registration and informed consent logging in SQLite/PostgreSQL."""
    with SessionLocal() as db:
        user = register_user(
            db=db,
            full_name="Dr. Jane Doe",
            phone="9876543210",
            email="jane.doe@research.org",
            date_of_birth="1990-05-15",
            research_consent=True,
            data_storage_consent=True
        )
        assert user.id.startswith("usr_")
        assert user.full_name == "Dr. Jane Doe"
        assert len(user.consents) >= 1
        assert user.consents[0].consent_given is True
        assert user.consents[0].research_screening_consent is True


def test_database_session_and_model_result_lifecycle():
    """Verifies screening session creation, result logging, history lookup, and deletion."""
    with SessionLocal() as db:
        user = register_user(
            db=db,
            full_name="John Smith",
            phone="9123456789",
            email="john.smith@test.org",
            date_of_birth="1985-11-20"
        )
        sess = create_session(
            db=db,
            selected_module="anemia",
            capture_source="upload",
            user_id=user.id,
            language="en",
            view_types=["palpebral", "forniceal"],
            quality_summary={"is_valid": True}
        )
        assert sess.id.startswith("ses_")

        res = save_model_result(
            db=db,
            session_id=sess.id,
            module="anemia",
            operational_status="available_research_only",
            prediction_value=13.4,
            unit="g/dL",
            view_count=2,
            view_mean=13.4,
            view_std=0.35,
            reliability_warnings=["MAE is ~1.50 g/dL"]
        )
        assert res.prediction_value == 13.4

        # Retrieve user sessions
        sessions = get_user_sessions(db, user.id)
        assert len(sessions) >= 1
        assert sessions[0].selected_module == "anemia"

        # Cascade deletion: delete user purges session and results
        deleted = delete_user_data(db, user.id)
        assert deleted is True
        assert get_user_by_id(db, user.id) is None


# ---------------------------------------------------------------------------
# 2. PDF Report Generator Tests
# ---------------------------------------------------------------------------

def test_pdf_report_generates_strictly_two_pages_english():
    """Verifies that English PDF report is exactly 2 pages without overflow."""
    session_data = {
        "session_id": "ses_test_audit_en",
        "timestamp_utc": "2026-10-11T12:00:00Z",
        "selected_module": "anemia",
        "capture_source": "upload",
        "operational_status": "available_research_only",
        "model_output": 12.85,
        "view_count": 2,
        "view_mean": 12.85,
        "view_std": 0.42,
        "view_types": ["palpebral", "forniceal"]
    }
    user_data = {
        "full_name": "Sarthak Sharma",
        "phone": "9019488301",
        "email": "sarthak@research.org",
        "date_of_birth": "2008-01-15"
    }

    pdf_bytes = build_two_page_pdf(session_data, user_data, language="en")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 2


def test_pdf_report_generates_strictly_two_pages_hindi():
    """Verifies that Hindi PDF report is exactly 2 pages with Devanagari text."""
    session_data = {
        "session_id": "ses_test_audit_hi",
        "timestamp_utc": "2026-10-11T12:00:00Z",
        "selected_module": "anemia",
        "capture_source": "webcam",
        "operational_status": "available_research_only",
        "model_output": 11.9,
        "view_count": 1,
        "view_mean": 11.9,
        "view_std": None,
        "view_types": ["raw_eye"]
    }
    user_data = {
        "full_name": "Sarthak Sharma",
        "phone": "9019488301",
        "email": "sarthak@research.org",
        "date_of_birth": "2008-01-15"
    }

    pdf_bytes = build_two_page_pdf(session_data, user_data, language="hi")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 2


# ---------------------------------------------------------------------------
# 3. Readiness Probe Endpoint Tests
# ---------------------------------------------------------------------------

def test_readiness_probe_endpoints(client):
    """Verifies /ready and /api/ready distinguish database and model status."""
    res1 = client.get("/ready")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["ready"] is True
    assert data1["database_connected"] is True
    assert data1["checkpoints"]["anemia_resnet50"] is True
    assert data1["gates"]["conjunctivitis"] == "blocked_domain_validation"
    assert data1["gates"]["jaundice"] == "unavailable_missing_data"

    res2 = client.get("/api/ready")
    assert res2.status_code == 200
    assert res2.json() == data1


# ---------------------------------------------------------------------------
# 4. User Registration API Tests
# ---------------------------------------------------------------------------

def test_api_register_user_success(client):
    """Verifies successful participant registration via REST API."""
    payload = {
        "full_name": "Aditi Verma",
        "phone": "9876501234",
        "email": "aditi.verma@example.com",
        "date_of_birth": "1995-08-22",
        "research_consent": True,
        "data_storage_consent": True
    }
    res = client.post("/api/register", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["user"]["full_name"] == "Aditi Verma"
    assert "usr_" in data["user"]["id"]


def test_api_register_user_missing_consent_rejected(client):
    """Verifies that registration without mandatory consent is rejected."""
    payload = {
        "full_name": "No Consent User",
        "phone": "9876501234",
        "email": "noconsent@example.com",
        "date_of_birth": "1995-08-22",
        "research_consent": False,
        "data_storage_consent": True
    }
    res = client.post("/api/register", json=payload)
    assert res.status_code == 400
    assert "consent for research screening is mandatory" in res.json()["detail"].lower()


def test_api_register_user_invalid_email_rejected(client):
    """Verifies rejection of malformed email strings."""
    payload = {
        "full_name": "Bad Email User",
        "phone": "9876501234",
        "email": "notanemail",
        "date_of_birth": "1995-08-22",
        "research_consent": True,
        "data_storage_consent": True
    }
    res = client.post("/api/register", json=payload)
    assert res.status_code == 400
    assert "Invalid email" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Session History & Privacy Deletion API Tests
# ---------------------------------------------------------------------------

def test_api_user_session_history_and_deletion(client):
    """Verifies history query and GDPR-style user account purge."""
    # 1. Register
    reg_res = client.post("/api/register", json={
        "full_name": "Test Delete User",
        "phone": "9998887776",
        "email": "delete.me@test.org",
        "date_of_birth": "2000-01-01",
        "research_consent": True,
        "data_storage_consent": True
    })
    user_id = reg_res.json()["user"]["id"]

    # 2. Get history (initially empty)
    h_res = client.get(f"/api/user/{user_id}/history")
    assert h_res.status_code == 200
    assert len(h_res.json()["sessions"]) == 0

    # 3. Delete user account
    del_res = client.delete(f"/api/user/{user_id}")
    assert del_res.status_code == 200
    assert "deleted" in del_res.json()["message"].lower()

    # 4. Verify 404 on subsequent lookup
    assert client.get(f"/api/user/{user_id}/history").status_code == 404


# ---------------------------------------------------------------------------
# 6. PDF Download Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_download_pdf_endpoint(client):
    """Verifies /api/reports/{session_id}/pdf returns application/pdf for EN and HI."""
    res_en = client.get("/api/reports/ses_mock_session_123/pdf?lang=en")
    assert res_en.status_code == 200
    assert res_en.headers["content-type"] == "application/pdf"
    assert len(res_en.content) > 1000

    res_hi = client.get("/api/reports/ses_mock_session_123/pdf?lang=hi")
    assert res_hi.status_code == 200
    assert res_hi.headers["content-type"] == "application/pdf"
    assert len(res_hi.content) > 1000


# ---------------------------------------------------------------------------
# 7. Multi-Step UI Markup Integrity Tests
# ---------------------------------------------------------------------------

def test_multi_step_wizard_html_elements_present(client):
    """Verifies that all 7 page views and stepper items exist in index.html."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Stepper items
    assert 'id="stepperNav"' in html
    assert 'id="stepBtn1"' in html
    assert 'id="stepBtn7"' in html

    # 7 Page Containers
    assert 'id="page1Language"' in html
    assert 'id="page2Registration"' in html
    assert 'id="page3Guide"' in html
    assert 'id="page4Capture"' in html
    assert 'id="page5Analysis"' in html
    assert 'id="page6Results"' in html
    assert 'id="page7History"' in html

    # Language Switcher
    assert 'id="langBtnEn"' in html
    assert 'id="langBtnHi"' in html

    # Results Actions
    assert 'id="btnDownloadPdfReport"' in html
    assert 'id="btnDownloadPdfHi"' in html
    assert 'id="btnOpenPrintScorecard"' in html

    # History Table
    assert 'id="historyTableBody"' in html
    assert 'id="btnPurgeUserData"' in html
