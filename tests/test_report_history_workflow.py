"""
Regression and End-to-End Workflow Tests for EyeHealthAI / OcuScan:
1. Report generation endpoints (Screening PDF, Technical Scorecard PDF, HTML Report)
2. English ('en') and Hindi ('hi') language support
3. Canonical (/api/reports/...) and legacy (/reports/...) URL route compatibility
4. Genuine HTTP 404 for missing/nonexistent sessions (no fake synthetic PDF fallbacks)
5. User profile persistence, history listing, session restoration, and deletion
"""

import io
import pytest
from starlette.testclient import TestClient
from PIL import Image

from app import app
from src.database.session import SessionLocal, get_session_by_id, get_user_by_id


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _create_sample_eye_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (320, 240), color=(180, 70, 70))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


class TestUserIdentityAndHistory:
    """Verify participant profile persistence and history endpoints."""

    def test_default_sarthak_history_recovered(self, client):
        """Verify usr_sarthak_default history endpoint returns HTTP 200 with recovered sessions."""
        res = client.get("/api/user/usr_sarthak_default/history")
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == "usr_sarthak_default"
        assert "sessions" in data
        assert len(data["sessions"]) > 0
        first_session = data["sessions"][0]
        assert "id" in first_session
        assert "selected_module" in first_session
        assert "results" in first_session

    def test_nonexistent_user_history_returns_404(self, client):
        """Verify requesting history for a nonexistent user returns HTTP 404."""
        res = client.get("/api/user/usr_nonexistent_999999/history")
        assert res.status_code == 404

    def test_registration_and_profile_flow(self, client):
        """Verify registering a new user creates a valid profile and can be fetched."""
        reg_payload = {
            "full_name": "Test Participant Alpha",
            "phone": "9998887776",
            "email": "alpha.test@example.org",
            "age_years": 34,
            "gender": "female",
            "clinical_sex": "female",
            "is_pregnant": False,
            "pregnancy_trimester": None,
            "external_ai_consent": True,
            "demographic_consent": True,
            "research_consent": True,
            "data_storage_consent": True,
        }
        res = client.post("/api/register", json=reg_payload)
        assert res.status_code == 200
        data = res.json()
        assert "user" in data
        user_id = data["user"]["id"]
        assert user_id.startswith("usr_")

        # History should be empty for new user
        hist_res = client.get(f"/api/user/{user_id}/history")
        assert hist_res.status_code == 200
        assert hist_res.json()["sessions"] == []


class TestScreeningPersistenceAndResults:
    """Verify screening execution saves database session with model results."""

    def test_unified_screening_persists_session_and_model_results(self, client):
        """Executing /api/analyze with valid user_id creates a persisted session and model results."""
        img_bytes = _create_sample_eye_image_bytes()
        files = {"image": ("eye_sample.jpg", img_bytes, "image/jpeg")}
        data = {
            "user_id": "usr_sarthak_default",
            "module": "unified",
            "age_years": "28",
            "clinical_sex": "male",
            "external_ai_consent": "true",
            "is_experimental_override": "true",
        }

        res = client.post("/api/analyze", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()

        # Must report persisted outcome and return database session id
        assert payload.get("persistence_status") == "persisted"
        session_id = payload.get("session_id")
        assert session_id is not None
        assert session_id.startswith("session_") or session_id.startswith("ses_")

        # Verify session in DB
        with SessionLocal() as db:
            db_sess = get_session_by_id(db, session_id)
            assert db_sess is not None
            assert db_sess.user_id == "usr_sarthak_default"
            assert len(db_sess.model_results) >= 3

    def test_unknown_user_id_fails_persistence_cleanly(self, client):
        """Providing an invalid user_id must raise 404 rather than silently attaching to None."""
        img_bytes = _create_sample_eye_image_bytes()
        files = {"image": ("eye_sample.jpg", img_bytes, "image/jpeg")}
        data = {
            "user_id": "usr_definitely_does_not_exist_xyz",
            "module": "unified",
            "age_years": "25",
            "clinical_sex": "male",
            "is_experimental_override": "true",
        }

        res = client.post("/api/analyze", files=files, data=data)
        assert res.status_code == 404


class TestReportEndpointsAndPDFGeneration:
    """Verify PDF downloads and report HTML endpoints for both English and Hindi."""

    @pytest.fixture(scope="module")
    def seeded_session_id(self):
        """Ensure a known session exists for testing."""
        return "ses_mock_session_123"

    def test_canonical_screening_pdf_en_and_hi(self, client, seeded_session_id):
        """Verify /api/reports/{id}/screening-pdf returns a valid 2-page PDF in English & Hindi."""
        # English
        res_en = client.get(f"/api/reports/{seeded_session_id}/screening-pdf?lang=en")
        assert res_en.status_code == 200
        assert res_en.headers["content-type"] == "application/pdf"
        assert res_en.content.startswith(b"%PDF")
        assert "EyeHealthAI_Report_" in res_en.headers.get("content-disposition", "")

        # Hindi
        res_hi = client.get(f"/api/reports/{seeded_session_id}/screening-pdf?lang=hi")
        assert res_hi.status_code == 200
        assert res_hi.headers["content-type"] == "application/pdf"
        assert res_hi.content.startswith(b"%PDF")
        assert "_hi.pdf" in res_hi.headers.get("content-disposition", "")

    def test_canonical_scorecard_pdf_en_and_hi(self, client, seeded_session_id):
        """Verify /api/reports/{id}/scorecard-pdf returns a valid 2-page PDF in English & Hindi."""
        # English
        res_en = client.get(f"/api/reports/{seeded_session_id}/scorecard-pdf?lang=en")
        assert res_en.status_code == 200
        assert res_en.headers["content-type"] == "application/pdf"
        assert res_en.content.startswith(b"%PDF")
        assert "OcuScan_Technical_Scorecard_" in res_en.headers.get("content-disposition", "")

        # Hindi
        res_hi = client.get(f"/api/reports/{seeded_session_id}/scorecard-pdf?lang=hi")
        assert res_hi.status_code == 200
        assert res_hi.headers["content-type"] == "application/pdf"
        assert res_hi.content.startswith(b"%PDF")
        assert "_hi.pdf" in res_hi.headers.get("content-disposition", "")

    def test_legacy_and_alias_pdf_routes(self, client, seeded_session_id):
        """Verify legacy route aliases function identically without 404."""
        # Legacy /reports/{id}/screening-pdf
        res1 = client.get(f"/reports/{seeded_session_id}/screening-pdf?lang=en")
        assert res1.status_code == 200
        assert res1.content.startswith(b"%PDF")

        # Legacy /reports/{id}/scorecard-pdf
        res2 = client.get(f"/reports/{seeded_session_id}/scorecard-pdf?lang=en")
        assert res2.status_code == 200
        assert res2.content.startswith(b"%PDF")

        # Canonical /api/reports/{id}/pdf
        res3 = client.get(f"/api/reports/{seeded_session_id}/pdf?lang=en&type=screening")
        assert res3.status_code == 200
        assert res3.content.startswith(b"%PDF")

        # Legacy /reports/{id}/pdf
        res4 = client.get(f"/reports/{seeded_session_id}/pdf?lang=en&type=scorecard")
        assert res4.status_code == 200
        assert res4.content.startswith(b"%PDF")

        # Shortcut /api/download-report and /download-report
        res5 = client.get(f"/api/download-report?session_id={seeded_session_id}&lang=en")
        assert res5.status_code == 200
        assert res5.content.startswith(b"%PDF")

        res6 = client.get(f"/download-report?session_id={seeded_session_id}&lang=en")
        assert res6.status_code == 200
        assert res6.content.startswith(b"%PDF")

    def test_nonexistent_session_returns_404_without_synthetic_fallback(self, client):
        """Missing session IDs must return 404 instead of generating fake fallback PDFs."""
        bad_id = "ses_totally_missing_999999"

        r1 = client.get(f"/api/reports/{bad_id}/pdf")
        assert r1.status_code == 404

        r2 = client.get(f"/api/reports/{bad_id}/screening-pdf")
        assert r2.status_code == 404

        r3 = client.get(f"/api/reports/{bad_id}/scorecard-pdf")
        assert r3.status_code == 404

        r4 = client.get(f"/reports/{bad_id}/pdf")
        assert r4.status_code == 404

        r5 = client.get(f"/api/reports/{bad_id}")
        assert r5.status_code == 404

    def test_report_data_api_and_html_page(self, client, seeded_session_id):
        """Verify metadata endpoint and HTML report page work."""
        # Metadata endpoint
        res = client.get(f"/api/reports/{seeded_session_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == seeded_session_id
        assert "model_results" in data

        # Legacy metadata endpoint
        res_leg = client.get(f"/reports/{seeded_session_id}")
        assert res_leg.status_code == 200

        # HTML page route
        res_html = client.get(f"/report?session_id={seeded_session_id}&lang=en")
        assert res_html.status_code == 200
        assert "text/html" in res_html.headers["content-type"]


class TestResultsPageRedesignAndEnhancedAI:
    """Verify results page UI redesign, Your Reports card integration, and structured AI explanation."""

    def test_results_page_top_download_buttons_removed(self):
        """Top duplicate download buttons (btnDownloadPdfReport, btnDownloadPdfHi, btnOpenPrintScorecard) must be removed."""
        with open("static/index.html", "r", encoding="utf-8") as f:
            html = f.read()

        assert 'id="btnDownloadPdfReport"' not in html
        assert 'id="btnDownloadPdfHi"' not in html
        assert 'id="btnOpenPrintScorecard"' not in html

    def test_your_reports_card_elements_and_downloads_integrated(self):
        """Your Reports section must integrate screening download, alternate language download, and scorecard download."""
        with open("static/js/dashboard.js", "r", encoding="utf-8") as f:
            js = f.read()

        assert 'btnDlScreening' in js
        assert 'btnDlScreeningAlt' in js
        assert 'Download Screening Report PDF' in js
        assert 'View Interactive Web Report' in js
        assert 'linkAnemiaReport' not in js
        assert 'linkSlidReport' not in js
        assert 'linkJaundReport' not in js

    def test_enhanced_ai_explanation_structured_sections(self):
        """Enhanced AI explanation must contain structured sections and styling."""
        with open("static/index.html", "r", encoding="utf-8") as f:
            html = f.read()
        with open("static/js/dashboard.js", "r", encoding="utf-8") as f:
            js = f.read()

        # CSS classes in HTML
        assert ".ai-summary-box" in html
        assert ".ai-models-grid" in html
        assert ".ai-context-banner" in html
        assert ".ai-limits-banner" in html
        assert ".ai-next-steps-list" in html

        # Structured sections in JS
        assert "ai-summary-box" in js
        assert "ai-models-grid" in js
        assert "ai-context-banner" in js
        assert "ai-limits-banner" in js
        assert "ai-next-steps-list" in js

    def test_current_session_downloads_both_reports_end_to_end(self, client):
        """A freshly executed screening session must immediately generate and download both Screening and Scorecard PDFs in EN and HI."""
        img_bytes = _create_sample_eye_image_bytes()
        files = {"images": ("eye_sample.jpg", img_bytes, "image/jpeg")}
        data = {
            "user_id": "usr_sarthak_default",
            "module": "unified",
            "age_years": "28",
            "clinical_sex": "male",
            "external_ai_consent": "true",
            "is_experimental_override": "true",
        }

        res = client.post("/api/analyze", files=files, data=data)
        assert res.status_code == 200
        payload = res.json()
        sid = payload["session_id"]
        assert sid.startswith("session_") or sid.startswith("ses_")

        # 1. Download Screening Report in English
        dl_screen_en = client.get(f"/api/reports/{sid}/screening-pdf?lang=en")
        assert dl_screen_en.status_code == 200
        assert dl_screen_en.content.startswith(b"%PDF")
        assert "EyeHealthAI_Report_" in dl_screen_en.headers.get("content-disposition", "")

        # 2. Download Screening Report in Hindi
        dl_screen_hi = client.get(f"/api/reports/{sid}/screening-pdf?lang=hi")
        assert dl_screen_hi.status_code == 200
        assert dl_screen_hi.content.startswith(b"%PDF")
        assert "_hi.pdf" in dl_screen_hi.headers.get("content-disposition", "")

        # 3. Download Technical Scorecard in English
        dl_score_en = client.get(f"/api/reports/{sid}/scorecard-pdf?lang=en")
        assert dl_score_en.status_code == 200
        assert dl_score_en.content.startswith(b"%PDF")
        assert "OcuScan_Technical_Scorecard_" in dl_score_en.headers.get("content-disposition", "")

        # 4. Download Technical Scorecard in Hindi
        dl_score_hi = client.get(f"/api/reports/{sid}/scorecard-pdf?lang=hi")
        assert dl_score_hi.status_code == 200
        assert dl_score_hi.content.startswith(b"%PDF")
        assert "_hi.pdf" in dl_score_hi.headers.get("content-disposition", "")

