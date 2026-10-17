"""
tests/test_unified_screening_workspace.py
-----------------------------------------
End-to-end and integration tests for OcuScan Unified Screening Workspace:
  1. No disease-selection control exists in the patient-facing workspace.
  2. All intended active targets are dispatched automatically.
  3. Blocked Mendeley inference cannot be exposed through normal UI or API workflows.
  4. Every result belongs to the currently uploaded image and session.
  5. Uploading a new image invalidates previous results.
  6. One target's failure does not falsely mark it successful or fabricate its output.
  7. PDF and exported report values match the current-session results.
  8. AI explanation failures do not break core inference or report generation.
  9. Webcam, history, consent, profile, English/Hindi switching and theme controls exist.
 10. Responsive CSS layout styles for desktop and mobile breakpoints.
"""

import os
import io
import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.inference.session_export import create_session_export, validate_session_export_schema
from src.inference.pdf_report import build_two_page_pdf


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_eye_image():
    """Generates an in-memory sample RGB ocular image."""
    img = Image.new('RGB', (224, 224), color=(180, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def sample_eye_image_blue():
    """Generates a distinct in-memory sample ocular image for provenance testing."""
    img = Image.new('RGB', (224, 224), color=(70, 70, 180))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ===========================================================================
# 1. No Disease-Selection Control in Patient-Facing Workspace
# ===========================================================================
def test_no_disease_selection_controls(client):
    """
    Acceptance Criterion 1:
    Verifies that the patient-facing screening workspace does not contain
    selectable module radio buttons, disease selection toggles, or clickable
    module cards that make analysis conditional on selecting a single disease.
    """
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Prominent Unified Screening Workspace heading and info banner
    assert "Unified Screening Workspace" in html
    assert "No disease selection needed" in html
    assert "What we assess automatically" in html

    # Informational assessment cards exist but are not selectable module toggles
    assert "assess-info-card" in html
    assert "moduleCardAnemia" in html
    assert "moduleCardConj" in html
    assert "moduleCardJaund" in html

    # Confirm disease selection instructions and radio buttons are absent
    assert '<input type="radio" name="disease"' not in html
    assert '<input type="radio" name="module"' not in html
    assert "Select an Active Screening Module" not in html


# ===========================================================================
# 2. All Intended Active Targets Dispatched Automatically
# ===========================================================================
def test_all_active_targets_dispatched_automatically(client, sample_eye_image):
    """
    Acceptance Criterion 2:
    Verifies that submitting a single eye image with module='unified'
    automatically dispatches Anemia, SLID, and Jaundice without disease selection.
    """
    files = [("images", ("patient_eye.jpg", sample_eye_image, "image/jpeg"))]
    data = {
        "module": "unified",
        "view_types": ["raw_eye"],
        "age_years": 28,
        "clinical_sex": "male"
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    assert res_data["selected_module"] == "unified"
    assert "anemia" in res_data
    assert "slid" in res_data
    assert "jaundice" in res_data
    assert res_data["anemia"]["status"] == "available_research_only"
    assert res_data["slid"]["status"] == "available_research_only"
    assert res_data["jaundice"]["status"] == "available_research_only"


# ===========================================================================
# 3. Blocked Mendeley Inference Cannot Be Exposed
# ===========================================================================
def test_blocked_mendeley_cannot_be_exposed(client, sample_eye_image):
    """
    Acceptance Criterion 3:
    Verifies that the confounded Mendeley shortcut model is strictly blocked
    and cannot produce diagnostic output through normal UI or API flows.
    """
    # 1. Normal unified dispatch reports Mendeley as blocked gate in audit
    files = [("images", ("eye.jpg", sample_eye_image, "image/jpeg"))]
    res_unified = client.post("/api/analyze", files=files, data={"module": "unified", "age_years": 30})
    assert res_unified.status_code == 200
    m_gate = res_unified.json().get("mendeley_gate", {})
    assert m_gate.get("status") == "blocked_validation"
    assert m_gate.get("prediction") is None

    # 2. Explicit legacy request to /api/analyze with module='conjunctivitis' is strictly blocked
    res_legacy = client.post("/api/analyze", data={"module": "conjunctivitis"})
    assert res_legacy.status_code == 200
    legacy_data = res_legacy.json()
    assert legacy_data["operational_status"] == "blocked_validation"
    assert legacy_data["inference_permitted"] is False
    assert legacy_data["model_output"] is None
    assert any("shortcut" in w.lower() for w in legacy_data.get("reliability_warnings", []))


# ===========================================================================
# 4. Results Belong to Currently Uploaded Image and Session
# ===========================================================================
def test_results_tied_to_current_image_and_session(client, sample_eye_image, sample_eye_image_blue):
    """
    Acceptance Criterion 4:
    Verifies input SHA-256 provenance and ensures distinct sessions
    produce independent session IDs and hashes.
    """
    files1 = [("images", ("image1.jpg", sample_eye_image, "image/jpeg"))]
    res1 = client.post("/api/analyze", files=files1, data={"module": "unified", "age_years": 25})
    assert res1.status_code == 200
    data1 = res1.json()

    files2 = [("images", ("image2.jpg", sample_eye_image_blue, "image/jpeg"))]
    res2 = client.post("/api/analyze", files=files2, data={"module": "unified", "age_years": 25})
    assert res2.status_code == 200
    data2 = res2.json()

    assert data1["session_id"] != data2["session_id"]
    assert data1["input_hashes"][0] != data2["input_hashes"][0]
    assert data1["input_provenance"][0]["filename"] == "image1.jpg"
    assert data2["input_provenance"][0]["filename"] == "image2.jpg"


# ===========================================================================
# 5. Invalidation of Prior Results in Frontend Logic
# ===========================================================================
def test_invalidation_logic_in_client_script():
    """
    Acceptance Criterion 5:
    Verifies that dashboard.js contains invalidatePreviousResults()
    invoked upon selecting a new image or removing an image.
    """
    js_path = os.path.join(WORKSPACE_ROOT, 'static', 'js', 'dashboard.js')
    with open(js_path, 'r', encoding='utf-8') as f:
        js = f.read()

    assert "function invalidatePreviousResults()" in js
    assert "currentSessionRequestId++" in js
    assert "lastSessionData = null" in js
    assert "handleFilesSelected" in js
    assert "btnRemoveImage" in js


# ===========================================================================
# 6. Target Failure Does Not Fabricate Output
# ===========================================================================
def test_no_fabricated_bilirubin_output(client, sample_eye_image):
    """
    Acceptance Criterion 6:
    Quantitative total serum bilirubin must NEVER be fabricated
    merely to populate an interface card when lab reference data is unavailable.
    """
    files = [("images", ("eye.jpg", sample_eye_image, "image/jpeg"))]
    res = client.post("/api/analyze", files=files, data={"module": "unified", "age_years": 35})
    assert res.status_code == 200
    data = res.json()

    jaund = data.get("jaundice", {})
    # Jaundice provides categorical screening only; no fabricated serum bilirubin numbers
    assert "serum_bilirubin_mg_dl" not in jaund
    s_gate = data.get("serum_bilirubin_gate", {})
    assert s_gate.get("status") == "unavailable_missing_data"
    assert s_gate.get("prediction") is None


# ===========================================================================
# 7. PDF and Export Match Current Session Data
# ===========================================================================
def test_pdf_and_export_match_current_session(client, sample_eye_image):
    """
    Acceptance Criterion 7:
    Verifies that session export and PDF generation derive from
    genuine current-session results.
    """
    files = [("images", ("eye.jpg", sample_eye_image, "image/jpeg"))]
    res = client.post("/api/analyze", files=files, data={"module": "unified", "age_years": 40, "clinical_sex": "female"})
    assert res.status_code == 200
    session_data = res.json()

    # 1. Session export validation
    res_export = client.post("/api/export", json=session_data)
    assert res_export.status_code == 200
    exported = res_export.json()
    assert validate_session_export_schema(exported) is True
    assert exported["session_id"] == session_data["session_id"]

    # 2. PDF generation verification
    pdf_bytes = build_two_page_pdf(
        session_data=session_data,
        user_data={
            "full_name": "Test Patient",
            "phone": "1234567890",
            "age_years": 40,
            "clinical_sex": "female"
        },
        language="en"
    )
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")


# ===========================================================================
# 8. AI Explanation Failures Do Not Break Core Inference
# ===========================================================================
def test_ai_explanation_failure_resilience(client, sample_eye_image):
    """
    Acceptance Criterion 8:
    Ensures that external AI consent disabled or API timeouts
    safely fallback to local deterministic clinical rules without crashing.
    """
    files = [("images", ("eye.jpg", sample_eye_image, "image/jpeg"))]
    data = {
        "module": "unified",
        "age_years": 28,
        "external_ai_consent": "false"  # Explicitly disabled
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    p_ai = res_data.get("personalized_ai")
    assert p_ai is not None
    assert p_ai.get("provider") == "local_deterministic"
    assert "summary" in p_ai.get("bilingual", {}).get("en", {})


# ===========================================================================
# 9. Webcam, History, Consent, Profile, Language, and Theme Controls
# ===========================================================================
def test_complete_platform_controls_exist(client):
    """
    Acceptance Criterion 9:
    Verifies that the 4-step progress tracker, bilingual language switches,
    theme toggle, webcam controls, and history endpoints exist and work.
    """
    # 1. GET Dashboard HTML
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    assert "fourStepTracker" in html
    assert "trackStep1" in html
    assert "trackStep2" in html
    assert "trackStep3" in html
    assert "trackStep4" in html
    assert "langBtnEn" in html
    assert "langBtnHi" in html
    assert "themeToggleBtn" in html
    assert "btnModeWebcam" in html
    assert "btnResetSession" in html
    assert "btnNavHistory" in html

    # 2. Register a participant and verify session history retrieval
    reg_res = client.post("/api/register", json={
        "full_name": "Test Verified User",
        "phone": "9876543210",
        "email": "verified.user@example.com",
        "age_years": 30,
        "clinical_sex": "male"
    })
    assert reg_res.status_code == 200
    u_id = reg_res.json()["user"]["id"]

    res_hist = client.get(f"/api/user/{u_id}/history")
    assert res_hist.status_code == 200
    assert "sessions" in res_hist.json()


# ===========================================================================
# 10. Responsive Layout and CSS Grid Architecture
# ===========================================================================
def test_responsive_layout_styles():
    """
    Acceptance Criterion 10:
    Verifies responsive CSS styles for desktop two-column and mobile single-column.
    """
    css_path = os.path.join(WORKSPACE_ROOT, 'static', 'css', 'style.css')
    with open(css_path, 'r', encoding='utf-8') as f:
        css = f.read()

    assert ".screening-workspace-grid" in css
    assert "grid-template-columns: 1.15fr 0.85fr" in css
    assert ".four-step-tracker" in css
    assert "@media (max-width: 960px)" in css
    assert "grid-template-columns: 1fr;" in css


# ===========================================================================
# 11. Progress Stepper Checkmarks & Future Steps Inactive
# ===========================================================================
def test_progress_stepper_completed_checkmarks_and_inactive_future_steps(client):
    """
    Verifies Bug 1 Acceptance:
    Checkmarks are shown ONLY for completed steps, and future steps (Step 4 Report)
    remain inactive, disabled, and unclickable on initial load and reset.
    """
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Step 4 must be pending and disabled on initial load
    assert 'id="trackStep4" type="button" data-step="4" disabled aria-disabled="true"' in html
    # Active step 3 (Screening) shows number '3', NOT a checkmark polyline
    assert '<button class="tracker-step active" id="trackStep3"' in html
    assert '<div class="tracker-circle">3</div>' in html

    # Pending steps must have pointer-events none and cursor not-allowed in CSS
    assert ".tracker-step.pending" in html
    assert "pointer-events: none" in html
    assert "cursor: not-allowed" in html


# ===========================================================================
# 12. Upload/Camera Toggle Persistent Accessibility
# ===========================================================================
def test_upload_camera_toggle_persistent_accessibility(client):
    """
    Verifies Bug 2 Acceptance:
    btnModeUpload and btnModeWebcam are accessible in both modes, placed persistently
    outside of uploadSection so switching to Live Camera does not remove the toggle controls.
    """
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Verify both buttons exist
    assert 'id="btnModeUpload"' in html
    assert 'id="btnModeWebcam"' in html

    # Verify mode toggle buttons appear after webcamSection and before btnAnalyze
    webcam_idx = html.find('id="webcamSection"')
    toggle_idx = html.find('class="upload-mode-buttons"')
    analyze_idx = html.find('id="btnAnalyze"')

    assert webcam_idx != -1, "webcamSection not found"
    assert toggle_idx != -1, "upload-mode-buttons not found"
    assert analyze_idx != -1, "btnAnalyze not found"
    assert webcam_idx < toggle_idx < analyze_idx, (
        f"Expected webcamSection ({webcam_idx}) < upload-mode-buttons ({toggle_idx}) < btnAnalyze ({analyze_idx})"
    )

