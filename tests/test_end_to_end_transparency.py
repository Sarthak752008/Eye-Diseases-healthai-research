"""
End-to-end automated verification for OcuScan 4-stage visual pipeline transparency:
- 01 Original Upload
- 02 Detected Eye Crop
- 03 Final Model Input (224x224 RGB)
- 04 AI Detection Visualization (Grad-CAM feature attribution)
- Dual 2-Page PDF Reports (Screening & Technical Scorecard)
- History and Web Report REST endpoints
"""

import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import app
from pypdf import PdfReader

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_eye_image():
    # Use real dataset image if available or create a high-contrast eye-like image
    real_path = os.path.join(WORKSPACE_ROOT, "dataset anemia", "India", "1", "20200118_164733.jpg")
    if os.path.exists(real_path):
        with open(real_path, "rb") as f:
            return f.read()
    img = Image.new("RGB", (400, 300), color=(180, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_end_to_end_image_transparency_and_attribution(client, sample_eye_image):
    """Verifies that unified inference produces complete image_sequence and visualizations."""
    reg_res = client.post("/api/register", json={
        "full_name": "Test Transparency User",
        "email": "transparency@example.com",
        "phone": "+919876543210",
        "date_of_birth": "1995-05-15",
        "gender": "male",
        "research_consent": True,
        "data_storage_consent": True
    })
    assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
    user_id = reg_res.json()["user"]["id"]

    files = [("images", ("eye.jpg", sample_eye_image, "image/jpeg"))]
    data = {
        "module": "unified",
        "capture_source": "upload",
        "user_id": user_id
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    result = res.json()

    # 1. Verify image_sequence exists with all 4 stages
    assert "image_sequence" in result, "image_sequence missing from response"
    seq = result["image_sequence"]
    assert seq.get("stage_01_upload"), "stage_01_upload missing"
    assert seq.get("stage_02_crop"), "stage_02_crop missing"
    assert seq.get("stage_03_model_input"), "stage_03_model_input missing"
    assert seq.get("stage_04_visualization"), "stage_04_visualization missing"

    # All stages should be non-empty base64 data URIs
    for stage_key in ["stage_01_upload", "stage_02_crop", "stage_03_model_input", "stage_04_visualization"]:
        stage_obj = seq[stage_key]
        uri = stage_obj["image_b64"] if isinstance(stage_obj, dict) else stage_obj
        assert uri.startswith("data:image/jpeg;base64,"), f"{stage_key} is not a valid JPEG data URI"
        assert len(uri) > 500, f"{stage_key} appears too short"

    # 2. Verify visualizations dictionary with Grad-CAM layers
    assert "visualizations" in result, "visualizations missing from response"
    vis = result["visualizations"]
    assert "anemia" in vis, "anemia visualization missing"
    anemia_vis = vis["anemia"]
    assert anemia_vis.get("overlay"), "anemia overlay missing"
    assert anemia_vis.get("heatmap"), "anemia heatmap missing"
    assert anemia_vis.get("input"), "anemia input missing"
    assert "layer4" in anemia_vis.get("target_layer", ""), "anemia target_layer mismatch"

    if "slid" in vis:
        slid_vis = vis["slid"]
        assert slid_vis.get("overlay")
        assert "features" in slid_vis.get("target_layer", "")

    if "jaundice" in vis:
        jaund_vis = vis["jaundice"]
        assert jaund_vis.get("overlay")
        assert "features" in jaund_vis.get("target_layer", "")

    session_id = result.get("session_id")
    assert session_id, "session_id missing from response"

    # 3. Verify Patient Screening Report PDF (Page 1 has visual transparency panel, total 2 pages)
    pdf_res = client.get(f"/api/reports/{session_id}/screening-pdf?lang=en")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    pdf_reader = PdfReader(io.BytesIO(pdf_res.content))
    assert len(pdf_reader.pages) == 2, f"Screening PDF must be exactly 2 pages, got {len(pdf_reader.pages)}"
    page1_text = pdf_reader.pages[0].extract_text()
    assert "Visual Transparency" in page1_text or "Visual Inspection" in page1_text or "Grad-CAM" in page1_text

    # 4. Verify Technical Research Scorecard PDF (Page 1 has 4-stage progression panel, total 2 pages)
    scorecard_res = client.get(f"/api/reports/{session_id}/scorecard-pdf?lang=en")
    assert scorecard_res.status_code == 200
    assert scorecard_res.headers["content-type"] == "application/pdf"
    sc_reader = PdfReader(io.BytesIO(scorecard_res.content))
    assert len(sc_reader.pages) == 2, f"Scorecard PDF must be exactly 2 pages, got {len(sc_reader.pages)}"
    sc_page1_text = sc_reader.pages[0].extract_text()
    assert "4-Stage Input Progression" in sc_page1_text or "01 Original" in sc_page1_text

    # 5. Verify /api/reports/{session_id} returns image_sequence and visualizations for web report & history
    rpt_res = client.get(f"/api/reports/{session_id}")
    assert rpt_res.status_code == 200
    rpt_data = rpt_res.json()
    rep = rpt_data.get("report", rpt_data)
    assert "image_sequence" in rep or (rep.get("quality_summary") and "image_sequence" in rep.get("quality_summary"))
