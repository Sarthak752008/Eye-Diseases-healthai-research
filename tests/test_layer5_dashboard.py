"""
tests/test_layer5_dashboard.py
------------------------------
Layer 5 test suite: Dashboard, API Endpoints, Quality Pre-Check, and Session Export.
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
def sample_image_bytes():
    """Generates an in-memory sample RGB ocular image."""
    img = Image.new('RGB', (224, 224), color=(180, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def real_anemia_image_paths():
    return [
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_forniceal.png'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_forniceal_palpebral.png'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_palpebral.png')
    ]


def test_dashboard_html_served(client):
    """Checks that the main dashboard HTML is served at root."""
    res = client.get("/")
    assert res.status_code == 200
    assert "EyeHealthAI" in res.text
    assert "RESEARCH USE ONLY" in res.text
    assert "moduleCardAnemia" in res.text
    assert "moduleCardConj" in res.text
    assert "moduleCardJaund" in res.text


def test_api_status_endpoint(client):
    """Verifies module status reporting from the REST API."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "EyeHealthAI"
    assert data["modules"]["anemia"]["status"] == "available_research_only"
    assert data["modules"]["conjunctivitis"]["status"] == "blocked_validation"
    assert data["modules"]["jaundice"]["status"] == "unavailable_missing_data"


def test_quality_check_valid_image(client, sample_image_bytes):
    """Tests the pre-inference quality checking endpoint with valid image."""
    files = {"image": ("test_eye.jpg", sample_image_bytes, "image/jpeg")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert "brightness" in data["quality"]
    assert "sharpness" in data["quality"]


def test_quality_check_corrupt_file(client):
    """Tests quality check rejection for empty/corrupt uploads."""
    files = {"image": ("corrupt.jpg", b"corrupt bytes", "image/jpeg")}
    res = client.post("/api/quality-check", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False
    assert "error" in data


def test_analyze_anemia_single_view(client, real_anemia_image_paths):
    """Tests single-view research inference via /api/analyze."""
    img_path = real_anemia_image_paths[0]
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    files = [("images", ("20200118_164733.jpg", img_bytes, "image/jpeg"))]
    data = {"module": "anemia", "view_types": ["raw_eye"]}

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "anemia"
    assert result["operational_status"] == "available_research_only"
    assert result["inference_permitted"] is True
    assert result["unit"] == "g/dL"
    assert result["view_count"] == 1
    assert result["view_mean"] > 0
    assert result["view_std"] == 0.0
    assert len(result["model_output"]) == 1
    assert "RESEARCH USE ONLY" in result["research_disclaimer"]
    assert len(result["reliability_warnings"]) >= 3


def test_analyze_anemia_multi_view(client, real_anemia_image_paths):
    """Tests multi-view aggregation and std computation via /api/analyze."""
    files = []
    for p in real_anemia_image_paths:
        with open(p, "rb") as f:
            files.append(("images", (os.path.basename(p), f.read(), "image/jpeg")))

    data = {
        "module": "anemia",
        "view_types": ["raw_eye", "forniceal", "forniceal_palpebral", "palpebral"]
    }

    res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    result = res.json()

    assert result["operational_status"] == "available_research_only"
    assert result["view_count"] == 4
    assert len(result["model_output"]) == 4
    assert result["view_std"] > 0.0
    # True Hb of IND_001 is 12.2 g/dL
    assert 10.0 <= result["view_mean"] <= 13.5


def test_conjunctivitis_blocked_gate_enforcement(client):
    """Ensures conjunctivitis request enforces blocked status and withholds prediction."""
    res = client.post("/api/analyze", data={"module": "conjunctivitis"})
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "conjunctivitis"
    assert result["operational_status"] == "blocked_validation"
    assert result["inference_permitted"] is False
    assert result["model_output"] is None
    assert any("shortcut" in w.lower() for w in result["reliability_warnings"])


def test_jaundice_missing_data_gate_enforcement(client):
    """Ensures jaundice request enforces unavailable status without fabricated values."""
    res = client.post("/api/analyze", data={"module": "jaundice"})
    assert res.status_code == 200
    result = res.json()

    assert result["selected_module"] == "jaundice"
    assert result["operational_status"] == "unavailable_missing_data"
    assert result["inference_permitted"] is False
    assert result["model_output"] is None
    assert any("no jaundice" in w.lower() for w in result["reliability_warnings"])


def test_session_export_endpoint(client):
    """Tests generation and download of session audit reports."""
    mock_session = create_session_export(
        module="anemia",
        inference_result={
            "status": "available_research_only",
            "prediction": [{"view_index": 0, "view_type": "palpebral", "hb_pred_g_dl": 12.3}],
            "unit": "g/dL",
            "view_count": 1,
            "view_mean": 12.3,
            "view_std": 0.0,
            "warnings": ["Warning 1"],
            "limitations": ["Limitation 1"],
            "disclaimer": "RESEARCH USE ONLY"
        },
        filenames=["sample.jpg"]
    )

    res = client.post("/api/export", json=mock_session)
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]
    assert "attachment" in res.headers["content-disposition"]
    exported = res.json()
    assert validate_session_export_schema(exported) is True
    assert exported["selected_module"] == "anemia"
    assert exported["view_mean"] == 12.3


def test_session_export_schema_rejection(client):
    """Tests rejection of incomplete/invalid session export payload."""
    res = client.post("/api/export", json={"incomplete": "payload"})
    assert res.status_code == 400
    assert "Invalid session report schema" in res.json()["detail"]
