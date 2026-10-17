"""
tests/test_complete_evaluation_pipeline.py

Verification of the complete multi-target evaluation pipeline:
1. Every implemented, eligible analysis runs automatically whenever an image is uploaded:
   - Anemia (ResNet-50 Hb continuous regression)
   - SLID (MobileNetV3-Small conjunctival injection)
   - Jaundice (MobileNetV3-Small scleral icterus)
   - Anatomical Segmentation (ClinicalOcularSegmenter-v1.2 with adaptive chromatic clustering)
   - Secondary Vision (gpt-5.6-terra / deterministic fallback)
   - Reference Case Retrieval (MedSigLIP 2048-dim embeddings)
   - Calibrated Multi-Target Bayesian Fusion (concordance score & risk tier)
2. All results persist to SQLite under one authoritative session.
3. Both English and Hindi PDFs generate strictly 2 pages with genuine metrics.
"""

import os
import io
import json
import pytest
import pandas as pd
import pypdf
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.database.session import SessionLocal, get_session_by_id
from src.segmentation.clinical_segmenter import get_ocular_segmenter
from PIL import Image

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_eye_image_path():
    anemia_df = pd.read_csv(os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv'))
    test_rows = anemia_df[anemia_df['split'] == 'test']
    path = os.path.join(WORKSPACE_ROOT, test_rows.iloc[0]['relative_path'])
    assert os.path.exists(path), f"Test image does not exist: {path}"
    return path


def test_complete_evaluation_pipeline_all_modules_run(client, sample_eye_image_path):
    """
    Validates that POST /api/analyze automatically triggers ALL eligible modules
    and returns comprehensive, non-zero results.
    """
    with open(sample_eye_image_path, 'rb') as f:
        img_bytes = f.read()

    files = [('images', ('patient_eye.jpg', img_bytes, 'image/jpeg'))]
    data = {'module': 'unified', 'capture_source': 'upload'}

    resp = client.post('/api/analyze', files=files, data=data)
    assert resp.status_code == 200, f"Analyze failed: {resp.text}"
    res = resp.json()

    # 1. Authoritative session metadata
    sid = res.get('session_id')
    assert sid is not None
    assert res.get('operational_status') == 'available_research_only'

    # 2. Model A: Anemia (Continuous Hb)
    assert 'anemia' in res and res['anemia'] is not None
    hb = res.get('view_mean')
    assert isinstance(hb, (int, float))
    assert 4.0 <= hb <= 20.0

    # 3. Model B: SLID Conjunctival Injection
    assert 'slid' in res and res['slid'] is not None
    s_raw = res['slid'].get('prediction') or res['slid'].get('raw_prediction')
    assert isinstance(s_raw, list) and len(s_raw) > 0
    assert s_raw[0].get('predicted_class') in ('control_normal', 'conjunctival_injection')
    assert 0.0 <= s_raw[0].get('confidence') <= 1.0

    # 4. Model C: Jaundice / Scleral Icterus
    assert 'jaundice' in res and res['jaundice'] is not None
    j_raw = res['jaundice'].get('prediction') or res['jaundice'].get('raw_prediction')
    assert isinstance(j_raw, list) and len(j_raw) > 0
    assert j_raw[0].get('predicted_class') in ('normal', 'visibly_jaundiced')
    assert 0.0 <= j_raw[0].get('confidence') <= 1.0

    # 5. Ocular Anatomical Segmentation
    assert 'segmentation' in res and res['segmentation'] is not None
    seg = res['segmentation']
    cov = seg.get('coverage_pct', {})
    assert 'conjunctiva' in cov and 'sclera' in cov
    assert cov['conjunctiva'] > 0.0 or cov['sclera'] > 0.0

    # 6. Secondary Vision
    assert 'secondary_vision' in res and res['secondary_vision'] is not None
    sec = res['secondary_vision']
    assert sec.get('status') in ('success', 'available', 'offline_fallback')
    assert len(sec.get('summary', '')) > 10

    # 7. Reference Retrieval (MedSigLIP)
    assert 'reference_retrieval' in res and res['reference_retrieval'] is not None
    ret = res['reference_retrieval']
    assert len(ret.get('top_matches', [])) > 0

    # 8. Calibrated Bayesian Ensemble
    assert 'calibrated_ensemble' in res and res['calibrated_ensemble'] is not None
    ens = res['calibrated_ensemble']
    assert ens.get('status') == 'success'
    assert 'tier_label' in ens
    assert ens.get('concordance_score') is not None

    # 9. Module Execution Audit
    audit = res.get('module_execution_audit')
    assert audit is not None
    assert 'anemia' in audit
    assert 'slid_injection' in audit
    assert 'scleral_icterus' in audit
    assert 'mendeley_conjunctivitis' in audit
    assert audit['mendeley_conjunctivitis']['status'] == 'blocked_validation'

    # 10. Database Persistence
    with SessionLocal() as db:
        db_sess = get_session_by_id(db, sid)
        assert db_sess is not None
        saved_mods = {r.module: r for r in db_sess.model_results}
        assert 'anemia' in saved_mods
        assert 'slid' in saved_mods
        assert 'jaundice' in saved_mods

    # 11. PDF Generation (English and Hindi, strictly 2 pages)
    for lang in ('en', 'hi'):
        pdf_resp = client.get(f'/api/reports/{sid}/pdf?lang={lang}')
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers['content-type'] == 'application/pdf'
        reader = pypdf.PdfReader(io.BytesIO(pdf_resp.content))
        assert len(reader.pages) == 2, f"PDF for {lang} has {len(reader.pages)} pages instead of strictly 2."


def test_segmentation_non_zero_tissue_exposure(sample_eye_image_path):
    """Verifies that the segmentation engine detects genuine tissue regions."""
    segmenter = get_ocular_segmenter()
    with Image.open(sample_eye_image_path) as img:
        res = segmenter.segment(img)

    cov = res.coverage_pct
    assert (cov['conjunctiva'] > 0.0 or cov['sclera'] > 0.0), f"Expected tissue coverage, got {cov}"
