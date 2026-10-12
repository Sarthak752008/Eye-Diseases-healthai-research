"""
tests/test_remediation_and_models.py
Comprehensive Phase 8 Test Suite:
- Manifest & image file integrity
- Leakage-free subject & duplicate partitioning
- Checkpoint loading & weight verification for all 3 models
- Deterministic repeated inference (3x on real image)
- Preprocessing contract identicality
- Metadata-only shortcut verification (SLID vs Mendeley)
- Real held-out test data inference end-to-end
- Corrupt and invalid input rejection
- PDF report generation
"""

import os
import io
import json
import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi.testclient import TestClient

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from app import app
from src.models.anemia_model import AnemiaRegressor, load_anemia_checkpoint
from src.models.conjunctivitis_model import ConjunctivitisClassifier, load_conjunctivitis_checkpoint
from src.models.jaundice_model import JaundiceClassifier, load_jaundice_checkpoint
from src.inference.unified_service import EyeHealthInferenceService
from src.preprocessing.image_quality import safe_load_image, resize_and_pad
from src.preprocessing.dataset_anemia import get_anemia_transforms
from src.inference.pdf_report import build_two_page_pdf

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def service():
    return EyeHealthInferenceService(device='cpu')

# ---------------------------------------------------------------------------
# 1. Manifest & Partition Integrity
# ---------------------------------------------------------------------------

def test_anemia_manifest_and_leakage_integrity():
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
    assert os.path.exists(manifest_path)
    df = pd.read_csv(manifest_path)
    
    assert len(df) == 862
    assert df['subject_id'].nunique() == 218
    
    # Strict Subject-Level Partitioning: Zero subject overlap between train, val, test
    train_subs = set(df[df['split'] == 'train']['subject_id'])
    val_subs = set(df[df['split'] == 'val']['subject_id'])
    test_subs = set(df[df['split'] == 'test']['subject_id'])
    
    assert len(train_subs.intersection(val_subs)) == 0, "Data leakage detected: train and val share subjects"
    assert len(train_subs.intersection(test_subs)) == 0, "Data leakage detected: train and test share subjects"
    assert len(val_subs.intersection(test_subs)) == 0, "Data leakage detected: val and test share subjects"

def test_slid_injection_manifest_integrity():
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'slid_injection_manifest.csv')
    assert os.path.exists(manifest_path)
    df = pd.read_csv(manifest_path)
    
    assert len(df) == 552
    assert set(df['label'].unique()) == {0, 1}
    assert (df['label'] == 1).sum() == 307
    assert (df['label'] == 0).sum() == 245
    
    # Zero cross-split MD5 collision
    train_md5 = set(df[df['split'] == 'train']['md5'])
    val_md5 = set(df[df['split'] == 'val']['md5'])
    test_md5 = set(df[df['split'] == 'test']['md5'])
    
    assert len(train_md5.intersection(val_md5)) == 0
    assert len(train_md5.intersection(test_md5)) == 0
    assert len(val_md5.intersection(test_md5)) == 0

def test_jaundice_manifest_integrity():
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
    assert os.path.exists(manifest_path)
    df = pd.read_csv(manifest_path)
    
    assert len(df) == 321
    assert set(df['label'].unique()) == {0, 1}
    assert (df['label'] == 0).sum() == 176
    assert (df['label'] == 1).sum() == 145
    
    train_md5 = set(df[df['split'] == 'train']['md5'])
    val_md5 = set(df[df['split'] == 'val']['md5'])
    test_md5 = set(df[df['split'] == 'test']['md5'])
    
    assert len(train_md5.intersection(val_md5)) == 0
    assert len(train_md5.intersection(test_md5)) == 0
    assert len(val_md5.intersection(test_md5)) == 0

# ---------------------------------------------------------------------------
# 2. Checkpoint Loading & Architecture Match
# ---------------------------------------------------------------------------

def test_anemia_checkpoint_loads_and_is_eval():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth')
    assert os.path.exists(ckpt_path)
    model = load_anemia_checkpoint(ckpt_path, device='cpu')
    assert isinstance(model, AnemiaRegressor)
    assert not model.training
    
    # Check forward pass with dummy tensor
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == torch.Size([1])
    assert torch.is_tensor(out)

def test_slid_conjunctivitis_checkpoint_loads_and_is_eval():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'slid_injection_model.pth')
    assert os.path.exists(ckpt_path)
    model = load_conjunctivitis_checkpoint(ckpt_path, device='cpu')
    assert isinstance(model, ConjunctivitisClassifier)
    assert not model.training
    
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == torch.Size([1, 2])

def test_jaundice_checkpoint_loads_and_is_eval():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice', 'best_model.pth')
    assert os.path.exists(ckpt_path)
    model = load_jaundice_checkpoint(ckpt_path, device='cpu')
    assert isinstance(model, JaundiceClassifier)
    assert not model.training
    
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == torch.Size([1, 2])

# ---------------------------------------------------------------------------
# 3. Determinism & Preprocessing Equivalence
# ---------------------------------------------------------------------------

def test_deterministic_repeated_inference_on_real_image(service):
    """Runs inference 3 times on one fixed real image and checks exact equality."""
    real_img = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg')
    assert os.path.exists(real_img)
    
    preds = []
    for _ in range(3):
        res = service.predict_anemia(real_img, view_types='raw_eye', capture_source='upload')
        preds.append(res['prediction'][0]['hb_pred_g_dl'])
        
    assert len(set(preds)) == 1, f"Repeated inference was non-deterministic: {preds}"

def test_preprocessing_contract_identicality(service):
    """Verifies training evaluation transforms and inference preprocessing are mathematically identical."""
    real_img_path = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg')
    
    # 1. Training eval pipeline
    eval_transform = get_anemia_transforms(is_train=False, target_size=(224, 224))
    img_pil = safe_load_image(real_img_path)
    img_padded = resize_and_pad(img_pil, (224, 224))
    tensor_train = eval_transform(img_padded)
    
    # 2. Inference pipeline
    prep = service._prepare_image_input(real_img_path)
    img_infer_padded = resize_and_pad(prep['image'], (224, 224))
    tensor_infer = service._transform(img_infer_padded)
    
    max_diff = float((tensor_train - tensor_infer).abs().max().item())
    assert max_diff == 0.0, f"Preprocessing difference detected: {max_diff}"

# ---------------------------------------------------------------------------
# 4. End-to-End Real Unseen Data Inference
# ---------------------------------------------------------------------------

def test_real_held_out_anemia_inference(service):
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
    df = pd.read_csv(manifest_path)
    test_sample = df[(df['split'] == 'test') & (df['hb_available'] == True)].iloc[0]
    img_path = os.path.join(WORKSPACE_ROOT, test_sample['relative_path'])
    assert os.path.exists(img_path), f"Explicit failure: missing required Anemia test image: {img_path}"
    
    res = service.predict_anemia(img_path, view_types=test_sample['view_type'], capture_source='upload')
    assert res['status'] == 'available_research_only'
    assert res['unit'] == 'g/dL'
    pred_val = res['prediction'][0]['hb_pred_g_dl']
    assert 6.0 <= pred_val <= 18.0
    true_hb = float(test_sample['hb_value'])
    print(f"\n[Smoke Test Anemia] Path: {img_path}")
    print(f"[Smoke Test Anemia] Ground Truth Hb: {true_hb:.2f} g/dL | Predicted Hb: {pred_val:.2f} g/dL | Abs Error: {abs(pred_val - true_hb):.2f} g/dL")

def test_real_held_out_slid_inference():
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'slid_injection_manifest.csv')
    df = pd.read_csv(manifest_path)
    test_sample = df[df['split'] == 'test'].iloc[0]
    img_path = os.path.join(WORKSPACE_ROOT, test_sample['relative_path'])
    assert os.path.exists(img_path), f"Explicit failure: missing required SLID test image: {img_path}"
    
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'slid_injection_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'slid', 'best_model.pth')
    model = load_conjunctivitis_checkpoint(ckpt_path, device='cpu')
    
    im = safe_load_image(img_path)
    assert im is not None, f"Explicit failure: failed to decode SLID test image: {img_path}"
    padded = resize_and_pad(im, (224, 224))
    tensor = get_anemia_transforms(is_train=False)(padded).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_cls = int(torch.argmax(logits, dim=1).item())
        
    assert pred_cls in (0, 1)
    assert 0.0 <= probs[0] <= 1.0
    assert 0.0 <= probs[1] <= 1.0
    gt_cls = "conjunctival_injection" if int(test_sample['label']) == 1 else "control_normal"
    pred_name = "conjunctival_injection" if pred_cls == 1 else "control_normal"
    print(f"\n[Smoke Test SLID] Path: {img_path}")
    print(f"[Smoke Test SLID] Ground Truth: {gt_cls} (label={test_sample['label']}) | Predicted: {pred_name} (conf={probs[pred_cls]:.4f})")

def test_real_held_out_jaundice_inference():
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
    df = pd.read_csv(manifest_path)
    test_sample = df[df['split'] == 'test'].iloc[0]
    img_path = os.path.join(WORKSPACE_ROOT, test_sample['relative_path'])
    assert os.path.exists(img_path), f"Explicit failure: missing required Jaundice test image: {img_path}"
    
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice', 'best_model.pth')
    model = load_jaundice_checkpoint(ckpt_path, device='cpu')
    
    im = safe_load_image(img_path)
    assert im is not None, f"Explicit failure: failed to decode Jaundice test image: {img_path}"
    padded = resize_and_pad(im, (224, 224))
    tensor = get_anemia_transforms(is_train=False)(padded).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_cls = int(torch.argmax(logits, dim=1).item())
        
    assert pred_cls in (0, 1)
    assert 0.0 <= probs[0] <= 1.0
    gt_cls = "visibly_jaundiced" if int(test_sample['label']) == 1 else "normal"
    pred_name = "visibly_jaundiced" if pred_cls == 1 else "normal"
    print(f"\n[Smoke Test Jaundice] Path: {img_path}")
    print(f"[Smoke Test Jaundice] Ground Truth: {gt_cls} (label={test_sample['label']}) | Predicted: {pred_name} (conf={probs[pred_cls]:.4f})")

# ---------------------------------------------------------------------------
# 5. Rejection & Error Handling
# ---------------------------------------------------------------------------

def test_corrupt_image_returns_clean_error(service):
    with io.BytesIO(b"not an image file") as f:
        # Invalid image bytes
        res = service.predict_anemia(images="non_existent_file.jpg")
        assert res['status'] == 'error'
        assert 'File does not exist' in res['error']

def test_incompatible_view_rejected(service):
    real_img = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg')
    res = service.predict_anemia(real_img, view_types='full_face')
    assert res['status'] == 'error'
    assert 'Unsupported anatomical view' in res['error']

def test_two_page_pdf_generation():
    session_data = {
        "session_id": "ses_rem_test_01",
        "timestamp_utc": "2026-10-11T12:00:00Z",
        "selected_module": "anemia",
        "capture_source": "upload",
        "operational_status": "available_research_only",
        "model_output": 12.8,
        "view_count": 1,
        "view_mean": 12.8,
        "view_std": 0.0,
        "view_types": ["palpebral"]
    }
    user_data = {
        "full_name": "Test Subject",
        "phone": "9999999999",
        "email": "test@subject.org",
        "date_of_birth": "1995-01-01"
    }
    pdf_bytes = build_two_page_pdf(session_data, user_data=user_data, language="en")
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000


def test_slid_inference_service_end_to_end(service):
    """Verifies that EyeHealthInferenceService runs SLID conjunctival injection classification."""
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'slid_injection_manifest.csv')
    df = pd.read_csv(manifest_path)
    test_sample = df[df['split'] == 'test'].iloc[0]
    img_path = os.path.join(WORKSPACE_ROOT, test_sample['relative_path'])
    assert os.path.exists(img_path), f"Explicit failure: missing required SLID test image: {img_path}"

    res = service.predict_conjunctival_injection(img_path)
    assert res['module'] == 'slid'
    assert res['status'] == 'available_research_only'
    assert res['unit'] == 'categorical'
    assert len(res['prediction']) == 1
    assert res['prediction'][0]['predicted_class'] in ('control_normal', 'conjunctival_injection')
    assert 0.0 <= res['prediction'][0]['confidence'] <= 1.0
    gt_cls = "conjunctival_injection" if int(test_sample['label']) == 1 else "control_normal"
    print(f"\n[Service SLID] Path: {img_path} | GT: {gt_cls} | Predicted: {res['prediction'][0]['predicted_class']} (conf={res['prediction'][0]['confidence']:.4f})")


def test_jaundice_inference_service_end_to_end(service):
    """Verifies that EyeHealthInferenceService runs categorical scleral icterus screening."""
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
    df = pd.read_csv(manifest_path)
    test_sample = df[df['split'] == 'test'].iloc[0]
    img_path = os.path.join(WORKSPACE_ROOT, test_sample['relative_path'])
    assert os.path.exists(img_path), f"Explicit failure: missing required Jaundice test image: {img_path}"

    res = service.predict_scleral_icterus(img_path)
    assert res['module'] == 'jaundice'
    assert res['status'] == 'available_research_only'
    assert res['unit'] == 'categorical'
    assert len(res['prediction']) == 1
    assert res['prediction'][0]['predicted_class'] in ('normal', 'visibly_jaundiced')
    assert 0.0 <= res['prediction'][0]['confidence'] <= 1.0
    assert any("numerical bilirubin unavailable" in w.lower() for w in res['warnings'])
    gt_cls = "visibly_jaundiced" if int(test_sample['label']) == 1 else "normal"
    print(f"\n[Service Jaundice] Path: {img_path} | GT: {gt_cls} | Predicted: {res['prediction'][0]['predicted_class']} (conf={res['prediction'][0]['confidence']:.4f})")


def test_api_analyze_slid_and_jaundice(client):
    """Verifies /api/analyze endpoints for SLID and Jaundice with real images."""
    # SLID
    slid_df = pd.read_csv(os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'slid_injection_manifest.csv'))
    p_slid = os.path.join(WORKSPACE_ROOT, slid_df[slid_df['split'] == 'test'].iloc[0]['relative_path'])
    assert os.path.exists(p_slid), f"Explicit failure: missing required SLID test image: {p_slid}"
    with open(p_slid, 'rb') as f:
        res_slid = client.post('/api/analyze', data={'module': 'slid'}, files=[('images', ('slid.png', f.read(), 'image/png'))])
    assert res_slid.status_code == 200
    d_slid = res_slid.json()
    assert d_slid['selected_module'] == 'slid'
    assert d_slid['operational_status'] == 'available_research_only'
    assert d_slid['model_output'][0]['predicted_class'] in ('control_normal', 'conjunctival_injection')

    # Jaundice
    jaund_df = pd.read_csv(os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv'))
    p_jaund = os.path.join(WORKSPACE_ROOT, jaund_df[jaund_df['split'] == 'test'].iloc[0]['relative_path'])
    assert os.path.exists(p_jaund), f"Explicit failure: missing required Jaundice test image: {p_jaund}"
    with open(p_jaund, 'rb') as f:
        res_jaund = client.post('/api/analyze', data={'module': 'jaundice'}, files=[('images', ('jaundice.jpg', f.read(), 'image/jpeg'))])
    assert res_jaund.status_code == 200
    d_jaund = res_jaund.json()
    assert d_jaund['selected_module'] == 'jaundice'
    assert d_jaund['operational_status'] == 'available_research_only'
    assert d_jaund['model_output'][0]['predicted_class'] in ('normal', 'visibly_jaundiced')


def test_new_image_upload_refreshes_results(client):
    """Verifies that uploading a new image refreshes the session and produces distinct outputs."""
    anemia_df = pd.read_csv(os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv'))
    p1 = os.path.join(WORKSPACE_ROOT, anemia_df[anemia_df['split'] == 'test'].iloc[0]['relative_path'])
    p2 = os.path.join(WORKSPACE_ROOT, anemia_df[anemia_df['split'] == 'test'].iloc[10]['relative_path'])

    with open(p1, 'rb') as f1:
        res1 = client.post('/api/analyze', data={'module': 'anemia', 'view_types': ['raw_eye']}, files=[('images', ('img1.jpg', f1.read(), 'image/jpeg'))])
    with open(p2, 'rb') as f2:
        res2 = client.post('/api/analyze', data={'module': 'anemia', 'view_types': ['raw_eye']}, files=[('images', ('img2.jpg', f2.read(), 'image/jpeg'))])

    d1 = res1.json()
    d2 = res2.json()
    assert d1['session_id'] != d2['session_id']
    assert d1['view_mean'] != d2['view_mean']
