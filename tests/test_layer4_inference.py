"""
tests/test_layer4_inference.py
------------------------------
Layer 4 test suite: Unified Inference Service and Reliability Gates.
"""

import os
import tempfile
import pytest
import numpy as np
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.unified_service import (
    EyeHealthInferenceService,
    InferenceResult,
    COMPATIBLE_ANEMIA_VIEWS,
    get_inference_service
)


@pytest.fixture
def service():
    return EyeHealthInferenceService(device='cpu')


@pytest.fixture
def sample_anemia_images():
    img_paths = [
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733.jpg'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_forniceal.png'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_forniceal_palpebral.png'),
        os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', '1', '20200118_164733_palpebral.png'),
    ]
    return img_paths


def test_response_schema_keys(service):
    """Verifies that all modules produce identical schema keys."""
    expected_keys = {
        'module', 'status', 'input_quality', 'prediction', 'unit',
        'view_count', 'view_mean', 'view_std', 'preprocessing',
        'warnings', 'limitations', 'disclaimer', 'error'
    }

    conj = service.predict_conjunctivitis()
    jaund = service.predict_jaundice()
    anemia_err = service.predict('anemia', images=[])

    assert set(conj.keys()) == expected_keys
    assert set(jaund.keys()) == expected_keys
    assert set(anemia_err.keys()) == expected_keys


def test_conjunctivitis_reliability_gate(service):
    """Ensures Conjunctivitis prediction is blocked per Layer 3.5 findings."""
    res = service.predict_conjunctivitis()
    assert res['module'] == 'conjunctivitis'
    assert res['status'] == 'blocked_validation'
    assert res['prediction'] is None
    assert res['unit'] is None
    assert any("shortcut" in w.lower() for w in res['warnings'])
    assert any("revalidation" in lim.lower() for lim in res['limitations'])


def test_jaundice_missing_data_gate(service):
    """Ensures Jaundice module reports unavailable due to missing dataset."""
    res = service.predict_jaundice()
    assert res['module'] == 'jaundice'
    assert res['status'] == 'unavailable_missing_data'
    assert res['prediction'] is None
    assert res['unit'] is None
    assert any("no jaundice" in w.lower() for w in res['warnings'])


def test_anemia_single_view_inference(service, sample_anemia_images):
    """Validates single-view anemia prediction."""
    img_path = sample_anemia_images[0]
    res = service.predict_anemia(img_path, view_types='raw_eye')

    assert res['module'] == 'anemia'
    assert res['status'] == 'available_research_only'
    assert res['unit'] == 'g/dL'
    assert res['view_count'] == 1
    assert res['view_std'] == 0.0
    assert len(res['prediction']) == 1
    assert res['prediction'][0]['view_type'] == 'raw_eye'
    assert 5.0 <= res['view_mean'] <= 20.0
    assert res['prediction'][0]['hb_pred_g_dl'] == res['view_mean']
    assert len(res['input_quality']) == 1
    assert 'sharpness' in res['input_quality'][0]
    assert len(res['warnings']) >= 3
    assert "RESEARCH USE ONLY" in res['disclaimer']


def test_anemia_multi_view_aggregation(service, sample_anemia_images):
    """Tests multi-view input handling, averaging, and standard deviation."""
    views = ['raw_eye', 'forniceal', 'forniceal_palpebral', 'palpebral']
    res = service.predict_anemia(sample_anemia_images, view_types=views)

    assert res['status'] == 'available_research_only'
    assert res['view_count'] == 4
    assert len(res['prediction']) == 4

    # Verify arithmetic mean and sample std calculation
    preds = [p['hb_pred_g_dl'] for p in res['prediction']]
    expected_mean = round(float(np.mean(preds)), 2)
    expected_std = round(float(np.std(preds, ddof=1)), 2)

    assert abs(res['view_mean'] - expected_mean) <= 0.02
    assert abs(res['view_std'] - expected_std) <= 0.02
    assert res['view_std'] > 0.0


def test_anemia_incompatible_view_rejection(service, sample_anemia_images):
    """Checks that arbitrary non-ocular views cannot be averaged."""
    res = service.predict_anemia(sample_anemia_images[0], view_types='cornea_arbitrary')
    assert res['status'] == 'error'
    assert 'Unsupported anatomical view' in res['error']


def test_invalid_image_handling(service):
    """Checks handling of missing file, non-image format, and corrupt images."""
    # 1. Non-existent file
    res1 = service.predict_anemia('non_existent_image_12345.jpg')
    assert res1['status'] == 'error'
    assert 'does not exist' in res1['error']

    # 2. Unsupported file extension
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not an image")
        txt_path = f.name
    try:
        res2 = service.predict_anemia(txt_path)
        assert res2['status'] == 'error'
        assert 'Unsupported image file format' in res2['error']
    finally:
        os.remove(txt_path)

    # 3. Corrupt image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b"corrupt junk data")
        bad_jpg = f.name
    try:
        res3 = service.predict_anemia(bad_jpg)
        assert res3['status'] == 'error'
        assert 'Corrupt or unreadable' in res3['error']
    finally:
        os.remove(bad_jpg)


def test_missing_checkpoint_handling():
    """Checks graceful error when model checkpoint file does not exist."""
    fake_service = EyeHealthInferenceService(
        anemia_checkpoint_path='saved_models/anemia/non_existent_model.pth'
    )
    # Create a small valid dummy PIL image
    img = Image.new('RGB', (100, 100), color=(150, 50, 50))
    res = fake_service.predict_anemia(img)
    assert res['status'] == 'error'
    assert 'checkpoint not found' in res['error'].lower()


def test_unified_predict_gateway(service, sample_anemia_images):
    """Tests common routing predict() method for all modules."""
    # Anemia route
    r_anemia = service.predict('anemia', images=sample_anemia_images[0], view_types='palpebral')
    assert r_anemia['module'] == 'anemia'
    assert r_anemia['status'] == 'available_research_only'

    # Conjunctivitis route
    r_conj = service.predict('conjunctivitis')
    assert r_conj['module'] == 'conjunctivitis'
    assert r_conj['status'] == 'blocked_validation'

    # Jaundice route
    r_jaund = service.predict('jaundice')
    assert r_jaund['module'] == 'jaundice'
    assert r_jaund['status'] == 'unavailable_missing_data'

    # Unknown route
    r_unknown = service.predict('unknown_disease')
    assert r_unknown['status'] == 'error'
    assert "Unknown module" in r_unknown['error']
