"""
test_layers.py - Unit and regression tests for Layers 1-3.
"""

import os
import torch
import pandas as pd
import pytest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from src.models.anemia_model import load_anemia_checkpoint
from src.models.conjunctivitis_model import load_conjunctivitis_checkpoint
from src.models.jaundice_model import JaundiceClassifier


def test_manifests_exist():
    anemia_m = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
    conj_m = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'conjunctivitis_manifest.csv')
    jaund_m = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
    
    assert os.path.exists(anemia_m), "Anemia manifest missing"
    assert os.path.exists(conj_m), "Conjunctivitis manifest missing"
    assert os.path.exists(jaund_m), "Jaundice manifest missing"
    
    anemia_df = pd.read_csv(anemia_m)
    conj_df = pd.read_csv(conj_m)
    
    assert len(anemia_df) == 862, f"Expected 862 records, got {len(anemia_df)}"
    assert len(conj_df) == 1006, f"Expected 1006 records, got {len(conj_df)}"


def test_anemia_checkpoint_loads():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth')
    assert os.path.exists(ckpt_path), "Anemia checkpoint missing"
    model = load_anemia_checkpoint(ckpt_path, device='cpu')
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == torch.Size([2]), f"Expected shape [2], got {out.shape}"


def test_conjunctivitis_checkpoint_loads():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'best_model.pth')
    assert os.path.exists(ckpt_path), "Conjunctivitis checkpoint missing"
    model = load_conjunctivitis_checkpoint(ckpt_path, device='cpu')
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {out.shape}"


def test_conjunctivitis_v2_clean_checkpoint_loads():
    ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'best_model_v2_clean.pth')
    assert os.path.exists(ckpt_path), "Conjunctivitis v2 clean checkpoint missing"
    model = load_conjunctivitis_checkpoint(ckpt_path, device='cpu')
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {out.shape}"


def test_jaundice_architecture_forward():
    model = JaundiceClassifier(pretrained=False, num_classes=2)
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {out.shape}"


def test_metadata_baseline_report_exists_and_valid():
    report_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'metadata_baseline_metrics.json')
    assert os.path.exists(report_path), "Metadata baseline metrics report missing"
    
    import json
    with open(report_path, 'r') as f:
        data = json.load(f)
        
    assert 'dataset_audit' in data
    assert 'metadata_baselines_test_results' in data
    dt_metrics = data['metadata_baselines_test_results']['decision_tree_stump_metadata']
    assert dt_metrics['accuracy'] >= 0.95, f"Expected metadata accuracy >= 0.95, got {dt_metrics['accuracy']}"


def test_layer_3_5_reliability_documentation_exists():
    doc_path = os.path.join(WORKSPACE_ROOT, 'docs', 'RELIABILITY_AND_DOMAIN_VALIDATION.md')
    assert os.path.exists(doc_path), "Reliability documentation missing"
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "Data Collection Specification" in content
    assert "Metadata-Only Baseline" in content
    assert "Source-Independent Evaluation Plan" in content
    assert "Integration Readiness Gates" in content


