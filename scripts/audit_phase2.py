"""
scripts/audit_phase2.py
Comprehensive Phase 2 empirical audit of checkpoints, preprocessing contracts,
inference determinism, target mapping, and baseline comparison.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.models.anemia_model import AnemiaRegressor, load_anemia_checkpoint
from src.models.conjunctivitis_model import ConjunctivitisClassifier, load_conjunctivitis_checkpoint
from src.inference.unified_service import EyeHealthInferenceService
from src.preprocessing.image_quality import safe_load_image, resize_and_pad
from src.preprocessing.dataset_anemia import get_anemia_transforms, AnemiaDataset

def audit_all():
    results = {}
    print("=== STARTING PHASE 2 AUDIT ===")

    # 1. Checkpoints inspection
    anemia_ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth')
    conj_ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'best_model.pth')
    conj_clean_ckpt_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'best_model_v2_clean.pth')

    ckpts = {
        'anemia': anemia_ckpt_path,
        'conjunctivitis_v1': conj_ckpt_path,
        'conjunctivitis_v2_clean': conj_clean_ckpt_path
    }

    results['checkpoints'] = {}
    for name, path in ckpts.items():
        exists = os.path.exists(path)
        if not exists:
            results['checkpoints'][name] = {'exists': False}
            continue
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        # Check non-zero weights, weight statistics
        weight_norms = []
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                weight_norms.append(float(v.norm().item()))
        results['checkpoints'][name] = {
            'exists': True,
            'epoch': ckpt.get('epoch', None),
            'metrics': ckpt.get('metrics', {}),
            'total_param_tensors': len(state_dict),
            'mean_weight_norm': float(np.mean(weight_norms)) if weight_norms else 0.0,
            'contains_trained_weights': bool(len(weight_norms) > 0 and np.mean(weight_norms) > 0.1)
        }

    # 2. Architecture Match
    anemia_model = load_anemia_checkpoint(anemia_ckpt_path, device='cpu')
    results['anemia_model_loaded_successfully'] = True
    results['anemia_model_is_eval'] = not anemia_model.training

    conj_model = load_conjunctivitis_checkpoint(conj_clean_ckpt_path, device='cpu')
    results['conj_model_loaded_successfully'] = True
    results['conj_model_is_eval'] = not conj_model.training

    # 3. Check Preprocessing Contracts (RGB/BGR, normalization, resize/padding)
    # In training:
    # safe_load_image (loads PIL RGB) -> resize_and_pad((224, 224), bilinear, black pad) -> transforms.ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # In unified_service:
    # safe_load_image (loads PIL RGB) -> resize_and_pad((224, 224), bilinear, black pad) -> transforms.ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Let's verify exact numerical equivalence on a sample image!
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
    df_anemia = pd.read_csv(manifest_path)
    sample_row = df_anemia[df_anemia['hb_available'] == True].iloc[0]
    sample_path = os.path.join(WORKSPACE_ROOT, sample_row['relative_path'])

    # Training preprocessing pipeline output:
    eval_transform = get_anemia_transforms(is_train=False, target_size=(224, 224))
    img_pil = safe_load_image(sample_path)
    img_resized = resize_and_pad(img_pil, (224, 224))
    tensor_train_pipeline = eval_transform(img_resized)

    # Unified service preprocessing output:
    service = EyeHealthInferenceService(device='cpu')
    prep = service._prepare_image_input(sample_path)
    raw_img = prep['image']
    resized_img = resize_and_pad(raw_img, (224, 224))
    tensor_infer_pipeline = service._transform(resized_img)

    max_diff = float((tensor_train_pipeline - tensor_infer_pipeline).abs().max().item())
    results['preprocessing_max_diff'] = max_diff
    results['preprocessing_strictly_identical'] = (max_diff == 0.0)

    # 4. Deterministic Repeated Inference Check
    # "For one fixed real dataset image, run inference three times using identical configuration and compare predictions."
    preds_3 = []
    for i in range(3):
        res = service.predict_anemia(images=sample_path, view_types=sample_row['view_type'], capture_source='upload')
        preds_3.append(res['prediction'][0]['hb_pred_g_dl'])
    results['repeated_inference_predictions'] = preds_3
    results['is_deterministic'] = len(set(preds_3)) == 1

    # 5. Target Mapping Integrity
    # Check that India.xlsx and Italy.xlsx match the manifest
    india_excel = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'India', 'India.xlsx')
    italy_excel = os.path.join(WORKSPACE_ROOT, 'dataset anemia', 'Italy', 'Italy.xlsx')
    df_india = pd.read_excel(india_excel)
    df_italy = pd.read_excel(italy_excel)

    # Let's check mapping
    results['india_excel_rows'] = len(df_india)
    results['italy_excel_rows'] = len(df_italy)
    
    # Manifest counts
    india_manifest = df_anemia[df_anemia['dataset_source'] == 'India']
    italy_manifest = df_anemia[df_anemia['dataset_source'] == 'Italy']
    results['india_manifest_records'] = len(india_manifest)
    results['italy_manifest_records'] = len(italy_manifest)
    results['india_manifest_subjects'] = int(india_manifest['subject_id'].nunique())
    results['italy_manifest_subjects'] = int(italy_manifest['subject_id'].nunique())

    # 6. Baseline vs Model Comparison on Test Set
    test_df = df_anemia[df_anemia['split'] == 'test'].copy()
    train_df = df_anemia[df_anemia['split'] == 'train'].copy()
    val_df = df_anemia[df_anemia['split'] == 'val'].copy()

    train_valid = train_df[train_df['hb_available'] == True]
    train_hb = train_valid['hb_value'].values
    train_mean = float(np.mean(train_hb))
    train_median = float(np.median(train_hb))

    test_valid = test_df[test_df['hb_available'] == True].copy()
    test_actual = test_valid['hb_value'].values

    mean_baseline_errors = np.abs(test_actual - train_mean)
    median_baseline_errors = np.abs(test_actual - train_median)

    mean_baseline_mae = float(np.mean(mean_baseline_errors))
    median_baseline_mae = float(np.mean(median_baseline_errors))
    mean_baseline_rmse = float(np.sqrt(np.mean((test_actual - train_mean)**2)))
    median_baseline_rmse = float(np.sqrt(np.mean((test_actual - train_median)**2)))

    # Evaluate ResNet-50 predictions on test set
    anemia_model.eval()
    test_preds = []
    with torch.no_grad():
        for _, r in test_valid.iterrows():
            p = os.path.join(WORKSPACE_ROOT, r['relative_path'])
            im = safe_load_image(p)
            im_r = resize_and_pad(im, (224, 224))
            t = service._transform(im_r).unsqueeze(0)
            pred = float(anemia_model(t).item())
            test_preds.append(pred)
    test_preds = np.array(test_preds)

    model_errors = np.abs(test_actual - test_preds)
    model_view_mae = float(np.mean(model_errors))
    model_view_rmse = float(np.sqrt(np.mean((test_actual - test_preds)**2)))

    # Patient-level aggregation on test set:
    test_valid['pred'] = test_preds
    patient_test = test_valid.groupby('subject_id').agg({'hb_value': 'first', 'pred': 'mean'}).reset_index()
    patient_actual = patient_test['hb_value'].values
    patient_pred = patient_test['pred'].values
    patient_mae = float(np.mean(np.abs(patient_actual - patient_pred)))
    patient_rmse = float(np.sqrt(np.mean((patient_actual - patient_pred)**2)))

    patient_mean_baseline_mae = float(np.mean(np.abs(patient_actual - train_mean)))
    patient_median_baseline_mae = float(np.mean(np.abs(patient_actual - train_median)))
    patient_mean_baseline_rmse = float(np.sqrt(np.mean((patient_actual - train_mean)**2)))
    patient_median_baseline_rmse = float(np.sqrt(np.mean((patient_actual - train_median)**2)))

    # Correlation and R^2
    corr = float(np.corrcoef(patient_actual, patient_pred)[0, 1])
    ss_res = np.sum((patient_actual - patient_pred)**2)
    ss_tot = np.sum((patient_actual - train_mean)**2)
    r2 = float(1.0 - (ss_res / ss_tot))

    results['baselines'] = {
        'train_mean_hb': train_mean,
        'train_median_hb': train_median,
        'view_level': {
            'mean_baseline_mae': mean_baseline_mae,
            'median_baseline_mae': median_baseline_mae,
            'mean_baseline_rmse': mean_baseline_rmse,
            'median_baseline_rmse': median_baseline_rmse,
            'model_mae': model_view_mae,
            'model_rmse': model_view_rmse,
        },
        'patient_level': {
            'mean_baseline_mae': patient_mean_baseline_mae,
            'median_baseline_mae': patient_median_baseline_mae,
            'mean_baseline_rmse': patient_mean_baseline_rmse,
            'median_baseline_rmse': patient_median_baseline_rmse,
            'model_mae': patient_mae,
            'model_rmse': patient_rmse,
            'pearson_correlation': corr,
            'r2_score': r2,
            'test_patient_count': len(patient_test)
        }
    }

    print("Audit completed successfully.")
    with open(os.path.join(WORKSPACE_ROOT, 'reports', 'audit_phase2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    audit_all()
