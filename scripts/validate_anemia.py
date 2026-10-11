"""
validate_anemia.py
------------------
Model A validation and baseline comparison:

1. Reuse the existing ResNet-50 checkpoint and verified patient-level splits.
2. Compare model performance against mean and median Hb baselines.
3. Check per-patient prediction consistency (multiple views -> single patient).
4. Report MAE, RMSE, R² at both image-view level and patient level.
5. Do NOT claim clinical suitability or create unsupported confidence intervals.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.preprocessing.dataset_anemia import build_anemia_dataloaders
from src.models.anemia_model import load_anemia_checkpoint
from src.evaluation.evaluate import compute_regression_metrics


def collect_predictions(model, loader, device='cpu'):
    """Run inference on a loader and collect per-image predictions."""
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['hb']
            preds = model(images)
            for i in range(len(targets)):
                records.append({
                    'image_id': batch['image_id'][i],
                    'subject_id': batch['subject_id'][i],
                    'view_type': batch['view_type'][i],
                    'source': batch['source'][i],
                    'hb_true': float(targets[i]),
                    'hb_pred': float(preds[i]),
                })
    return pd.DataFrame(records)


def baseline_metrics(y_true, name):
    """Compute mean and median baseline metrics."""
    mean_pred = np.full_like(y_true, y_true.mean())
    median_pred = np.full_like(y_true, np.median(y_true))
    mean_m = compute_regression_metrics(y_true, mean_pred)
    median_m = compute_regression_metrics(y_true, median_pred)
    print(f"  {name} Mean Baseline:   MAE={mean_m['mae_g_dl']:.4f}  RMSE={mean_m['rmse_g_dl']:.4f}  R²={mean_m['r2']:.4f}")
    print(f"  {name} Median Baseline: MAE={median_m['mae_g_dl']:.4f}  RMSE={median_m['rmse_g_dl']:.4f}  R²={median_m['r2']:.4f}")
    return mean_m, median_m


def main():
    device = 'cpu'
    manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
    processed_dir = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'anemia')
    checkpoint_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth')

    # ---- Load model ----
    print("Loading Anemia ResNet-50 checkpoint...")
    model = load_anemia_checkpoint(checkpoint_path, device=device)

    # ---- Load data ----
    loaders = build_anemia_dataloaders(
        manifest_csv=manifest_path,
        workspace_root=WORKSPACE_ROOT,
        batch_size=16,
        use_processed_dir=processed_dir,
    )

    # ---- Check target distribution ----
    manifest_df = pd.read_csv(manifest_path)
    valid_df = manifest_df[(manifest_df['hb_available'] == True) & (manifest_df['is_duplicate'] == False)]
    train_hb = valid_df[valid_df['split'] == 'train']['hb_value']
    val_hb = valid_df[valid_df['split'] == 'val']['hb_value']
    test_hb = valid_df[valid_df['split'] == 'test']['hb_value']

    print(f"\n=== Hb Target Distribution (images, non-duplicate) ===")
    for name, series in [('Train', train_hb), ('Val', val_hb), ('Test', test_hb)]:
        print(f"  {name}: N={len(series)}, mean={series.mean():.2f}, median={series.median():.2f}, "
              f"std={series.std():.2f}, min={series.min():.1f}, max={series.max():.1f}")

    # ---- Image-view level predictions ----
    print(f"\n=== Image-View Level Evaluation ===")
    results = {}
    for split_name, loader in [('val', loaders['val']), ('test', loaders['test'])]:
        pred_df = collect_predictions(model, loader, device)
        y_true = pred_df['hb_true'].values
        y_pred = pred_df['hb_pred'].values

        model_metrics = compute_regression_metrics(y_true, y_pred)
        print(f"\n  [{split_name.upper()}] Model (N={len(y_true)} views):")
        print(f"    MAE={model_metrics['mae_g_dl']:.4f}  RMSE={model_metrics['rmse_g_dl']:.4f}  "
              f"R²={model_metrics['r2']:.4f}  Pearson r={model_metrics['pearson_r']:.4f}")

        mean_baseline, median_baseline = baseline_metrics(y_true, split_name.upper())

        results[f'{split_name}_view_level'] = {
            'model': model_metrics,
            'mean_baseline': mean_baseline,
            'median_baseline': median_baseline,
        }

        # ---- Patient-level aggregation ----
        patient_df = pred_df.groupby('subject_id').agg(
            hb_true=('hb_true', 'first'),
            hb_pred_mean=('hb_pred', 'mean'),
            hb_pred_median=('hb_pred', 'median'),
            hb_pred_std=('hb_pred', 'std'),
            n_views=('hb_pred', 'count'),
        ).reset_index()

        y_true_p = patient_df['hb_true'].values
        y_pred_p = patient_df['hb_pred_mean'].values

        patient_model = compute_regression_metrics(y_true_p, y_pred_p)
        print(f"\n  [{split_name.upper()}] Patient-Level (N={len(y_true_p)} subjects, mean of views):")
        print(f"    MAE={patient_model['mae_g_dl']:.4f}  RMSE={patient_model['rmse_g_dl']:.4f}  "
              f"R²={patient_model['r2']:.4f}  Pearson r={patient_model['pearson_r']:.4f}")

        patient_mean_base, patient_median_base = baseline_metrics(y_true_p, f"{split_name.upper()} Patient")

        # Per-patient prediction consistency
        consistent_patients = patient_df[patient_df['n_views'] > 1]
        if len(consistent_patients) > 0:
            mean_pred_std = consistent_patients['hb_pred_std'].mean()
            max_pred_std = consistent_patients['hb_pred_std'].max()
            print(f"\n  [{split_name.upper()}] Per-Patient Prediction Consistency (multi-view patients):")
            print(f"    Mean std across views: {mean_pred_std:.4f} g/dL")
            print(f"    Max std across views:  {max_pred_std:.4f} g/dL")
        else:
            mean_pred_std = 0.0
            max_pred_std = 0.0

        results[f'{split_name}_patient_level'] = {
            'model': patient_model,
            'mean_baseline': patient_mean_base,
            'median_baseline': patient_median_base,
            'per_patient_pred_std_mean': round(mean_pred_std, 4),
            'per_patient_pred_std_max': round(max_pred_std, 4),
        }

    # ---- Save full validation report ----
    report = {
        'model_name': 'ResNet50 Anemia Regressor',
        'checkpoint': checkpoint_path,
        'target_distribution': {
            'train': {'N': len(train_hb), 'mean': round(float(train_hb.mean()), 2),
                      'std': round(float(train_hb.std()), 2)},
            'val': {'N': len(val_hb), 'mean': round(float(val_hb.mean()), 2),
                    'std': round(float(val_hb.std()), 2)},
            'test': {'N': len(test_hb), 'mean': round(float(test_hb.mean()), 2),
                     'std': round(float(test_hb.std()), 2)},
        },
        'results': results,
        'clinical_suitability_note': (
            'These metrics represent a research baseline only. '
            'Clinical suitability requires prospective validation on external cohorts, '
            'comparison with published non-invasive Hb estimation benchmarks, '
            'and assessment of clinically meaningful error thresholds (e.g. ±1 g/dL for anemia screening). '
            'No confidence intervals or clinical decision thresholds are claimed.'
        ),
    }
    report_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full validation report: {report_path}")


if __name__ == '__main__':
    main()
