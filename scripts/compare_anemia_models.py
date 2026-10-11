"""
scripts/compare_anemia_models.py
Trains and evaluates AnemiaMobileNetRegressor and benchmarks against ResNet-50
and central tendency baselines (Mean, Median) on held-out subject partitions.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.preprocessing.dataset_anemia import build_anemia_dataloaders
from src.models.anemia_model import (
    AnemiaRegressor,
    AnemiaMobileNetRegressor,
    load_anemia_checkpoint,
    save_anemia_checkpoint
)
from src.evaluation.evaluate import compute_regression_metrics

SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'anemia')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['hb'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_subjects = []
    all_views = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['hb'].to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_subjects.extend(batch['subject_id'])
            all_views.extend(batch['view_type'])
            
    df_res = pd.DataFrame({
        'subject_id': all_subjects,
        'view_type': all_views,
        'actual': all_targets,
        'pred': all_preds
    })
    
    # View-level metrics
    view_mae = float(np.mean(np.abs(df_res['actual'] - df_res['pred'])))
    view_rmse = float(np.sqrt(np.mean((df_res['actual'] - df_res['pred'])**2)))
    
    # Patient-level metrics (mean aggregation across views)
    patient_df = df_res.groupby('subject_id').agg({'actual': 'first', 'pred': 'mean'}).reset_index()
    pat_actual = patient_df['actual'].values
    pat_pred = patient_df['pred'].values
    pat_mae = float(np.mean(np.abs(pat_actual - pat_pred)))
    pat_rmse = float(np.sqrt(np.mean((pat_actual - pat_pred)**2)))
    
    corr = float(np.corrcoef(pat_actual, pat_pred)[0, 1]) if len(pat_actual) > 1 else 0.0
    ss_res = np.sum((pat_actual - pat_pred)**2)
    ss_tot = np.sum((pat_actual - np.mean(pat_actual))**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Per-view breakdown
    view_breakdown = {}
    for v, group in df_res.groupby('view_type'):
        view_breakdown[v] = {
            'count': len(group),
            'mae': float(np.mean(np.abs(group['actual'] - group['pred']))),
            'rmse': float(np.sqrt(np.mean((group['actual'] - group['pred'])**2)))
        }
        
    return {
        'view_level': {'mae': view_mae, 'rmse': view_rmse},
        'patient_level': {'mae': pat_mae, 'rmse': pat_rmse, 'pearson_r': corr, 'r2': r2, 'count': len(patient_df)},
        'view_breakdown': view_breakdown
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparing Anemia Models on device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    loaders = build_anemia_dataloaders(
        manifest_csv=MANIFEST_PATH,
        workspace_root=WORKSPACE_ROOT,
        batch_size=16,
        use_processed_dir=PROCESSED_DIR
    )
    
    # 1. Baseline metrics from Train Mean and Median
    df_manifest = pd.read_csv(MANIFEST_PATH)
    train_hb = df_manifest[(df_manifest['split'] == 'train') & (df_manifest['hb_available'] == True)]['hb_value'].values
    train_mean = float(np.mean(train_hb))
    train_median = float(np.median(train_hb))
    
    test_rows = df_manifest[(df_manifest['split'] == 'test') & (df_manifest['hb_available'] == True)]
    test_actual_views = test_rows['hb_value'].values
    test_pat = test_rows.groupby('subject_id')['hb_value'].first().values
    
    baselines = {
        'train_mean_value': train_mean,
        'train_median_value': train_median,
        'mean_baseline': {
            'view_mae': float(np.mean(np.abs(test_actual_views - train_mean))),
            'view_rmse': float(np.sqrt(np.mean((test_actual_views - train_mean)**2))),
            'patient_mae': float(np.mean(np.abs(test_pat - train_mean))),
            'patient_rmse': float(np.sqrt(np.mean((test_pat - train_mean)**2)))
        },
        'median_baseline': {
            'view_mae': float(np.mean(np.abs(test_actual_views - train_median))),
            'view_rmse': float(np.sqrt(np.mean((test_actual_views - train_median)**2))),
            'patient_mae': float(np.mean(np.abs(test_pat - train_median))),
            'patient_rmse': float(np.sqrt(np.mean((test_pat - train_median)**2)))
        }
    }
    
    # 2. Existing ResNet-50 Checkpoint
    resnet_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    resnet_model = load_anemia_checkpoint(resnet_path, device=str(device))
    val_resnet = evaluate_model(resnet_model, loaders['val'], device)
    test_resnet = evaluate_model(resnet_model, loaders['test'], device)
    print("ResNet-50 Test Patient MAE:", test_resnet['patient_level']['mae'])
    
    # 3. Train MobileNetV3-Small Regressor candidate
    print("Training Candidate: MobileNetV3-Small Regressor (5 epochs)...")
    mobilenet = AnemiaMobileNetRegressor(pretrained=True, dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(mobilenet.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    
    best_mobilenet_val_mae = float('inf')
    best_mobilenet_metrics = {}
    mobilenet_ckpt_path = os.path.join(SAVED_MODELS_DIR, 'mobilenet_model.pth')
    
    for epoch in range(1, 6):
        loss = train_epoch(mobilenet, loaders['train'], criterion, optimizer, device)
        val_eval = evaluate_model(mobilenet, loaders['val'], device)
        scheduler.step()
        val_mae = val_eval['patient_level']['mae']
        print(f"MobileNet Epoch {epoch} - Loss: {loss:.4f}, Val Pat MAE: {val_mae:.4f}")
        if val_mae < best_mobilenet_val_mae:
            best_mobilenet_val_mae = val_mae
            best_mobilenet_metrics = val_eval
            save_anemia_checkpoint(
                model=mobilenet,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'mae_g_dl': val_mae, 'rmse_g_dl': val_eval['patient_level']['rmse']},
                save_path=mobilenet_ckpt_path,
                config={'model': 'MobileNetV3-Small Regressor', 'epochs': 5}
            )
            
    # Evaluate best MobileNet on test set
    mobilenet_best = AnemiaMobileNetRegressor(pretrained=False)
    mobilenet_best.load_state_dict(torch.load(mobilenet_ckpt_path, map_location='cpu')['model_state_dict'])
    mobilenet_best.to(device).eval()
    test_mobilenet = evaluate_model(mobilenet_best, loaders['test'], device)
    print("MobileNetV3 Test Patient MAE:", test_mobilenet['patient_level']['mae'])
    
    # 4. Compare and select final model based ON VALIDATION PERFORMANCE ONLY
    # Rule: "Choose the final model using validation data only. Never select it based on the best test-set score."
    print("\n=== MODEL SELECTION (BASED ON VALIDATION DATA ONLY) ===")
    print(f"ResNet-50 Val Patient MAE:    {val_resnet['patient_level']['mae']:.4f} g/dL")
    print(f"MobileNetV3 Val Patient MAE:  {best_mobilenet_metrics['patient_level']['mae']:.4f} g/dL")
    
    if val_resnet['patient_level']['mae'] <= best_mobilenet_metrics['patient_level']['mae']:
        chosen_model = "ResNet-50 Regressor"
        chosen_rationale = f"ResNet-50 achieved lower validation MAE ({val_resnet['patient_level']['mae']:.4f} vs {best_mobilenet_metrics['patient_level']['mae']:.4f} g/dL)."
    else:
        chosen_model = "MobileNetV3-Small Regressor"
        chosen_rationale = f"MobileNetV3-Small achieved lower validation MAE ({best_mobilenet_metrics['patient_level']['mae']:.4f} vs {val_resnet['patient_level']['mae']:.4f} g/dL)."
        
    comparison = {
        'baselines': baselines,
        'resnet50': {
            'validation': val_resnet,
            'test': test_resnet
        },
        'mobilenet_v3': {
            'validation': best_mobilenet_metrics,
            'test': test_mobilenet
        },
        'model_selection': {
            'selected_model': chosen_model,
            'selection_basis': 'Validation Patient MAE',
            'rationale': chosen_rationale
        }
    }
    
    out_path = os.path.join(REPORTS_DIR, 'anemia_model_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison report saved to {out_path}")
    print(json.dumps(comparison['model_selection'], indent=2))

if __name__ == '__main__':
    main()
