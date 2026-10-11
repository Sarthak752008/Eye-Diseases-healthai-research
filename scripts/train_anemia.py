"""
train_anemia.py - Training pipeline for ResNet50 Anemia (Hb) Regressor.
Predicts numerical laboratory Hemoglobin (g/dL) with strict subject-level isolation.
Reports MAE and RMSE in g/dL on validation and test sets.
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
    save_anemia_checkpoint,
    load_anemia_checkpoint
)
from src.evaluation.evaluate import compute_regression_metrics

SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'anemia')

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['hb'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
    return avg_loss, mae


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['hb'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(np.array(all_targets), np.array(all_preds))
    metrics['loss'] = round(avg_loss, 4)
    return avg_loss, metrics


def main(epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Anemia Model Training on device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data with strict subject-level splitting
    loaders = build_anemia_dataloaders(
        manifest_csv=MANIFEST_PATH,
        workspace_root=WORKSPACE_ROOT,
        batch_size=batch_size,
        use_processed_dir=PROCESSED_DIR
    )
    print(f"Dataset split sizes: Train={len(loaders['datasets']['train'])}, "
          f"Val={len(loaders['datasets']['val'])}, Test={len(loaders['datasets']['test'])}")
    
    # 2. Build Model
    model = AnemiaRegressor(pretrained=True, dropout=0.2).to(device)
    # Smooth L1 / Huber Loss is robust to slight outliers
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_mae = float('inf')
    best_val_metrics = {}
    training_log = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_metrics = evaluate_loader(model, loaders['val'], criterion, device)
        scheduler.step()
        epoch_dur = time.time() - t0
        
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_mae_g_dl': round(train_mae, 4),
            'val_loss': round(val_loss, 4),
            'val_mae_g_dl': val_metrics['mae_g_dl'],
            'val_rmse_g_dl': val_metrics['rmse_g_dl'],
            'val_r2': val_metrics['r2'],
            'val_pearson_r': val_metrics['pearson_r'],
            'epoch_time_sec': round(epoch_dur, 2)
        }
        training_log.append(log_entry)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f} g/dL | "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_metrics['mae_g_dl']:.2f} g/dL, "
              f"RMSE: {val_metrics['rmse_g_dl']:.2f} g/dL ({epoch_dur:.1f}s)")
        
        # Select best checkpoint on validation MAE
        if val_metrics['mae_g_dl'] < best_val_mae:
            best_val_mae = val_metrics['mae_g_dl']
            best_val_metrics = val_metrics
            checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
            save_anemia_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                save_path=checkpoint_path,
                config={'model': 'ResNet50 Regressor', 'target': 'Hemoglobin (g/dL)', 'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
            )
            
    total_time = time.time() - start_time
    print(f"Anemia training completed in {total_time:.1f}s.")
    
    # 3. Evaluate Best Model on Held-Out Test Set
    best_model = load_anemia_checkpoint(os.path.join(SAVED_MODELS_DIR, 'best_model.pth'), device=str(device))
    test_loss, test_metrics = evaluate_loader(best_model, loaders['test'], criterion, device)
    print(f"=== Final Held-Out Test Metrics ===")
    print(f"Test MAE:   {test_metrics['mae_g_dl']:.4f} g/dL")
    print(f"Test RMSE:  {test_metrics['rmse_g_dl']:.4f} g/dL")
    print(f"Test R²:    {test_metrics['r2']:.4f}")
    print(f"Test Pearson r: {test_metrics['pearson_r']:.4f}")
    
    # 4. Save artifacts
    log_df = pd.DataFrame(training_log)
    log_csv = os.path.join(SAVED_MODELS_DIR, 'training_log.csv')
    log_df.to_csv(log_csv, index=False)
    
    metrics_summary = {
        'model_name': 'ResNet50 Anemia Regressor',
        'target': 'Hemoglobin (g/dL)',
        'training_time_seconds': round(total_time, 2),
        'best_val_metrics': best_val_metrics,
        'final_test_metrics': test_metrics
    }
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
        
    curve_path = os.path.join(REPORTS_DIR, 'anemia_training_curve.json')
    with open(curve_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
        
    print(f"Anemia training artifacts saved in {SAVED_MODELS_DIR}")
    return metrics_summary


if __name__ == '__main__':
    main(epochs=5, batch_size=16, lr=1e-4)
