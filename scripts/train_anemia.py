"""
train_anemia.py - Training pipeline for Anemia (Hb) Regressors.
Predicts numerical laboratory Hemoglobin (g/dL) with strict subject-level isolation.
Supports ResNet-50 and MobileNetV3-Small architectures.
Tracks both view-level and patient-level metrics (MAE, RMSE, Pearson r, R^2).
Saves best_model.pth, last_checkpoint.pth, and training history logs.
"""

import os
import sys
import json
import time
import argparse
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
    save_anemia_checkpoint,
    load_anemia_checkpoint
)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    view_mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
    return avg_loss, view_mae


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_subjects = []
    all_views = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['hb'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_subjects.extend(batch['subject_id'])
            all_views.extend(batch['view_type'])
            
    avg_loss = total_loss / len(loader.dataset)
    
    df_eval = pd.DataFrame({
        'subject_id': all_subjects,
        'view_type': all_views,
        'actual': all_targets,
        'pred': all_preds
    })
    
    # View-level metrics
    view_diff = df_eval['actual'] - df_eval['pred']
    view_mae = float(np.mean(np.abs(view_diff)))
    view_rmse = float(np.sqrt(np.mean(view_diff**2)))
    
    # Patient-level aggregation
    patient_df = df_eval.groupby('subject_id').agg({'actual': 'first', 'pred': 'mean'}).reset_index()
    pat_actual = patient_df['actual'].values
    pat_pred = patient_df['pred'].values
    
    pat_diff = pat_actual - pat_pred
    patient_mae = float(np.mean(np.abs(pat_diff)))
    patient_rmse = float(np.sqrt(np.mean(pat_diff**2)))
    
    if len(pat_actual) > 1 and np.std(pat_actual) > 0 and np.std(pat_pred) > 0:
        pearson_r = float(np.corrcoef(pat_actual, pat_pred)[0, 1])
    else:
        pearson_r = 0.0
        
    ss_res = np.sum(pat_diff**2)
    ss_tot = np.sum((pat_actual - np.mean(pat_actual))**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    metrics = {
        'loss': round(avg_loss, 4),
        'view_mae_g_dl': round(view_mae, 4),
        'view_rmse_g_dl': round(view_rmse, 4),
        'patient_mae_g_dl': round(patient_mae, 4),
        'patient_rmse_g_dl': round(patient_rmse, 4),
        'pearson_r': round(pearson_r, 4),
        'r2': round(r2, 4),
        'sample_count': len(df_eval),
        'patient_count': len(patient_df)
    }
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Anemia (Hb) Regression Model")
    parser.add_argument('--model_type', type=str, choices=['resnet50', 'mobilenet'], default='resnet50',
                        help="Model architecture: resnet50 or mobilenet")
    parser.add_argument('--epochs', type=int, default=10, help="Total training epochs (up to 100)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=12, help="Early stopping patience in epochs")
    parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint if exists")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Anemia Model Training ({args.model_type.upper()}) on {device} ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data with strict subject-level splitting
    loaders = build_anemia_dataloaders(
        manifest_csv=MANIFEST_PATH,
        workspace_root=WORKSPACE_ROOT,
        batch_size=args.batch_size,
        use_processed_dir=PROCESSED_DIR
    )
    print(f"Dataset split sizes: Train={len(loaders['datasets']['train'])}, "
          f"Val={len(loaders['datasets']['val'])}, Test={len(loaders['datasets']['test'])}")
    
    # 2. Build Model
    if args.model_type == 'resnet50':
        model = AnemiaRegressor(pretrained=True, dropout=0.2).to(device)
        best_ckpt_name = 'best_model.pth'
        last_ckpt_name = 'last_checkpoint.pth'
    else:
        model = AnemiaMobileNetRegressor(pretrained=True, dropout=0.2).to(device)
        best_ckpt_name = 'mobilenet_model.pth'
        last_ckpt_name = 'mobilenet_last_checkpoint.pth'

    best_ckpt_path = os.path.join(SAVED_MODELS_DIR, best_ckpt_name)
    last_ckpt_path = os.path.join(SAVED_MODELS_DIR, last_ckpt_name)

    # Smooth L1 / Huber Loss
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 1
    best_val_mae = float('inf')
    best_val_metrics = {}
    epochs_no_improve = 0
    training_log = []
    
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming training from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_mae = ckpt.get('best_score', float('inf'))
        print(f"Resumed at epoch {start_epoch}, previous best Val MAE: {best_val_mae}")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_metrics = evaluate_loader(model, loaders['val'], criterion, device)
        scheduler.step()
        epoch_dur = time.time() - t0
        
        current_val_mae = val_metrics['patient_mae_g_dl']
        
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_mae_g_dl': round(train_mae, 4),
            'val_loss': round(val_loss, 4),
            'val_view_mae_g_dl': val_metrics['view_mae_g_dl'],
            'val_patient_mae_g_dl': val_metrics['patient_mae_g_dl'],
            'val_patient_rmse_g_dl': val_metrics['patient_rmse_g_dl'],
            'val_pearson_r': val_metrics['pearson_r'],
            'val_r2': val_metrics['r2'],
            'epoch_time_sec': round(epoch_dur, 2)
        }
        training_log.append(log_entry)
        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Pat MAE: {val_metrics['patient_mae_g_dl']:.2f} g/dL, "
              f"Pat RMSE: {val_metrics['patient_rmse_g_dl']:.2f} g/dL, R²: {val_metrics['r2']:.3f} ({epoch_dur:.1f}s)")
        
        # Save last checkpoint for resumption
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_val_mae,
            'metrics': val_metrics,
            'config': vars(args)
        }, last_ckpt_path)
        
        # Checkpoint selection on Patient-level Validation MAE
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            
            save_anemia_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                save_path=best_ckpt_path,
                config={'model': args.model_type, 'target': 'Hemoglobin (g/dL)', 'lr': args.lr, 'batch_size': args.batch_size, 'epochs': args.epochs}
            )
            print(f"  --> New best validation checkpoint saved! (Val Patient MAE: {best_val_mae:.4f} g/dL)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {args.patience} epochs).")
                break
                
    total_time = time.time() - start_time
    print(f"Anemia training ({args.model_type}) finished in {total_time:.1f}s.")
    
    # 3. Final Held-Out Evaluation on Best Model Checkpoint
    if os.path.exists(best_ckpt_path):
        if args.model_type == 'resnet50':
            eval_best = load_anemia_checkpoint(best_ckpt_path, device=str(device))
        else:
            eval_best = AnemiaMobileNetRegressor(pretrained=False).to(device)
            ckpt = torch.load(best_ckpt_path, map_location=device)
            eval_best.load_state_dict(ckpt['model_state_dict'])
            eval_best.eval()
            
        test_loss, test_metrics = evaluate_loader(eval_best, loaders['test'], criterion, device)
        print(f"=== Held-Out Test Results ({args.model_type}) ===")
        print(f"Patient MAE:    {test_metrics['patient_mae_g_dl']:.4f} g/dL")
        print(f"Patient RMSE:   {test_metrics['patient_rmse_g_dl']:.4f} g/dL")
        print(f"View MAE:       {test_metrics['view_mae_g_dl']:.4f} g/dL")
        print(f"View RMSE:      {test_metrics['view_rmse_g_dl']:.4f} g/dL")
        print(f"Pearson r:      {test_metrics['pearson_r']:.4f}")
        print(f"R² Score:       {test_metrics['r2']:.4f}")
    else:
        test_metrics = {}

    # 4. Save artifacts
    if training_log:
        log_df = pd.DataFrame(training_log)
        log_csv = os.path.join(SAVED_MODELS_DIR, f'{args.model_type}_training_log.csv')
        log_df.to_csv(log_csv, index=False)
        
    metrics_summary = {
        'model_name': f'{args.model_type} Anemia Regressor',
        'target': 'Hemoglobin (g/dL)',
        'training_time_seconds': round(total_time, 2),
        'best_val_metrics': best_val_metrics,
        'final_test_metrics': test_metrics,
        'config': vars(args)
    }
    summary_path = os.path.join(SAVED_MODELS_DIR, f'{args.model_type}_metrics_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
        
    return metrics_summary


if __name__ == '__main__':
    main()
