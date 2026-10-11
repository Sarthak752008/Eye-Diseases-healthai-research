"""
train_conjunctivitis.py - Training pipeline for MobileNetV3-Small Conjunctivitis Classifier.
Trains on verified Normal vs Conjunctivitis images using stratified splits.
Saves model checkpoint, training logs, loss curves, and validation/test metrics.
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

from src.preprocessing.dataset_conjunctivitis import build_conjunctivitis_dataloaders
from src.models.conjunctivitis_model import (
    ConjunctivitisClassifier,
    save_conjunctivitis_checkpoint,
    load_conjunctivitis_checkpoint
)
from src.evaluation.evaluate import compute_classification_metrics

SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'conjunctivitis_manifest.csv')
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'conjunctivitis')

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, acc


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = round(avg_loss, 4)
    return avg_loss, metrics


def main(epochs: int = 5, batch_size: int = 32, lr: float = 5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Conjunctivitis Model Training on device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    loaders = build_conjunctivitis_dataloaders(
        manifest_csv=MANIFEST_PATH,
        workspace_root=WORKSPACE_ROOT,
        batch_size=batch_size,
        use_processed_dir=PROCESSED_DIR
    )
    
    # Compute class weights to account for class imbalance (454 normal : 250 conjunctivitis)
    train_labels = [row['class_label'] for _, row in pd.read_csv(MANIFEST_PATH)[lambda x: x['split'] == 'train'].iterrows()]
    n_samples = len(train_labels)
    n_normal = sum(1 for l in train_labels if l == 0)
    n_conj = sum(1 for l in train_labels if l == 1)
    class_weights = torch.tensor([n_samples / (2.0 * n_normal), n_samples / (2.0 * n_conj)], dtype=torch.float32).to(device)
    print(f"Class weights: Normal={class_weights[0]:.2f}, Conjunctivitis={class_weights[1]:.2f}")
    
    # 2. Build Model
    model = ConjunctivitisClassifier(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_val_metrics = {}
    training_log = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_metrics = evaluate_loader(model, loaders['val'], criterion, device)
        scheduler.step()
        epoch_dur = time.time() - t0
        
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall_sensitivity'],
            'val_f1': val_metrics['f1_score'],
            'val_roc_auc': val_metrics['roc_auc'],
            'epoch_time_sec': round(epoch_dur, 2)
        }
        training_log.append(log_entry)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f} ({epoch_dur:.1f}s)")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
            save_conjunctivitis_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                save_path=checkpoint_path,
                class_mapping={'0': 'normal', '1': 'conjunctivitis'},
                config={'model': 'MobileNetV3-Small', 'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
            )
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s.")
    
    # 3. Evaluate Best Model on Test Set
    best_model = load_conjunctivitis_checkpoint(os.path.join(SAVED_MODELS_DIR, 'best_model.pth'), device=str(device))
    test_loss, test_metrics = evaluate_loader(best_model, loaders['test'], criterion, device)
    print(f"=== Final Test Metrics ===")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall_sensitivity']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")
    
    # 4. Save artifacts
    # Training log CSV
    log_df = pd.DataFrame(training_log)
    log_csv = os.path.join(SAVED_MODELS_DIR, 'training_log.csv')
    log_df.to_csv(log_csv, index=False)
    
    # Class mapping JSON
    mapping_path = os.path.join(SAVED_MODELS_DIR, 'class_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({'0': 'normal', '1': 'conjunctivitis'}, f, indent=2)
        
    # Metrics JSON
    metrics_summary = {
        'model_name': 'MobileNetV3-Small Conjunctivitis Classifier',
        'training_time_seconds': round(total_time, 2),
        'best_val_metrics': best_val_metrics,
        'final_test_metrics': test_metrics
    }
    metrics_path = os.path.join(SAVED_MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
        
    # Training curve report
    curve_path = os.path.join(REPORTS_DIR, 'conjunctivitis_training_curve.json')
    with open(curve_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
        
    print(f"Conjunctivitis training artifacts saved in {SAVED_MODELS_DIR}")
    return metrics_summary


if __name__ == '__main__':
    main(epochs=5, batch_size=32, lr=5e-4)
