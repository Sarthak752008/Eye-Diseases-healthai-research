"""
scripts/train_stye.py
---------------------
Trains and evaluates StyeClassifier (MobileNetV3-Small) on the Stye / Hordeolum Dataset.
Preserves subject-isolated splits, evaluates held-out metrics, and saves model artifacts.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.models.stye_model import StyeClassifier, save_stye_checkpoint
from src.preprocessing.image_quality import safe_load_image, resize_and_pad

MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'stye_hordeolum_manifest.csv')
SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'stye')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class StyeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self._cache:
            im = self._cache[idx]
        else:
            row = self.df.iloc[idx]
            img_path = os.path.join(WORKSPACE_ROOT, row['relative_path'])
            im = safe_load_image(img_path)
            if im is None:
                im = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                im = resize_and_pad(im, (224, 224))
            self._cache[idx] = im

        if self.transform:
            tensor = self.transform(im)
        else:
            tensor = transforms.ToTensor()(im)

        row = self.df.iloc[idx]
        return {
            'image': tensor,
            'label': torch.tensor(int(row['label']), dtype=torch.long),
            'image_id': row['image_id'],
            'filename': row['filename']
        }


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall_sensitivity': float(rec),
        'specificity': float(spec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            targets = batch['label'].to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    return compute_metrics(all_targets, all_preds, all_probs), all_targets, all_preds, all_probs


def train_stye_model(epochs: int = 10, batch_size: int = 32, lr: float = 3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING STYE / HORDEOLUM CLASSIFIER (Device: {device}) ===")

    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"Data Splits: Train={len(train_df)} (Subjects: {train_df['subject_id'].nunique()}), "
          f"Val={len(val_df)} (Subjects: {val_df['subject_id'].nunique()}), "
          f"Test={len(test_df)} (Subjects: {test_df['subject_id'].nunique()})")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),  # conservative to preserve erythema
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = StyeDataset(train_df, transform=train_transform)
    val_ds = StyeDataset(val_df, transform=eval_transform)
    test_ds = StyeDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_neg = (train_df['label'] == 0).sum()
    n_pos = (train_df['label'] == 1).sum()
    weight_neg = float(n_pos) / float(n_neg) if n_neg > 0 else 1.0
    class_weights = torch.tensor([weight_neg, 1.0], dtype=torch.float).to(device)
    print(f"Class counts in train: Neg(Normal)={n_neg}, Pos(Stye)={n_pos}. Weights: {class_weights.tolist()}")

    model = StyeClassifier(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_val_metrics = {}
    best_checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    log_records = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_probs, train_targets = [], [], []

        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            train_probs.extend(probs.detach().cpu().numpy().tolist())
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_targets.extend(labels.cpu().numpy().tolist())

        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        train_metrics = compute_metrics(train_targets, train_preds, train_probs)

        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1_score']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} Rec: {val_metrics['recall_sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f} F1: {val_metrics['f1_score']:.4f} AUC: {val_metrics['roc_auc']:.4f}", flush=True)

        log_records.append({
            'epoch': epoch,
            'train_loss': round(epoch_loss, 4),
            'train_acc': round(train_metrics['accuracy'], 4),
            'train_f1': round(train_metrics['f1_score'], 4),
            'val_acc': round(val_metrics['accuracy'], 4),
            'val_rec': round(val_metrics['recall_sensitivity'], 4),
            'val_spec': round(val_metrics['specificity'], 4),
            'val_f1': round(val_metrics['f1_score'], 4),
            'val_auc': round(val_metrics['roc_auc'], 4)
        })

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_val_metrics = val_metrics
            save_stye_checkpoint(
                model=model,
                save_path=best_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                class_mapping={'0': 'normal', '1': 'stye_hordeolum'},
                config={
                    'architecture': 'MobileNetV3-Small',
                    'input_size': [224, 224],
                    'normalization': 'ImageNet',
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'class_weights': class_weights.tolist()
                }
            )

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.2f}s. Best Val F1: {best_val_f1:.4f}")

    best_model = StyeClassifier(pretrained=False, num_classes=2).to(device)
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    best_model.load_state_dict(ckpt['model_state_dict'])

    test_metrics, test_targets, test_preds, test_probs = evaluate_model(best_model, test_loader, device)
    print("\n" + "=" * 60)
    print("FINAL HELD-OUT TEST METRICS (ZERO SUBJECT LEAKAGE):")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Sensitivity: {test_metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score:    {test_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TN={test_metrics['true_negatives']}, FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}, TP={test_metrics['true_positives']}")
    print("=" * 60)

    pd.DataFrame(log_records).to_csv(os.path.join(SAVED_MODELS_DIR, 'training_log.csv'), index=False)

    class_mapping = {'0': 'normal', '1': 'stye_hordeolum'}
    with open(os.path.join(SAVED_MODELS_DIR, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    metrics_payload = {
        'disease': 'Stye / Hordeolum',
        'architecture': 'MobileNetV3-Small',
        'dataset': 'Roboflow Stye/Hordeolum paired with Normal Ocular Controls',
        'modality': 'Anterior Eyelid / Ocular Photography',
        'total_images': len(df),
        'unique_subjects': int(df['subject_id'].nunique()),
        'subject_isolated_split': {
            'train_subjects': int(train_df['subject_id'].nunique()),
            'val_subjects': int(val_df['subject_id'].nunique()),
            'test_subjects': int(test_df['subject_id'].nunique()),
            'train_images': len(train_df),
            'val_images': len(val_df),
            'test_images': len(test_df)
        },
        'validation_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'training_duration_seconds': round(train_duration, 2)
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    print("Saved metrics.json, class_mapping.json, and training_log.csv successfully.")
    return metrics_payload

if __name__ == '__main__':
    train_stye_model(epochs=6, batch_size=32, lr=3e-4)
