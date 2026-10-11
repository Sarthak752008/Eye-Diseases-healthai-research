"""
scripts/train_jaundice.py
Trains and evaluates JaundiceClassifier (MobileNetV3-Small) on the Diverse Jaundice Eye Dataset.
Preserves scleral color fidelity, respects CC BY-NC 4.0 terms, and reports held-out metrics.
"""

import os
import sys
import json
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

from src.models.jaundice_model import JaundiceClassifier, save_jaundice_checkpoint
from src.preprocessing.image_quality import safe_load_image, resize_and_pad

MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

class JaundiceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(WORKSPACE_ROOT, row['relative_path'])
        im = safe_load_image(img_path)
        if im is None:
            im = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            im = resize_and_pad(im, (224, 224))
            
        if self.transform:
            tensor = self.transform(im)
        else:
            tensor = transforms.ToTensor()(im)
            
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
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING JAUNDICE / SCLERAL ICTERUS MODEL ON {device} ===")
    df = pd.read_csv(MANIFEST_PATH)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Color-preserving transforms: NO hue shift (crucial for icterus / yellowing)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(JaundiceDataset(train_df, train_transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(JaundiceDataset(val_df, eval_transform), batch_size=16, shuffle=False)
    test_loader = DataLoader(JaundiceDataset(test_df, eval_transform), batch_size=16, shuffle=False)
    
    model = JaundiceClassifier(pretrained=True, num_classes=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    
    best_val_f1 = -1.0
    best_val_metrics = {}
    ckpt_path = os.path.join(SAVED_MODELS_DIR, 'best_model.pth')
    
    print("Training MobileNetV3-Small for 5 epochs...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_val_metrics = val_metrics
            save_jaundice_checkpoint(
                model=model,
                save_path=ckpt_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                class_mapping={'0': 'normal', '1': 'visibly_jaundiced'},
                config={'model': 'MobileNetV3-Small', 'dataset': 'Bleachcarte_HF', 'target': 'Visible Scleral Icterus'}
            )
            
    # Evaluate best model on test set
    best_model = JaundiceClassifier(pretrained=False, num_classes=2)
    best_model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])
    best_model.to(device).eval()
    test_metrics = evaluate_model(best_model, test_loader, device)
    
    print("\n=== Jaundice Held-Out Test Results ===")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {test_metrics['recall_sensitivity']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"F1-Score:    {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")
    
    summary = {
        'model_name': 'MobileNetV3-Small Jaundice Classifier',
        'target': 'Visible Scleral Icterus (Research Only)',
        'licence': 'CC BY-NC 4.0 (Non-Commercial Research Only)',
        'best_validation_metrics': best_val_metrics,
        'final_test_metrics': test_metrics
    }
    
    with open(os.path.join(SAVED_MODELS_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(SAVED_MODELS_DIR, 'class_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({'0': 'normal', '1': 'visibly_jaundiced'}, f, indent=2)
        
    print(f"Jaundice training completed and saved to {SAVED_MODELS_DIR}")

if __name__ == '__main__':
    main()
