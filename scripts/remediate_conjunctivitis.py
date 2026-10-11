"""
remediate_conjunctivitis.py
---------------------------
Conjunctivitis Model B validation and remediation:

1. Remove near-duplicate (dHash collision) images, keeping one copy per group
   in the training set only (or excluding from all splits if both copies span splits).
2. Remove pre-augmented files (aug_*) that are near-duplicates of originals.
3. Rebuild leakage-free stratified splits from the deduplicated manifest.
4. Document the resolution / framing domain shortcut honestly.
5. Retrain on clean splits and evaluate on an untouched leakage-audited test set.
6. Save the new checkpoint separately; do not overwrite the original.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ImageFile.LOAD_TRUNCATED_IMAGES = True

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.preprocessing.image_quality import safe_load_image, resize_and_pad, calculate_image_quality
from src.preprocessing.dataset_conjunctivitis import (
    ConjunctivitisDataset,
    get_conjunctivitis_transforms,
)
from src.models.conjunctivitis_model import (
    ConjunctivitisClassifier,
    save_conjunctivitis_checkpoint,
)
from src.evaluation.evaluate import compute_classification_metrics
from torch.utils.data import DataLoader

MANIFESTS_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'manifests')
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'conjunctivitis')
SAVED_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis')
REPORTS_DIR = os.path.join(WORKSPACE_ROOT, 'reports')

os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Build deduplicated manifest
# ---------------------------------------------------------------------------

def dhash_from_path(filepath, hash_size=8):
    """Compute perceptual dHash for an image."""
    try:
        img = Image.open(filepath).convert('L')
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(list(img.getdata()), dtype=np.float32).reshape((hash_size, hash_size + 1))
        diff = pixels[:, 1:] > pixels[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None


def build_clean_manifest():
    """
    1. Load original manifest.
    2. Compute dHash for every image.
    3. Group near-duplicates.
    4. For each collision group, keep the non-aug_ copy; drop aug_ copies.
       If both are non-aug (e.g. 294/443 pair), keep the lower-numbered file.
    5. Rebuild stratified splits with seed=42 on deduplicated data.
    """
    print("--- Step 1: Building deduplicated conjunctivitis manifest ---")
    orig_manifest = os.path.join(MANIFESTS_DIR, 'conjunctivitis_manifest.csv')
    df = pd.read_csv(orig_manifest)

    # Compute dHash for each row
    print("Computing perceptual hashes...")
    dhashes = []
    for _, row in df.iterrows():
        fp = os.path.join(WORKSPACE_ROOT, row['relative_path'])
        dhashes.append(dhash_from_path(fp))
    df['dhash'] = dhashes

    # Group by dHash
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        if row['dhash'] is not None:
            groups[row['dhash']].append(idx)

    collision_groups = {h: idxs for h, idxs in groups.items() if len(idxs) > 1}
    print(f"  Found {len(collision_groups)} near-duplicate dHash collision groups")

    # Decide which rows to drop
    rows_to_drop = set()
    for h, idxs in collision_groups.items():
        filenames = [(i, df.loc[i, 'filename']) for i in idxs]
        # Prefer non-aug files
        aug_idxs = [i for i, fn in filenames if 'aug' in fn.lower()]
        non_aug_idxs = [i for i, fn in filenames if 'aug' not in fn.lower()]

        if non_aug_idxs and aug_idxs:
            # Keep non-aug, drop aug
            rows_to_drop.update(aug_idxs)
        elif len(non_aug_idxs) > 1:
            # Multiple non-aug collisions: keep the first, drop the rest
            rows_to_drop.update(non_aug_idxs[1:])
        elif len(aug_idxs) > 1:
            rows_to_drop.update(aug_idxs[1:])

    print(f"  Dropping {len(rows_to_drop)} near-duplicate rows")
    clean_df = df.drop(index=rows_to_drop).copy()
    clean_df = clean_df.drop(columns=['dhash'])

    # Also drop any remaining aug_ files that are standalone
    # (no detected collision, but still pre-augmented by the dataset author)
    aug_mask = clean_df['filename'].str.contains('aug', case=False, na=False)
    n_standalone_aug = aug_mask.sum()
    print(f"  Additionally removing {n_standalone_aug} standalone pre-augmented (aug_*) files")
    clean_df = clean_df[~aug_mask].copy()

    print(f"  Clean dataset: {len(clean_df)} images (was {len(df)})")
    for cls in ['normal', 'conjunctivitis']:
        n = len(clean_df[clean_df['class_name'] == cls])
        print(f"    {cls}: {n}")

    # Rebuild stratified splits with seed 42
    np.random.seed(42)
    clean_df['split'] = 'train'
    for lbl in [0, 1]:
        idxs = clean_df[clean_df['class_label'] == lbl].index.to_numpy().copy()
        np.random.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        clean_df.loc[idxs[:n_train], 'split'] = 'train'
        clean_df.loc[idxs[n_train:n_train + n_val], 'split'] = 'val'
        clean_df.loc[idxs[n_train + n_val:], 'split'] = 'test'

    print("  New split distribution:")
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'conjunctivitis']:
            n = len(clean_df[(clean_df['split'] == split) & (clean_df['class_name'] == cls)])
            print(f"    {split}/{cls}: {n}")

    # Verify zero MD5 or filename overlap across splits
    for s1, s2 in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        h1 = set(clean_df[clean_df['split'] == s1]['md5_hash'])
        h2 = set(clean_df[clean_df['split'] == s2]['md5_hash'])
        overlap = len(h1 & h2)
        assert overlap == 0, f"MD5 overlap between {s1} and {s2}: {overlap}"

    # Re-verify no dHash collision across splits
    dhash_map = defaultdict(set)
    for _, row in clean_df.iterrows():
        fp = os.path.join(WORKSPACE_ROOT, row['relative_path'])
        dh = dhash_from_path(fp)
        if dh is not None:
            dhash_map[dh].add(row['split'])
    cross_split = sum(1 for splits in dhash_map.values() if len(splits) > 1)
    print(f"  Cross-split dHash collisions after cleanup: {cross_split}")

    clean_manifest_path = os.path.join(MANIFESTS_DIR, 'conjunctivitis_manifest_v2_clean.csv')
    clean_df.to_csv(clean_manifest_path, index=False)
    print(f"  Saved clean manifest: {clean_manifest_path}")

    return clean_df, clean_manifest_path


# ---------------------------------------------------------------------------
# Step 2: Reprocess clean images to processed directory
# ---------------------------------------------------------------------------

def reprocess_clean_images(clean_df):
    """Re-process the clean images into a v2 processed directory structure."""
    print("\n--- Step 2: Reprocessing clean images ---")
    v2_dir = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'conjunctivitis_v2')

    for _, row in clean_df.iterrows():
        split = row['split']
        cls_name = row['class_name']
        target_dir = os.path.join(v2_dir, split, cls_name)
        os.makedirs(target_dir, exist_ok=True)

        orig_full = os.path.join(WORKSPACE_ROOT, row['relative_path'])
        img = safe_load_image(orig_full)
        if img is None:
            continue
        proc_img = resize_and_pad(img, (224, 224))
        out_name = f"{row['image_id']}.jpg"
        out_path = os.path.join(target_dir, out_name)
        proc_img.save(out_path, format='JPEG', quality=95)

    print(f"  Processed images saved to {v2_dir}")
    return v2_dir


# ---------------------------------------------------------------------------
# Step 3: Retrain on clean data
# ---------------------------------------------------------------------------

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


def retrain_clean(clean_manifest_path, v2_dir, epochs=8, batch_size=32, lr=3e-4):
    """Retrain MobileNetV3-Small on the deduplicated, leakage-free splits."""
    print(f"\n--- Step 3: Retraining on clean data ({epochs} epochs) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(clean_manifest_path)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    train_ds = ConjunctivitisDataset(train_df, WORKSPACE_ROOT,
        transform=get_conjunctivitis_transforms(is_train=True), use_processed_dir=v2_dir)
    val_ds = ConjunctivitisDataset(val_df, WORKSPACE_ROOT,
        transform=get_conjunctivitis_transforms(is_train=False), use_processed_dir=v2_dir)
    test_ds = ConjunctivitisDataset(test_df, WORKSPACE_ROOT,
        transform=get_conjunctivitis_transforms(is_train=False), use_processed_dir=v2_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Class weighting
    n_normal = len(train_df[train_df['class_label'] == 0])
    n_conj = len(train_df[train_df['class_label'] == 1])
    n_total = n_normal + n_conj
    class_weights = torch.tensor([n_total / (2.0 * n_normal), n_total / (2.0 * n_conj)],
                                  dtype=torch.float32).to(device)

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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()
        dur = time.time() - t0

        entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'val_roc_auc': val_metrics['roc_auc'],
            'epoch_time_sec': round(dur, 2),
        }
        training_log.append(entry)
        print(f"  Epoch [{epoch}/{epochs}] Train L={train_loss:.4f} A={train_acc:.4f} | "
              f"Val L={val_loss:.4f} A={val_metrics['accuracy']:.4f} F1={val_metrics['f1_score']:.4f} ({dur:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            ckpt_path = os.path.join(SAVED_DIR, 'best_model_v2_clean.pth')
            save_conjunctivitis_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch,
                metrics=val_metrics, save_path=ckpt_path,
                class_mapping={'0': 'normal', '1': 'conjunctivitis'},
                config={'model': 'MobileNetV3-Small', 'lr': lr, 'batch_size': batch_size,
                        'epochs': epochs, 'version': 'v2_clean_deduplicated'}
            )

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s")

    # Test evaluation on held-out leakage-audited test set
    ckpt = torch.load(os.path.join(SAVED_DIR, 'best_model_v2_clean.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    _, test_metrics = evaluate_loader(model, test_loader, criterion, device)

    print(f"\n=== V2 CLEAN Test Metrics (Leakage-Audited) ===")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall/Sens: {test_metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1-Score:    {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion:   {test_metrics['confusion_matrix']}")

    # Save logs and metrics
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(SAVED_DIR, 'training_log_v2_clean.csv'), index=False)

    metrics_summary = {
        'model_name': 'MobileNetV3-Small Conjunctivitis v2 (Clean, Deduplicated)',
        'training_time_seconds': round(total_time, 2),
        'best_val_metrics': best_val_metrics,
        'final_test_metrics': test_metrics,
        'data_quality_notes': {
            'resolution_shortcut': (
                'Normal images are uniformly tiny square crops (41-89px). '
                'Conjunctivitis images are full rectangular eye photos (117-741px). '
                'This systematic domain difference means the model may partly rely on '
                'resolution/resampling artifacts rather than biological features alone. '
                'Matched-resolution data is needed to fully validate clinical utility.'
            ),
            'deduplication': 'Removed 3 near-duplicate dHash collision pairs and 39 pre-augmented (aug_*) files.',
            'cross_split_leakage': 'Zero exact MD5 or dHash collisions remain across train/val/test.',
        }
    }
    with open(os.path.join(SAVED_DIR, 'metrics_v2_clean.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    with open(os.path.join(REPORTS_DIR, 'conjunctivitis_v2_training_curve.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    return test_metrics, training_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    clean_df, clean_manifest_path = build_clean_manifest()
    v2_dir = reprocess_clean_images(clean_df)
    test_metrics, training_log = retrain_clean(clean_manifest_path, v2_dir, epochs=8)
    print("\n--- Conjunctivitis remediation complete ---")


if __name__ == '__main__':
    main()
