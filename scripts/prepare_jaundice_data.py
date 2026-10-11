"""
scripts/prepare_jaundice_data.py
Extracts web-scraped images from Hugging Face Bleachcarte/Jaundice_Dataset
and creates data/manifests/jaundice_manifest.csv with strict CC BY-NC 4.0 metadata.
"""

import os
import sys
import io
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_IMG_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'jaundice', 'images')
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'jaundice_manifest.csv')
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def main():
    print("=== PREPARING JAUNDICE DATASET (Bleachcarte/Jaundice_Dataset) ===")
    url_train = 'https://huggingface.co/datasets/Bleachcarte/Jaundice_Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'
    url_test = 'https://huggingface.co/datasets/Bleachcarte/Jaundice_Dataset/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet'
    
    df_train = pd.read_parquet(url_train)
    df_test = pd.read_parquet(url_test)
    
    print(f"Loaded train: {len(df_train)}, test: {len(df_test)}")
    
    # Class mapping: 0 -> healthy (normal), 1 -> jaundice (visibly jaundiced)
    class_names = {0: 'normal', 1: 'visibly_jaundiced'}
    
    records = []
    
    # Extract train
    for idx, row in df_train.iterrows():
        img_dict = row['image']
        label = int(row['label'])
        raw_bytes = img_dict['bytes']
        md5_hash = hashlib.md5(raw_bytes).hexdigest()
        
        fname = f"jaundice_train_{idx:03d}_{md5_hash[:8]}.jpg"
        fpath = os.path.join(OUT_IMG_DIR, fname)
        
        im = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        im.save(fpath, format='JPEG', quality=95)
        w, h = im.size
        
        records.append({
            'image_id': f"jaundice_train_{idx:03d}",
            'filename': fname,
            'relative_path': os.path.relpath(fpath, WORKSPACE_ROOT).replace('\\', '/'),
            'label': label,
            'class_name': class_names[label],
            'dataset_source': 'Bleachcarte_HF',
            'license': 'CC BY-NC 4.0',
            'md5': md5_hash,
            'original_split': 'train',
            'width': w,
            'height': h,
            'aspect_ratio': round(w / h, 4),
            'area_pixels': w * h
        })
        
    # Extract test
    for idx, row in df_test.iterrows():
        img_dict = row['image']
        label = int(row['label'])
        raw_bytes = img_dict['bytes']
        md5_hash = hashlib.md5(raw_bytes).hexdigest()
        
        fname = f"jaundice_test_{idx:03d}_{md5_hash[:8]}.jpg"
        fpath = os.path.join(OUT_IMG_DIR, fname)
        
        im = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        im.save(fpath, format='JPEG', quality=95)
        w, h = im.size
        
        records.append({
            'image_id': f"jaundice_test_{idx:03d}",
            'filename': fname,
            'relative_path': os.path.relpath(fpath, WORKSPACE_ROOT).replace('\\', '/'),
            'label': label,
            'class_name': class_names[label],
            'dataset_source': 'Bleachcarte_HF',
            'license': 'CC BY-NC 4.0',
            'md5': md5_hash,
            'original_split': 'test',
            'width': w,
            'height': h,
            'aspect_ratio': round(w / h, 4),
            'area_pixels': w * h
        })
        
    df_manifest = pd.DataFrame(records)
    print(f"Total extracted: {len(df_manifest)}")
    
    # Partition original_split 'train' (N=256) into train (N=206) and val (N=50), keeping test (N=65) held out
    np.random.seed(42)
    df_manifest['split'] = 'train'
    test_mask = df_manifest['original_split'] == 'test'
    df_manifest.loc[test_mask, 'split'] = 'test'
    
    train_indices = df_manifest[df_manifest['original_split'] == 'train'].index.values.copy()
    np.random.shuffle(train_indices)
    
    val_count = 50
    val_indices = train_indices[:val_count]
    train_remain = train_indices[val_count:]
    
    df_manifest.loc[val_indices, 'split'] = 'val'
    df_manifest.loc[train_remain, 'split'] = 'train'
    
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved Jaundice manifest to {MANIFEST_PATH}")
    print("Split distribution:")
    print(pd.crosstab(df_manifest['split'], df_manifest['class_name'], margins=True))

if __name__ == '__main__':
    main()
