"""
scripts/prepare_slid_data.py
Extracts matched clinical slit-lamp images from SLID (xumingyu-hub/SLID)
and creates a leak-free stratified manifest for visible conjunctival injection screening.
"""

import os
import sys
import json
import zipfile
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SLID_ZIP = r"C:\Users\sarth\.gemini\antigravity-ide\brain\c9c42d26-1e4d-46e7-891e-3d0daa7e7903\scratch\slid\Original_Slit-lamp_Images.zip"
SLID_CSV = r"C:\Users\sarth\.gemini\antigravity-ide\brain\c9c42d26-1e4d-46e7-891e-3d0daa7e7903\scratch\slid\Annotations.csv"

OUT_IMG_DIR = os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'slid_injection', 'images')
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'slid_injection_manifest.csv')
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def main():
    print("=== PREPARING CLINICAL SLID CONJUNCTIVAL INJECTION DATASET ===")
    df_ann = pd.read_csv(SLID_CSV)
    
    image_lesions = {}
    for _, row in df_ann.iterrows():
        fn = str(row['filename']).strip()
        if fn not in image_lesions:
            image_lesions[fn] = set()
        try:
            data = json.loads(row['attributes'])
            lesion = data.get('lesion', '').strip()
            if lesion:
                image_lesions[fn].add(lesion)
        except Exception:
            pass
            
    selected = []
    for fn, lesions in image_lesions.items():
        if 'Conjunctival injection' in lesions:
            selected.append({'filename': fn, 'label': 1, 'class_name': 'conjunctival_injection'})
        elif len(lesions) == 0:
            selected.append({'filename': fn, 'label': 0, 'class_name': 'control_normal'})
            
    df_sel = pd.DataFrame(selected)
    print(f"Total candidate images: {len(df_sel)} (Normal: {(df_sel['label']==0).sum()}, Injection: {(df_sel['label']==1).sum()})")
    
    # Extract images from zip
    records = []
    with zipfile.ZipFile(SLID_ZIP, 'r') as z:
        namelist_map = {os.path.basename(n): n for n in z.namelist() if n.endswith('.png') and not n.startswith('__MACOSX')}
        
        for idx, row in df_sel.iterrows():
            fn = row['filename']
            if fn not in namelist_map:
                continue
            z_path = namelist_map[fn]
            data = z.read(z_path)
            
            # MD5 hash
            md5_hash = hashlib.md5(data).hexdigest()
            
            # Save extracted image
            out_name = f"slid_{fn}"
            out_file = os.path.join(OUT_IMG_DIR, out_name)
            if not os.path.exists(out_file):
                with open(out_file, 'wb') as f:
                    f.write(data)
                    
            # Open image for dimension audit
            with Image.open(out_file) as im:
                w, h = im.size
                
            records.append({
                'image_id': f"slid_{os.path.splitext(fn)[0]}",
                'filename': out_name,
                'relative_path': os.path.relpath(out_file, WORKSPACE_ROOT).replace('\\', '/'),
                'label': int(row['label']),
                'class_name': row['class_name'],
                'dataset_source': 'SLID_Zhejiang',
                'license': 'CC BY 4.0',
                'md5': md5_hash,
                'width': w,
                'height': h,
                'aspect_ratio': round(w / h, 4),
                'area_pixels': w * h
            })
            
    df_manifest = pd.DataFrame(records)
    print(f"Extracted {len(df_manifest)} images successfully.")
    
    # Stratified 70/15/15 split with seed 42
    np.random.seed(42)
    df_manifest['split'] = 'train'
    
    for cls in [0, 1]:
        cls_idx = df_manifest[df_manifest['label'] == cls].index.values.copy()
        np.random.shuffle(cls_idx)
        n_total = len(cls_idx)
        n_val = int(0.15 * n_total)
        n_test = int(0.15 * n_total)
        
        val_idx = cls_idx[:n_val]
        test_idx = cls_idx[n_val:n_val + n_test]
        train_idx = cls_idx[n_val + n_test:]
        
        df_manifest.loc[val_idx, 'split'] = 'val'
        df_manifest.loc[test_idx, 'split'] = 'test'
        df_manifest.loc[train_idx, 'split'] = 'train'
        
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved manifest to {MANIFEST_PATH}")
    print("Split distribution:")
    print(pd.crosstab(df_manifest['split'], df_manifest['class_name'], margins=True))

if __name__ == '__main__':
    main()
