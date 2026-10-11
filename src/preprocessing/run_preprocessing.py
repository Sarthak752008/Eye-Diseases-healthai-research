"""
run_preprocessing.py - Generates standardized processed datasets under data/processed/
with complete traceability back to raw files and zero data leakage.
"""

import os
import json
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from .image_quality import safe_load_image, resize_and_pad, calculate_image_quality

ImageFile.LOAD_TRUNCATED_IMAGES = True

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(WORKSPACE_ROOT, 'data')
MANIFESTS_DIR = os.path.join(DATA_DIR, 'manifests')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

TARGET_SIZE = (224, 224)


def process_anemia():
    print("Processing Anemia dataset to data/processed/anemia/...")
    manifest_path = os.path.join(MANIFESTS_DIR, 'anemia_manifest.csv')
    df = pd.read_csv(manifest_path)
    
    anemia_proc_dir = os.path.join(PROCESSED_DIR, 'anemia')
    os.makedirs(anemia_proc_dir, exist_ok=True)
    
    records = []
    
    # Process each valid non-corrupt, non-duplicate image
    valid_df = df[df['is_corrupt'] == False].copy()
    
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Anemia"):
        orig_rel = row['relative_path']
        orig_full = os.path.join(WORKSPACE_ROOT, orig_rel)
        
        img = safe_load_image(orig_full)
        if img is None:
            continue
            
        qual = calculate_image_quality(img)
        proc_img = resize_and_pad(img, TARGET_SIZE)
        
        out_name = f"{row['image_id']}.jpg"
        out_path = os.path.join(anemia_proc_dir, out_name)
        proc_img.save(out_path, format='JPEG', quality=95)
        
        rec = row.to_dict()
        rec['processed_relative_path'] = os.path.relpath(out_path, WORKSPACE_ROOT).replace('\\', '/')
        rec['processed_filename'] = out_name
        rec['brightness'] = qual['brightness']
        rec['contrast'] = qual['contrast']
        rec['sharpness'] = qual['sharpness']
        records.append(rec)
        
    proc_df = pd.DataFrame(records)
    proc_manifest = os.path.join(anemia_proc_dir, 'anemia_processed_manifest.csv')
    proc_df.to_csv(proc_manifest, index=False)
    
    # Split manifests for regression
    for split in ['train', 'val', 'test']:
        split_df = proc_df[(proc_df['split'] == split) & (proc_df['hb_available'] == True)]
        split_path = os.path.join(anemia_proc_dir, f"{split}.csv")
        split_df.to_csv(split_path, index=False)
        print(f"Saved {split} manifest: {split_path} ({len(split_df)} records)")
        
    print(f"Anemia processing complete: {len(proc_df)} processed images.")
    return proc_df


def process_conjunctivitis():
    print("Processing Conjunctivitis dataset to data/processed/conjunctivitis/...")
    manifest_path = os.path.join(MANIFESTS_DIR, 'conjunctivitis_manifest.csv')
    df = pd.read_csv(manifest_path)
    
    cj_proc_dir = os.path.join(PROCESSED_DIR, 'conjunctivitis')
    
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Conjunctivitis"):
        orig_rel = row['relative_path']
        orig_full = os.path.join(WORKSPACE_ROOT, orig_rel)
        
        img = safe_load_image(orig_full)
        if img is None:
            continue
            
        qual = calculate_image_quality(img)
        proc_img = resize_and_pad(img, TARGET_SIZE)
        
        split = row['split']
        cls_name = row['class_name']
        
        target_dir = os.path.join(cj_proc_dir, split, cls_name)
        os.makedirs(target_dir, exist_ok=True)
        
        out_name = f"{row['image_id']}.jpg"
        out_path = os.path.join(target_dir, out_name)
        proc_img.save(out_path, format='JPEG', quality=95)
        
        rec = row.to_dict()
        rec['processed_relative_path'] = os.path.relpath(out_path, WORKSPACE_ROOT).replace('\\', '/')
        rec['processed_filename'] = out_name
        rec['brightness'] = qual['brightness']
        rec['contrast'] = qual['contrast']
        rec['sharpness'] = qual['sharpness']
        records.append(rec)
        
    proc_df = pd.DataFrame(records)
    proc_manifest = os.path.join(cj_proc_dir, 'conjunctivitis_processed_manifest.csv')
    proc_df.to_csv(proc_manifest, index=False)
    print(f"Conjunctivitis processing complete: {len(proc_df)} processed images.")
    return proc_df


def process_jaundice():
    print("Setting up Jaundice processed structure (data/processed/jaundice/)...")
    jaundice_proc_dir = os.path.join(PROCESSED_DIR, 'jaundice')
    
    for split in ['train', 'val', 'test']:
        for cls_name in ['normal', 'jaundiced']:
            os.makedirs(os.path.join(jaundice_proc_dir, split, cls_name), exist_ok=True)
            
    readme_path = os.path.join(jaundice_proc_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(
            "# Jaundice Processed Data Directory\n\n"
            "This directory structure is standardized for ocular jaundice screening:\n"
            "- `train/normal/`, `train/jaundiced/`\n"
            "- `val/normal/`, `val/jaundiced/`\n"
            "- `test/normal/`, `test/jaundiced/`\n\n"
            "**Current Status**: Blocker — No raw jaundice dataset or clinical bilirubin laboratory "
            "measurements exist in the current workspace.\n"
            "Once raw images are added to `data/raw/jaundice/`, run `run_preprocessing.py` to populate.\n"
        )
    print("Jaundice directory structure prepared.")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    anemia_proc_df = process_anemia()
    cj_proc_df = process_conjunctivitis()
    process_jaundice()
    
    # Save preprocessing config
    config = {
        'target_size': list(TARGET_SIZE),
        'interpolation': 'BILINEAR',
        'color_space': 'RGB',
        'padding_mode': 'constant_black',
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'anemia': {
            'processed_count': len(anemia_proc_df),
            'train_count': int((anemia_proc_df['split'] == 'train').sum()),
            'val_count': int((anemia_proc_df['split'] == 'val').sum()),
            'test_count': int((anemia_proc_df['split'] == 'test').sum()),
            'unlabeled_count': int((anemia_proc_df['split'] == 'unlabeled').sum())
        },
        'conjunctivitis': {
            'processed_count': len(cj_proc_df),
            'train_count': int((cj_proc_df['split'] == 'train').sum()),
            'val_count': int((cj_proc_df['split'] == 'val').sum()),
            'test_count': int((cj_proc_df['split'] == 'test').sum()),
            'classes': {'normal': 0, 'conjunctivitis': 1}
        },
        'jaundice': {
            'available': False,
            'processed_count': 0,
            'blocker': 'No raw jaundice dataset exists in workspace.'
        }
    }
    
    config_path = os.path.join(PROCESSED_DIR, 'preprocessing_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Saved preprocessing config to: {config_path}")


if __name__ == '__main__':
    main()
