"""
dataset_anemia.py - PyTorch Dataset & Dataloaders for Anemia Hb Regression.
Strict subject-level partitioning and clinical color preservation.
"""

import os
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .image_quality import safe_load_image, resize_and_pad


def get_anemia_transforms(is_train: bool = True, target_size: tuple = (224, 224)):
    """
    Returns clinical transforms. Training augmentation uses subtle spatial shifts
    and strictly forbids aggressive color/hue shifts that distort conjunctival pallor.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            # Extremely mild brightness/contrast jitter; NO hue shift to preserve erythema
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class AnemiaDataset(Dataset):
    """
    PyTorch Dataset for Anemia Hemoglobin regression.
    Pairs ocular/conjunctiva images with laboratory-verified Hb measurements (g/dL).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        workspace_root: str,
        view_types: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_size: tuple = (224, 224),
        use_processed_dir: Optional[str] = None
    ):
        self.workspace_root = workspace_root
        self.target_size = target_size
        self.transform = transform
        self.use_processed_dir = use_processed_dir
        
        # Filter for valid Hb targets and not corrupt
        data = df[df['hb_available'] == True].copy()
        if 'is_corrupt' in data.columns:
            data = data[data['is_corrupt'] == False]
        if 'is_duplicate' in data.columns:
            # Exclude duplicate files
            data = data[data['is_duplicate'] == False]
            
        if view_types:
            data = data[data['view_type'].isin(view_types)]
            
        self.records = data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        
        if self.use_processed_dir:
            img_path = os.path.join(self.use_processed_dir, f"{row['image_id']}.jpg")
        else:
            rel_p = row['relative_path']
            img_path = os.path.join(self.workspace_root, rel_p)
            
        img = safe_load_image(img_path)
        if img is None:
            # Fallback black image if loading failed
            img = Image.new('RGB', self.target_size, (0, 0, 0))
        else:
            img = resize_and_pad(img, self.target_size)
            
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        hb_val = float(row['hb_value'])
        
        return {
            'image': img_tensor,
            'hb': torch.tensor(hb_val, dtype=torch.float32),
            'image_id': row['image_id'],
            'subject_id': row['subject_id'],
            'view_type': row['view_type'],
            'source': row['dataset_source']
        }


def build_anemia_dataloaders(
    manifest_csv: str,
    workspace_root: str,
    batch_size: int = 16,
    view_types: Optional[List[str]] = None,
    num_workers: int = 0,
    target_size: tuple = (224, 224),
    use_processed_dir: Optional[str] = None
) -> Dict[str, DataLoader]:
    """Builds subject-stratified DataLoaders for Anemia Hb regression."""
    df = pd.read_csv(manifest_csv)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    train_ds = AnemiaDataset(
        train_df, workspace_root, view_types=view_types,
        transform=get_anemia_transforms(is_train=True, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    val_ds = AnemiaDataset(
        val_df, workspace_root, view_types=view_types,
        transform=get_anemia_transforms(is_train=False, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    test_ds = AnemiaDataset(
        test_df, workspace_root, view_types=view_types,
        transform=get_anemia_transforms(is_train=False, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'datasets': {'train': train_ds, 'val': val_ds, 'test': test_ds}
    }
