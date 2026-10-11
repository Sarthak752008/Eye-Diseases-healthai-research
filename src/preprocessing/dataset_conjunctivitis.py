"""
dataset_conjunctivitis.py - PyTorch Dataset & Dataloaders for Conjunctivitis Classification.
Stratified Normal vs Conjunctivitis splits with color preservation.
"""

import os
from typing import Optional, Callable, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .image_quality import safe_load_image, resize_and_pad


def get_conjunctivitis_transforms(is_train: bool = True, target_size: tuple = (224, 224)):
    """
    Transforms for conjunctivitis classification.
    Avoids aggressive color transformations that alter erythema / redness.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Subtle brightness/contrast only; NO color solarization or hue inversion
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ConjunctivitisDataset(Dataset):
    """
    PyTorch Dataset for Conjunctivitis binary classification:
      0: Normal
      1: Conjunctivitis
    """
    def __init__(
        self,
        df: pd.DataFrame,
        workspace_root: str,
        transform: Optional[Callable] = None,
        target_size: tuple = (224, 224),
        use_processed_dir: Optional[str] = None
    ):
        self.workspace_root = workspace_root
        self.target_size = target_size
        self.transform = transform
        self.use_processed_dir = use_processed_dir
        
        data = df.copy()
        if 'is_corrupt' in data.columns:
            data = data[data['is_corrupt'] == False]
        if 'is_duplicate' in data.columns:
            data = data[data['is_duplicate'] == False]
            
        self.records = data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records.iloc[idx]
        
        if self.use_processed_dir:
            cls_name = row['class_name']
            split = row['split']
            img_path = os.path.join(self.use_processed_dir, split, cls_name, f"{row['image_id']}.jpg")
        else:
            rel_p = row['relative_path']
            img_path = os.path.join(self.workspace_root, rel_p)
            
        img = safe_load_image(img_path)
        if img is None:
            img = Image.new('RGB', self.target_size, (0, 0, 0))
        else:
            img = resize_and_pad(img, self.target_size)
            
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        label = int(row['class_label'])
        
        return {
            'image': img_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': row['image_id'],
            'class_name': row['class_name']
        }


def build_conjunctivitis_dataloaders(
    manifest_csv: str,
    workspace_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    target_size: tuple = (224, 224),
    use_processed_dir: Optional[str] = None
) -> Dict[str, DataLoader]:
    """Builds stratified DataLoaders for Conjunctivitis classification."""
    df = pd.read_csv(manifest_csv)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    train_ds = ConjunctivitisDataset(
        train_df, workspace_root,
        transform=get_conjunctivitis_transforms(is_train=True, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    val_ds = ConjunctivitisDataset(
        val_df, workspace_root,
        transform=get_conjunctivitis_transforms(is_train=False, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    test_ds = ConjunctivitisDataset(
        test_df, workspace_root,
        transform=get_conjunctivitis_transforms(is_train=False, target_size=target_size),
        target_size=target_size, use_processed_dir=use_processed_dir
    )
    
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'datasets': {'train': train_ds, 'val': val_ds, 'test': test_ds}
    }
