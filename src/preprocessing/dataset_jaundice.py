"""
dataset_jaundice.py - Reusable PyTorch Dataset & Dataloaders for Jaundice (Scleral Icterus).
Strictly preserves scleral yellowing (no hue shifting).
Documents data availability status and handles missing workspace data cleanly.
"""

import os
from typing import Optional, Callable, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .image_quality import safe_load_image, resize_and_pad


def get_jaundice_transforms(is_train: bool = True, target_size: tuple = (224, 224)):
    """
    Transforms for jaundice detection / scleral icterus.
    CRITICAL: Preserves yellow chromatic signals. NO hue or color-inversion transformations.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Extremely mild brightness/contrast; zero hue perturbation
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.0, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class JaundiceDataset(Dataset):
    """
    PyTorch Dataset for Jaundice classification (0: Normal, 1: Jaundiced)
    or numerical laboratory bilirubin regression.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        workspace_root: str,
        transform: Optional[Callable] = None,
        target_size: tuple = (224, 224),
        task: str = 'classification'  # 'classification' or 'regression'
    ):
        self.workspace_root = workspace_root
        self.transform = transform
        self.target_size = target_size
        self.task = task
        self.records = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.records) == 0:
            raise IndexError("JaundiceDataset is empty because no jaundice dataset exists in workspace.")
            
        row = self.records.iloc[idx]
        img_path = os.path.join(self.workspace_root, row['relative_path'])
        img = safe_load_image(img_path)
        if img is None:
            img = Image.new('RGB', self.target_size, (0, 0, 0))
        else:
            img = resize_and_pad(img, self.target_size)
            
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        if self.task == 'regression':
            target = torch.tensor(float(row['bilirubin_value']), dtype=torch.float32)
        else:
            target = torch.tensor(int(row['class_label']), dtype=torch.long)
            
        return {
            'image': img_tensor,
            'target': target,
            'image_id': row.get('image_id', f'jaundice_{idx}')
        }


def build_jaundice_dataloaders(
    manifest_csv: str,
    workspace_root: str,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Attempts to build Jaundice DataLoaders.
    Raises RuntimeError if manifest has 0 records (due to missing workspace data).
    """
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Jaundice manifest not found at: {manifest_csv}")
        
    df = pd.read_csv(manifest_csv)
    if len(df) == 0:
        raise RuntimeError(
            "Jaundice dataset contains 0 verified records. "
            "Cannot construct DataLoaders: no jaundice or bilirubin dataset exists in the workspace. "
            "Pipeline is ready and scaffolded for when data is provided."
        )
        
    # Standard splitting logic if records are present
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    train_ds = JaundiceDataset(train_df, workspace_root, transform=get_jaundice_transforms(is_train=True))
    val_ds = JaundiceDataset(val_df, workspace_root, transform=get_jaundice_transforms(is_train=False))
    test_ds = JaundiceDataset(test_df, workspace_root, transform=get_jaundice_transforms(is_train=False))
    
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }
