"""
jaundice_model.py - MobileNetV3-Small Model Pipeline for Ocular Jaundice (Scleral Icterus).
Maintains strict color-space fidelity for bilirubin yellowing.
Documents dataset availability status and blocker.
"""

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class JaundiceClassifier(nn.Module):
    """
    MobileNetV3-Small Classifier for Jaundice (Scleral Icterus) detection:
      Class 0: Normal
      Class 1: Visibly Jaundiced
    Preserves chromatic yellow signals without aggressive color alteration.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        in_features = self.backbone.classifier[0].in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class JaundiceRegressor(nn.Module):
    """
    Alternative architecture when laboratory total serum bilirubin (mg/dL) is available.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def save_jaundice_checkpoint(
    model: nn.Module,
    save_path: str,
    class_mapping: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'class_mapping': class_mapping or {'0': 'normal', '1': 'jaundiced'},
        'config': config or {}
    }
    torch.save(state, save_path)
    print(f"Saved Jaundice model checkpoint: {save_path}")


def load_jaundice_checkpoint(checkpoint_path: str, device: str = 'cpu') -> JaundiceClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = JaundiceClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
