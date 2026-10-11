"""
anemia_model.py - ResNet50 Regressor for Non-Invasive Hemoglobin (Hb) Estimation.
Predicts numerical laboratory hemoglobin in g/dL from ocular images.
"""

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class AnemiaRegressor(nn.Module):
    """
    ResNet-50 CNN Regressor for non-invasive ocular hemoglobin (Hb) estimation.
    Backbone: Pretrained ResNet-50.
    Head: 2048 -> 256 -> Dropout -> 1.
    Target: Numerical laboratory Hb value (g/dL).
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # 2048
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: [batch_size, 1] -> squeeze to [batch_size]
        return self.backbone(x).squeeze(-1)



class AnemiaMobileNetRegressor(nn.Module):
    """
    MobileNetV3-Small Regressor for non-invasive ocular hemoglobin (Hb) estimation.
    Backbone: Pretrained MobileNetV3-Small.
    Head: 1024 -> 256 -> Hardswish -> Dropout -> 1.
    Target: Numerical laboratory Hb value (g/dL).
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def save_anemia_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    config: Optional[Dict[str, Any]] = None
):
    """Saves model weights, optimizer state, epoch, and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config or {}
    }
    torch.save(state, save_path)
    print(f"Saved Anemia model checkpoint: {save_path} (Val MAE: {metrics.get('mae_g_dl', 'N/A')})")


def load_anemia_checkpoint(checkpoint_path: str, device: str = 'cpu') -> AnemiaRegressor:
    """Loads a saved checkpoint into an AnemiaRegressor instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AnemiaRegressor(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_anemia_mobilenet_checkpoint(checkpoint_path: str, device: str = 'cpu') -> AnemiaMobileNetRegressor:
    """Loads a saved checkpoint into an AnemiaMobileNetRegressor instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AnemiaMobileNetRegressor(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

