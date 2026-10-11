"""
conjunctivitis_model.py - MobileNetV3-Small Binary Classifier for Conjunctivitis Screening.
Differentiates Normal ocular surface vs. Conjunctivitis.
"""

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class ConjunctivitisClassifier(nn.Module):
    """
    MobileNetV3-Small Classifier for binary Conjunctivitis detection:
      Class 0: Normal
      Class 1: Conjunctivitis
    Backbone: Pretrained MobileNetV3-Small.
    Head: 1024 -> 256 -> Hardswish -> Dropout -> 2.
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
        # Returns raw unnormalized logits [batch_size, 2]
        return self.backbone(x)


def save_conjunctivitis_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    save_path: str,
    class_mapping: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Saves classifier weights, optimizer state, epoch, and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'class_mapping': class_mapping or {'0': 'normal', '1': 'conjunctivitis'},
        'config': config or {}
    }
    torch.save(state, save_path)
    print(f"Saved Conjunctivitis model checkpoint: {save_path} (Val Acc: {metrics.get('accuracy', 'N/A')})")


def load_conjunctivitis_checkpoint(checkpoint_path: str, device: str = 'cpu') -> ConjunctivitisClassifier:
    """Loads a saved checkpoint into a ConjunctivitisClassifier instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConjunctivitisClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
