"""
src/models/stye_model.py
------------------------
MobileNetV3-Small Deep Learning Architecture for Stye / Hordeolum Detection
from Anterior Ocular / Eyelid Surface Photography.

Classes (Binary Screening):
  Class 0: Normal (Healthy anterior ocular surface and eyelid margin)
  Class 1: Stye / Hordeolum (Acute focal inflammatory eyelid nodule)

Backbone: Pretrained MobileNetV3-Small.
Classifier Head: 1024 -> 256 -> Hardswish -> Dropout(0.2) -> 2.
"""

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class StyeClassifier(nn.Module):
    """
    MobileNetV3-Small Classifier for Stye / Hordeolum Detection:
      Class 0: Normal
      Class 1: Stye / Hordeolum
    Preserves chromatic erythema features without aggressive color distortion.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
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
        """Returns unnormalized logits [batch_size, 2]."""
        return self.backbone(x)


def save_stye_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    class_mapping: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Saves classifier weights, optimizer state, epoch, and validation metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics or {},
        'class_mapping': class_mapping or {'0': 'normal', '1': 'stye_hordeolum'},
        'config': config or {}
    }
    torch.save(state, save_path)
    print(f"Saved Stye model checkpoint: {save_path} (Val Acc: {metrics.get('accuracy', 'N/A') if metrics else 'N/A'})")


def load_stye_checkpoint(checkpoint_path: str, device: str = 'cpu') -> StyeClassifier:
    """Loads a saved checkpoint into a StyeClassifier instance."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = len(checkpoint.get('class_mapping', {'0': 'normal', '1': 'stye_hordeolum'}))
    model = StyeClassifier(pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
