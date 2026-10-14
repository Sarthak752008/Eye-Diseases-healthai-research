"""
src/segmentation/base.py
-------------------------
Model-independent interface and structured data schemas for ocular region segmentation.
Preserves anatomical landmarks, chromatic features, and clinical metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> Dict[str, int]:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}


@dataclass
class SegmentationResult:
    """
    Standardized result for ocular region segmentation.
    Contains binary masks, bounding boxes, clinical quality flags, and overlays.
    """
    status: str  # "success", "insufficient_region", "error"
    conjunctiva_mask: Optional[np.ndarray] = None  # 2D boolean array (H, W)
    sclera_mask: Optional[np.ndarray] = None        # 2D boolean array (H, W)
    iris_mask: Optional[np.ndarray] = None          # 2D boolean array (H, W)
    conjunctiva_bbox: Optional[BoundingBox] = None
    sclera_bbox: Optional[BoundingBox] = None
    coverage_pct: Dict[str, float] = field(default_factory=lambda: {"conjunctiva": 0.0, "sclera": 0.0})
    quality_flags: Dict[str, Any] = field(default_factory=lambda: {
        "is_conjunctiva_sufficient": False,
        "is_sclera_sufficient": False,
        "glare_occlusion_pct": 0.0,
        "adequate_exposure": True
    })
    warnings: List[str] = field(default_factory=list)
    overlay_base64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_masks: bool = False) -> Dict[str, Any]:
        """Serializes result to JSON-friendly dictionary."""
        d = {
            "status": self.status,
            "coverage_pct": self.coverage_pct,
            "quality_flags": self.quality_flags,
            "warnings": self.warnings,
            "conjunctiva_bbox": self.conjunctiva_bbox.to_dict() if self.conjunctiva_bbox else None,
            "sclera_bbox": self.sclera_bbox.to_dict() if self.sclera_bbox else None,
            "overlay_base64": self.overlay_base64,
            "metadata": self.metadata
        }
        if include_masks:
            d["conjunctiva_pixels"] = int(self.conjunctiva_mask.sum()) if self.conjunctiva_mask is not None else 0
            d["sclera_pixels"] = int(self.sclera_mask.sum()) if self.sclera_mask is not None else 0
        return d


class BaseOcularSegmenter(ABC):
    """
    Abstract interface for ocular region segmenters.
    Allows swappable backbones (heuristic, U-Net, DeepLab, SAM) without altering pipeline callers.
    """

    @abstractmethod
    def segment(self, image: Image.Image) -> SegmentationResult:
        """
        Segments ocular regions (conjunctiva, sclera, iris) in the input image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            SegmentationResult with masks, bboxes, metrics, and quality flags.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model name, version, supported regions, and license/citation info."""
        pass
