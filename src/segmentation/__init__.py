"""
src/segmentation package
------------------------
Ocular region segmentation, boundary detection, and anatomical adequacy checking.
"""

from src.segmentation.base import BaseOcularSegmenter, SegmentationResult, BoundingBox
from src.segmentation.clinical_segmenter import ClinicalOcularSegmenter, get_ocular_segmenter
from src.segmentation.evaluator import evaluate_segmentation_benchmark

__all__ = [
    "BaseOcularSegmenter",
    "SegmentationResult",
    "BoundingBox",
    "ClinicalOcularSegmenter",
    "get_ocular_segmenter",
    "evaluate_segmentation_benchmark"
]
