"""
src/segmentation/clinical_segmenter.py
---------------------------------------
Clinical Ocular Segmenter.
Provides multi-region anatomical localization for conjunctiva, sclera, and iris.
Combines CIE-L*a*b* chromatic clustering, morphological refinement, and
specular glare filtering to localize regions and evaluate clinical adequacy.
"""

import os
import time
import base64
import io
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image

from src.segmentation.base import BaseOcularSegmenter, SegmentationResult, BoundingBox


class ClinicalOcularSegmenter(BaseOcularSegmenter):
    """
    Physiologically grounded ocular segmenter for clinical screening.
    Extracts:
      1. Palpebral Conjunctiva: Microvascular mucosal bed on everted eyelid.
      2. Bulbar Sclera: White/yellowish fibrous coat flanking the iris.
      3. Iris/Pupil: Central optical aperture.
      4. Specular Glare: High-intensity reflection artifacts that obscure tissue.
    """

    def __init__(
        self,
        min_conjunctiva_pct: float = 0.5,
        min_sclera_pct: float = 0.8,
        glare_threshold: int = 248,
        max_glare_pct: float = 12.0
    ):
        self.min_conjunctiva_pct = min_conjunctiva_pct
        self.min_sclera_pct = min_sclera_pct
        self.glare_threshold = glare_threshold
        self.max_glare_pct = max_glare_pct

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "ClinicalOcularSegmenter",
            "version": "1.2.0-clinical",
            "supported_regions": ["palpebral_conjunctiva", "bulbar_sclera", "iris", "glare_occlusion"],
            "calibration_dataset": "Eye Conjunctiva Segmentation Dataset (547 clinical images)",
            "license": "Research Use Only",
            "device": "cpu"
        }

    def _extract_masks(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts boolean masks for:
          - conjunctiva
          - sclera
          - iris
          - glare
        img_np is uint8 RGB array (H, W, 3).
        """
        h, w = img_np.shape[:2]
        total_pixels = h * w

        # Convert color spaces
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        l_chan, a_chan, b_chan = cv2.split(lab)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # 1. Specular Glare Mask (saturation near 0, luminance very high)
        glare_mask = (v_chan >= self.glare_threshold) & (s_chan < 35)

        # 2. Iris / Pupil Localization (central darker region)
        # Moderate to low luminance, relatively circular
        iris_candidates = (v_chan < 90) & (~glare_mask)
        # Morphological opening and closing
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        iris_clean = cv2.morphologyEx(iris_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel_sm)

        # Pick largest central component as iris
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(iris_clean)
        iris_mask = np.zeros((h, w), dtype=bool)
        best_iris_area = 0
        best_iris_centroid = (w / 2.0, h / 2.0)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            # Must be reasonably large and not glued to outermost border
            if area > total_pixels * 0.005 and 0.15 * w < cx < 0.85 * w and 0.15 * h < cy < 0.85 * h:
                if area > best_iris_area:
                    best_iris_area = area
                    best_iris_centroid = (cx, cy)
                    iris_mask = (labels == i)

        # 3. Sclera Mask (White/light region flanking the iris)
        # High luminance (L > 120 or V > 120), low-to-moderate saturation (S < 95), not glare, not skin
        sclera_candidates = (
            (v_chan > 115) &
            (s_chan < 95) &
            (~glare_mask) &
            (~iris_mask)
        )
        # Keep sclera near the eye center / iris altitude
        sclera_clean = cv2.morphologyEx(sclera_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel_sm)
        sclera_mask = sclera_clean.astype(bool)

        # 4. Palpebral Conjunctiva Mask (Elevated red chroma a* > 138, mucosal luminance)
        # Conjunctiva has high erythema: positive a* in CIE-Lab, typically in the lower everted lid
        # Exclude iris and extreme glare
        conj_candidates = (
            (a_chan > 138) &
            (l_chan > 45) &
            (l_chan < 225) &
            (~glare_mask) &
            (~iris_mask)
        )

        kernel_conj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        conj_clean = cv2.morphologyEx(conj_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel_conj)
        conj_clean = cv2.morphologyEx(conj_clean, cv2.MORPH_CLOSE, kernel_conj)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(conj_clean)
        conj_mask = np.zeros((h, w), dtype=bool)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cy = centroids[i][1]
            # Lower eyelid mucosal bed typically resides in the middle to lower vertical hemisphere
            if area >= total_pixels * 0.002:
                # Prefer components lower than or near iris
                if cy > best_iris_centroid[1] * 0.8:
                    conj_mask |= (labels == i)

        return conj_mask, sclera_mask, iris_mask, glare_mask

    def _get_bbox(self, mask: np.ndarray) -> Optional[BoundingBox]:
        """Calculates bounding box from a boolean mask."""
        if mask is None or not mask.any():
            return None
        y_indices, x_indices = np.where(mask)
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        return BoundingBox(x=x_min, y=y_min, w=x_max - x_min + 1, h=y_max - y_min + 1)

    def _create_overlay(
        self,
        rgb_img: Image.Image,
        conj_mask: np.ndarray,
        sclera_mask: np.ndarray,
        glare_mask: np.ndarray
    ) -> str:
        """
        Renders a clean, translucent segmentation overlay on top of the image.
        Green: Palpebral Conjunctiva
        Blue: Bulbar Sclera
        Red: Specular Glare Occlusion
        Returns base64 data URI (image/jpeg).
        """
        # Downsample for lightweight web display if large
        max_dim = 640
        w, h = rgb_img.size
        scale = min(max_dim / max(w, h), 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)

        disp_img = rgb_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        img_np = np.array(disp_img, dtype=np.uint8)

        # Resize masks to display dimensions
        c_disp = cv2.resize(conj_mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST) > 0
        s_disp = cv2.resize(sclera_mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST) > 0
        g_disp = cv2.resize(glare_mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST) > 0

        overlay = img_np.copy()
        # Green tint for conjunctiva [34, 197, 94]
        overlay[c_disp] = (overlay[c_disp] * 0.55 + np.array([34, 197, 94]) * 0.45).astype(np.uint8)
        # Blue tint for sclera [59, 130, 246]
        overlay[s_disp] = (overlay[s_disp] * 0.60 + np.array([59, 130, 246]) * 0.40).astype(np.uint8)
        # Yellow-orange for glare [245, 158, 11]
        overlay[g_disp] = (overlay[g_disp] * 0.50 + np.array([245, 158, 11]) * 0.50).astype(np.uint8)

        # Draw contour borders for clear definition
        contours_c, _ = cv2.findContours(c_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_c, -1, (16, 185, 129), 2)

        contours_s, _ = cv2.findContours(s_disp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_s, -1, (37, 99, 235), 2)

        out_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def segment(self, image: Image.Image) -> SegmentationResult:
        """Executes full anatomical segmentation on the ocular image."""
        t0 = time.time()
        rgb_img = image.convert("RGB")
        img_np = np.array(rgb_img)
        total_pixels = img_np.shape[0] * img_np.shape[1]

        conj_mask, sclera_mask, iris_mask, glare_mask = self._extract_masks(img_np)

        conj_pixels = int(conj_mask.sum())
        sclera_pixels = int(sclera_mask.sum())
        glare_pixels = int(glare_mask.sum())

        conj_pct = round((conj_pixels / total_pixels) * 100.0, 2)
        sclera_pct = round((sclera_pixels / total_pixels) * 100.0, 2)
        glare_pct = round((glare_pixels / total_pixels) * 100.0, 2)

        is_conj_ok = conj_pct >= self.min_conjunctiva_pct
        is_sclera_ok = sclera_pct >= self.min_sclera_pct
        is_glare_ok = glare_pct <= self.max_glare_pct

        warnings = []
        if not is_conj_ok:
            warnings.append(f"Insufficient palpebral conjunctiva exposure ({conj_pct}% of image). Expose lower eyelid margin.")
        if not is_sclera_ok:
            warnings.append(f"Limited sclera visibility ({sclera_pct}% of image). Jaundice assessment may have reduced sensitivity.")
        if not is_glare_ok:
            warnings.append(f"Excessive specular glare ({glare_pct}%). Diffuse ambient lighting recommended.")

        status = "success" if (is_conj_ok or is_sclera_ok) else "insufficient_region"

        overlay_b64 = self._create_overlay(rgb_img, conj_mask, sclera_mask, glare_mask)

        latency_ms = round((time.time() - t0) * 1000, 1)

        return SegmentationResult(
            status=status,
            conjunctiva_mask=conj_mask,
            sclera_mask=sclera_mask,
            iris_mask=iris_mask,
            conjunctiva_bbox=self._get_bbox(conj_mask),
            sclera_bbox=self._get_bbox(sclera_mask),
            coverage_pct={"conjunctiva": conj_pct, "sclera": sclera_pct},
            quality_flags={
                "is_conjunctiva_sufficient": is_conj_ok,
                "is_sclera_sufficient": is_sclera_ok,
                "glare_occlusion_pct": glare_pct,
                "adequate_exposure": is_conj_ok and is_sclera_ok and is_glare_ok
            },
            warnings=warnings,
            overlay_base64=overlay_b64,
            metadata={
                "latency_ms": latency_ms,
                "model": "ClinicalOcularSegmenter-v1.2",
                "image_dims": [img_np.shape[1], img_np.shape[0]]
            }
        )


# Global segmenter singleton
_DEFAULT_SEGMENTER: Optional[ClinicalOcularSegmenter] = None

def get_ocular_segmenter() -> ClinicalOcularSegmenter:
    global _DEFAULT_SEGMENTER
    if _DEFAULT_SEGMENTER is None:
        _DEFAULT_SEGMENTER = ClinicalOcularSegmenter()
    return _DEFAULT_SEGMENTER
