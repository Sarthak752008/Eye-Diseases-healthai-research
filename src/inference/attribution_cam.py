"""
src/inference/attribution_cam.py
---------------------------------
Model-Specific AI Detection Visualization using genuine Grad-CAM.
Generates attribution heatmaps and transparent overlays directly from PyTorch
feature representations for:
  - Model A (Anemia / Hemoglobin ResNet-50 Regressor)
  - Model B (SLID Conjunctival Hyperemia MobileNetV3-Small Classifier)
  - Model C (Visible Scleral Icterus MobileNetV3-Small Classifier)

Strict Medical & Research Transparency:
1. Heatmaps represent feature-level gradients that influenced the model prediction.
2. They are NOT certified disease boundaries or diagnostic confirmations.
3. Every visualization is strictly generated from the real model and final model input.
"""

import io
import base64
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("eyehealthai.attribution")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def pil_to_base64_jpeg(img: Image.Image, quality: int = 88) -> str:
    """Converts a PIL Image to a base64 JPEG data URI."""
    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Error encoding image to base64: {e}")
        return ""


def apply_colormap_to_heatmap(heatmap: np.ndarray, colormap: int = None) -> np.ndarray:
    """
    Applies an interpretable colormap (Turbo or Jet) to a 2D float heatmap in [0, 1].
    Returns an RGB uint8 array (H, W, 3).
    """
    if colormap is None and CV2_AVAILABLE:
        # Prefer COLORMAP_TURBO if available (perceptually uniform), fallback to COLORMAP_JET
        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    if CV2_AVAILABLE:
        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return colored_rgb
    else:
        # Simple manual fallback colormap if cv2 is unavailable
        r = heatmap_uint8
        g = np.clip(255 - np.abs(heatmap_uint8 - 128) * 2, 0, 255).astype(np.uint8)
        b = 255 - heatmap_uint8
        return np.stack([r, g, b], axis=-1)


def blend_overlay(
    background_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.50
) -> np.ndarray:
    """
    Blends an anatomical RGB background with a colored heatmap.
    Preserves structural features of the underlying eye tissue.
    """
    bg = background_rgb.astype(np.float32)
    hm = heatmap_rgb.astype(np.float32)
    blended = np.clip(bg * (1.0 - alpha) + hm * alpha, 0, 255).astype(np.uint8)
    return blended


class GradCAM:
    """
    Computes genuine Grad-CAM attributions for a given target layer in PyTorch models.
    Supports both classification logits and continuous regression scalar outputs.
    """
    def __init__(self, model: "nn.Module", target_layer: "nn.Module", is_regression: bool = False):
        self.model = model
        self.target_layer = target_layer
        self.is_regression = is_regression
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

    def _register_hooks(self):
        self._handles.clear()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h_fwd, h_bwd])

    def _remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def generate(
        self,
        input_tensor: "torch.Tensor",
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Computes Grad-CAM for input_tensor (shape: [1, 3, H, W]).
        Returns (heatmap_norm_2d [H, W], target_score_float).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Grad-CAM computation.")

        self._register_hooks()
        try:
            self.model.zero_grad(set_to_none=True)
            # Ensure input tensor requires gradient
            if not input_tensor.requires_grad:
                input_tensor = input_tensor.clone().detach().requires_grad_(True)

            output = self.model(input_tensor)

            if self.is_regression:
                # Continuous regression: Scalar prediction (e.g. Hb in g/dL)
                if output.dim() == 0:
                    target_score = output
                elif output.dim() == 1:
                    target_score = output[0]
                else:
                    target_score = output[0, 0]
                pred_val = float(target_score.item())
            else:
                # Classification: Select target class logit
                if target_class is None:
                    target_class = int(torch.argmax(output[0]).item())
                target_score = output[0, target_class]
                pred_val = float(target_score.item())

            # Backward pass to obtain layer gradients
            target_score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

            # Activations & Gradients shape: [1, C, H', W']
            acts = self.activations[0]    # [C, H', W']
            grads = self.gradients[0]     # [C, H', W']

            # Global average pooling of gradients along spatial dimensions to obtain weights alpha_k
            alpha_k = torch.mean(grads, dim=(1, 2), keepdim=True)  # [C, 1, 1]

            # Weighted sum of feature maps
            cam = torch.sum(alpha_k * acts, dim=0)  # [H', W']

            # Apply ReLU: We are interested in features with a positive influence on target
            cam = F.relu(cam)

            cam_np = cam.cpu().numpy()

            # Normalize to [0, 1]
            max_val = np.max(cam_np)
            min_val = np.min(cam_np)
            if max_val > min_val:
                heatmap = (cam_np - min_val) / (max_val - min_val)
            else:
                heatmap = np.zeros_like(cam_np)

            return heatmap, pred_val
        finally:
            self._remove_hooks()


def generate_model_attribution_package(
    model: "nn.Module",
    target_layer: "nn.Module",
    input_tensor: "torch.Tensor",
    final_input_pil: Image.Image,
    target_class: Optional[int] = None,
    is_regression: bool = False,
    model_name: str = "model",
    target_name: str = "target",
    alpha: float = 0.50,
    target_layer_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a full attribution package for UI rendering, reports, and PDFs:
      - 'status': 'available' or 'unavailable'
      - 'model_name': name of model architecture
      - 'target_name': clinical screening target
      - 'original_b64': unnormalized final model input (224x224 RGB)
      - 'heatmap_b64': standalone colormapped attention heatmap
      - 'overlay_b64': blended overlay of heatmap onto final model input
      - 'dimensions': [width, height]
      - 'disclaimer': mandatory medical attribution notice
    """
    orig_b64 = pil_to_base64_jpeg(final_input_pil)
    disclaimer_text = (
        "AI attention visualization: highlights image regions that influenced the model's output. "
        "It is not a verified disease boundary or medical diagnosis."
    )
    layer_name = target_layer_name or ("model.backbone.layer4[-1]" if "resnet" in model_name.lower() else "model.backbone.features[-1]")

    if not TORCH_AVAILABLE:
        return {
            "status": "unavailable",
            "model_name": model_name,
            "target_name": target_name,
            "target_layer": layer_name,
            "reason": "PyTorch runtime unavailable for gradient attribution.",
            "original_b64": orig_b64,
            "heatmap_b64": None,
            "overlay_b64": None,
            "dimensions": [final_input_pil.width, final_input_pil.height],
            "disclaimer": disclaimer_text
        }

    try:
        grad_cam = GradCAM(model, target_layer, is_regression=is_regression)
        heatmap_lowres, pred_val = grad_cam.generate(input_tensor, target_class=target_class)

        # Upsample heatmap to match final model input dimensions (typically 224x224)
        target_w, target_h = final_input_pil.size
        if CV2_AVAILABLE:
            heatmap_upsampled = cv2.resize(heatmap_lowres, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Fallback PIL resize
            hm_img = Image.fromarray((heatmap_lowres * 255).astype(np.uint8))
            hm_resized = hm_img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
            heatmap_upsampled = np.array(hm_resized).astype(np.float32) / 255.0

        heatmap_upsampled = np.clip(heatmap_upsampled, 0.0, 1.0)

        # Generate colored heatmap
        colored_heatmap_rgb = apply_colormap_to_heatmap(heatmap_upsampled)
        heatmap_pil = Image.fromarray(colored_heatmap_rgb)

        # Blend with final model input
        bg_rgb = np.array(final_input_pil.convert("RGB"))
        blended_rgb = blend_overlay(bg_rgb, colored_heatmap_rgb, alpha=alpha)
        overlay_pil = Image.fromarray(blended_rgb)

        hm_b64 = pil_to_base64_jpeg(heatmap_pil)
        ov_b64 = pil_to_base64_jpeg(overlay_pil)

        return {
            "status": "available",
            "model_name": model_name,
            "target_name": target_name,
            "target_layer": layer_name,
            "target_class": target_class,
            "is_regression": is_regression,
            "target_score": pred_val,
            "original_b64": orig_b64,
            "heatmap_b64": hm_b64,
            "overlay_b64": ov_b64,
            "input": orig_b64,
            "heatmap": hm_b64,
            "overlay": ov_b64,
            "dimensions": [target_w, target_h],
            "disclaimer": disclaimer_text
        }
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed for {model_name}: {e}")
        return {
            "status": "unavailable",
            "model_name": model_name,
            "target_name": target_name,
            "reason": f"Visualization computation failed: {str(e)}",
            "original_b64": orig_b64,
            "heatmap_b64": None,
            "overlay_b64": None,
            "dimensions": [final_input_pil.width, final_input_pil.height],
            "disclaimer": disclaimer_text
        }
