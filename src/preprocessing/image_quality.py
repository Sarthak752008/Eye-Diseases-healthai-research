"""
image_quality.py - Image loading, RGB conversion, and ocular image quality checks.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_load_image(filepath: str) -> Optional[Image.Image]:
    """Safely loads an image, converts to RGB, handling broken headers."""
    if not os.path.exists(filepath):
        return None
    try:
        with Image.open(filepath) as img:
            img.load()
            # If RGBA, blend onto neutral black or white background
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (0, 0, 0))
                rgb_img.paste(img, mask=img.split()[3])
                return rgb_img
            elif img.mode != 'RGB':
                return img.convert('RGB')
            else:
                return img.copy()
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def calculate_image_quality(img: Image.Image) -> Dict[str, float]:
    """
    Computes objective image quality indicators:
      - brightness (mean luminance [0, 255])
      - contrast (standard deviation of luminance)
      - sharpness (variance of Laplacian proxy via numpy 2D gradients)
    """
    gray = np.array(img.convert('L'), dtype=np.float32)
    
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # 2D discrete Laplacian filter approximation for sharpness
    # [ 0,  1,  0]
    # [ 1, -4,  1]
    # [ 0,  1,  0]
    padded = np.pad(gray, 1, mode='edge')
    laplacian = (
        padded[1:-1, :-2] + padded[1:-1, 2:] +
        padded[:-2, 1:-1] + padded[2:, 1:-1] -
        4.0 * gray
    )
    sharpness = float(np.var(laplacian))
    
    return {
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'sharpness': round(sharpness, 2),
        'is_too_dark': brightness < 30.0,
        'is_too_bright': brightness > 230.0,
        'is_blurry': sharpness < 20.0
    }


def resize_and_pad(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resizes image maintaining aspect ratio and pads to target_size,
    avoiding geometric distortion of the ocular structure.
    """
    img.thumbnail(target_size, Image.Resampling.BILINEAR)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - img.width) // 2
    paste_y = (target_size[1] - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def verify_ocular_framing_or_detect_face(
    img: Image.Image,
    view_type: str = "unspecified"
) -> Dict[str, Any]:
    """
    Validates anatomical view compatibility and checks for invalid full-face inputs
    where automated face detection is supported.
    """
    compatible_views = {'palpebral', 'forniceal', 'forniceal_palpebral', 'raw_eye', 'unspecified'}
    clean_view = (view_type or "unspecified").strip().lower()

    # 1. Anatomical view check
    if clean_view not in compatible_views:
        is_face_view = clean_view in {'face', 'full_face', 'portrait', 'selfie'}
        return {
            "supported": True,
            "is_valid": False,
            "is_full_face": is_face_view,
            "reason": (
                f"Unsupported anatomical view '{view_type}'. "
                f"Compatible views are: {sorted(list(compatible_views))}. "
                "A full-face portrait cannot be analyzed for microvascular conjunctival pallor."
                if is_face_view else
                f"Unsupported anatomical view '{view_type}'. Compatible views are: {sorted(list(compatible_views))}."
            )
        }

    # 2. Automated detector check if OpenCV/cascade available
    try:
        import cv2  # type: ignore
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            gray = np.array(img.convert('L'))
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            img_w, img_h = img.size
            # A full-face portrait occupies a substantial portion of the frame
            significant_faces = [
                (x, y, w, h) for (x, y, w, h) in faces
                if (w / img_w >= 0.25 or h / img_h >= 0.25 or (w * h) / (img_w * img_h) >= 0.06)
            ]
            if len(significant_faces) > 0:
                return {
                    "supported": True,
                    "is_valid": False,
                    "is_full_face": True,
                    "reason": "Full face detected. Macro eye framing required (10–15 cm close-up of ocular conjunctiva)."
                }
            return {
                "supported": True,
                "is_valid": True,
                "is_full_face": False,
                "reason": "No full face detected in frame."
            }
    except Exception:
        pass

    # Automated face detection unavailable in this environment
    return {
        "supported": False,
        "is_valid": True,
        "is_full_face": False,
        "reason": "Automated face detection not supported in environment; manual macro eye alignment required."
    }

