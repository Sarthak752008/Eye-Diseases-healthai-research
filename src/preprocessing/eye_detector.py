"""
src/preprocessing/eye_detector.py
---------------------------------
Automated ocular region detection and eye-focused cropping.
Uses OpenCV Haar Cascades to locate the eye in a macro close-up or face frame,
provides real-time positional guidance, and produces aspect-ratio-preserving ocular crops.

Clinical Safety Disclaimer:
Ocular detection confirms eye-region framing only; it does NOT confirm that the
inner palpebral conjunctiva or sclera is properly visible or everted.
Forceful or painful eyelid manipulation must never be encouraged.
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
    CASCADE_DIR = cv2.data.haarcascades
    EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')
    FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
    LEFT_EYE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_lefteye_2splits.xml')
    RIGHT_EYE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_righteye_2splits.xml')
except Exception:
    OPENCV_AVAILABLE = False
    EYE_CASCADE_PATH = ""
    FACE_CASCADE_PATH = ""
    LEFT_EYE_PATH = ""
    RIGHT_EYE_PATH = ""


def _load_cascade(path: str):
    if not OPENCV_AVAILABLE or not os.path.exists(path):
        return None
    try:
        cascade = cv2.CascadeClassifier(path)
        return cascade if not cascade.empty() else None
    except Exception:
        return None


# Module-level cached classifiers
_eye_cascade = None
_face_cascade = None
_left_eye_cascade = None
_right_eye_cascade = None


def get_eye_cascade():
    global _eye_cascade
    if _eye_cascade is None and OPENCV_AVAILABLE:
        _eye_cascade = _load_cascade(EYE_CASCADE_PATH)
    return _eye_cascade


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None and OPENCV_AVAILABLE:
        _face_cascade = _load_cascade(FACE_CASCADE_PATH)
    return _face_cascade


def get_left_eye_cascade():
    global _left_eye_cascade
    if _left_eye_cascade is None and OPENCV_AVAILABLE:
        _left_eye_cascade = _load_cascade(LEFT_EYE_PATH)
    return _left_eye_cascade


def get_right_eye_cascade():
    global _right_eye_cascade
    if _right_eye_cascade is None and OPENCV_AVAILABLE:
        _right_eye_cascade = _load_cascade(RIGHT_EYE_PATH)
    return _right_eye_cascade


def detect_ocular_region(
    image: Union[Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """
    Detects the ocular region in an input image and computes optimal crop boundaries
    and positioning guidance.

    Returns:
        Dict containing detection status, bounding boxes, crop boundaries,
        actionable guidance, and clinical disclaimers.
    """
    # Standardize image to PIL and numpy grayscale
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            gray = image
            pil_img = Image.fromarray(image)
        elif image.ndim == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray(image)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.array(pil_img.convert('L'))
        else:
            pil_img = Image.fromarray(image)
            gray = np.array(pil_img.convert('L'))
    elif isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
        gray = np.array(pil_img.convert('L'))
    else:
        return {
            "eye_detected": False,
            "eye_count": 0,
            "primary_eye_box": None,
            "crop_box": None,
            "aspect_ratio": 1.0,
            "is_macro_framing": False,
            "guidance": "Invalid image format.",
            "anatomical_disclaimer": "Automated eye detection failed due to invalid input."
        }

    img_w, img_h = pil_img.size

    if not OPENCV_AVAILABLE:
        # Fallback when OpenCV is not installed
        # Provide center crop for manual macro alignment
        cw = int(img_w * 0.6)
        ch = int(img_h * 0.6)
        cx = int((img_w - cw) / 2)
        cy = int((img_h - ch) / 2)
        return {
            "eye_detected": False,
            "eye_count": 0,
            "primary_eye_box": None,
            "crop_box": {"x": cx, "y": cy, "w": cw, "h": ch},
            "aspect_ratio": round(cw / ch, 3) if ch > 0 else 1.0,
            "is_macro_framing": True,
            "guidance": "Position one eye inside the manual reticle (10–15 cm close-up).",
            "anatomical_disclaimer": "Manual macro eye alignment fallback active."
        }

    eye_c = get_eye_cascade()
    face_c = get_face_cascade()

    # Step 1: Detect candidate eyes
    candidate_eyes = []
    if eye_c is not None:
        eyes = eye_c.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(min(img_w, img_h) * 0.08), int(min(img_w, img_h) * 0.08))
        )
        if len(eyes) > 0:
            candidate_eyes.extend(list(eyes))

    # Supplemental search with dedicated left/right cascades if general eye missed
    if len(candidate_eyes) == 0:
        le_c = get_left_eye_cascade()
        re_c = get_right_eye_cascade()
        if le_c is not None:
            l_eyes = le_c.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(l_eyes) > 0:
                candidate_eyes.extend(list(l_eyes))
        if re_c is not None:
            r_eyes = re_c.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(r_eyes) > 0:
                candidate_eyes.extend(list(r_eyes))

    # Step 2: Check for full-face presence to guide user
    full_face_present = False
    if face_c is not None:
        faces = face_c.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        for (fx, fy, fw, fh) in faces:
            if fw / img_w >= 0.25 or fh / img_h >= 0.25 or (fw * fh) / (img_w * img_h) >= 0.06:
                full_face_present = True
                break

    if len(candidate_eyes) == 0:
        # No eye detected
        if full_face_present:
            guidance = "Full face detected. Move closer — keep one eye inside the frame."
        else:
            guidance = "No eye detected. Bring one eye into the center frame (10–15 cm close-up)."

        # Provide a default center macro framing box for manual fallback
        cw = int(img_w * 0.65)
        ch = int(img_h * 0.55)
        cx = int((img_w - cw) / 2)
        cy = int((img_h - ch) / 2)

        return {
            "eye_detected": False,
            "eye_count": 0,
            "primary_eye_box": None,
            "crop_box": {"x": cx, "y": cy, "w": cw, "h": ch},
            "aspect_ratio": round(cw / ch, 3) if ch > 0 else 1.0,
            "is_macro_framing": False,
            "guidance": guidance,
            "anatomical_disclaimer": "No ocular region detected. Position eye closer to camera."
        }

    # Step 3: Select primary eye
    # In macro eye photography, the primary eye is the largest and closest to center
    def eye_score(box):
        bx, by, bw, bh = box
        area = bw * bh
        cx = bx + bw / 2
        cy = by + bh / 2
        dist_from_center = np.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
        # Higher score = larger area, closer to center
        return area / (dist_from_center + 50.0)

    best_eye = max(candidate_eyes, key=eye_score)
    ex, ey, ew, eh = [int(v) for v in best_eye]

    # Step 4: Expand eye bounding box to include surrounding sclera, lids, and conjunctiva
    # Expand horizontally by 35% on each side, vertically by 35% on top and 45% on bottom (for palpebral conjunctiva)
    pad_x = int(ew * 0.35)
    pad_top = int(eh * 0.35)
    pad_bot = int(eh * 0.45)

    cx = max(0, ex - pad_x)
    cy = max(0, ey - pad_top)
    cw = min(img_w - cx, ew + 2 * pad_x)
    ch = min(img_h - cy, eh + pad_top + pad_bot)

    # Ensure minimum width and height
    cw = max(cw, 50)
    ch = max(ch, 50)

    is_macro = (cw / img_w >= 0.20) or (ew / img_w >= 0.15)

    if not is_macro:
        guidance = "Move closer to frame the eye clearly."
    elif full_face_present and len(candidate_eyes) >= 2:
        guidance = "Keep one eye inside the frame — move closer for a close-up macro view."
    else:
        guidance = "Keep one eye inside the frame. Hold steady to capture."

    return {
        "eye_detected": True,
        "eye_count": len(candidate_eyes),
        "primary_eye_box": {"x": ex, "y": ey, "w": ew, "h": eh},
        "crop_box": {"x": cx, "y": cy, "w": cw, "h": ch},
        "aspect_ratio": round(cw / ch, 3) if ch > 0 else 1.0,
        "is_macro_framing": is_macro,
        "guidance": guidance,
        "anatomical_disclaimer": (
            "Ocular region detection aids framing only; it does not confirm that "
            "the inner palpebral or forniceal conjunctiva is properly visible."
        )
    }


def crop_ocular_image(
    image: Image.Image,
    crop_box: Optional[Dict[str, int]] = None
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Performs an aspect-ratio-preserving crop of the ocular region.
    If crop_box is not provided, runs detect_ocular_region to determine optimal boundaries.
    """
    img_rgb = image.convert('RGB')
    orig_w, orig_h = img_rgb.size

    if crop_box is None:
        det = detect_ocular_region(img_rgb)
        crop_box = det.get("crop_box")
        eye_detected = det.get("eye_detected", False)
        guidance = det.get("guidance", "")
    else:
        eye_detected = True
        guidance = "Custom crop box applied."

    if not crop_box:
        return img_rgb, {
            "crop_applied": False,
            "crop_box": None,
            "original_dimensions": [orig_w, orig_h],
            "output_dimensions": [orig_w, orig_h],
            "aspect_ratio": round(orig_w / orig_h, 3) if orig_h > 0 else 1.0,
            "eye_detected": False,
            "guidance": "No crop box determined."
        }

    x = max(0, min(orig_w - 1, int(crop_box.get('x', 0))))
    y = max(0, min(orig_h - 1, int(crop_box.get('y', 0))))
    w = max(10, min(orig_w - x, int(crop_box.get('w', orig_w))))
    h = max(10, min(orig_h - y, int(crop_box.get('h', orig_h))))

    cropped_img = img_rgb.crop((x, y, x + w, y + h))
    out_w, out_h = cropped_img.size

    return cropped_img, {
        "crop_applied": True,
        "crop_box": {"x": x, "y": y, "w": w, "h": h},
        "original_dimensions": [orig_w, orig_h],
        "output_dimensions": [out_w, out_h],
        "aspect_ratio": round(out_w / out_h, 3) if out_h > 0 else 1.0,
        "eye_detected": eye_detected,
        "guidance": guidance
    }
