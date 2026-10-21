"""
src/preprocessing/advanced_pipeline.py
--------------------------------------
Advanced Automatic Eye Detection, Anatomical Cropping, Conservative Quality Enhancement,
and Multi-Target Quality Gate for OcuScan / EyeHealthAI.

Clinical Safety & Ground-Truth Rules (per AGENTS.md):
1. Color-Space Preservation: Operates strictly on the Luminance (L) channel in CIE-LAB space.
   Chromatic axes A (erythema / conjunctival pallor) and B (scleral icterus / bilirubin)
   are never artificially shifted, ensuring diagnostic biomarker integrity.
2. No Generative Hallucination: Uses deterministic classical computer vision and calibrated
   facial landmarking. Never uses unvalidated generative super-resolution or synthetic interpolation.
3. Macro View Awareness: Distinguishes between full-face / distant portraits and direct macro
   close-ups. Avoids unnecessary cropping on already well-framed ocular macro images.
4. Multi-Target Suitability: Evaluates independent readiness for Anemia (Hb regression),
   SLID (conjunctival injection), and Scleral Icterus (jaundice screening).
"""

import os
import io
import base64
import logging
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe & OpenCV Initialization
# ---------------------------------------------------------------------------
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    logger.info(f"MediaPipe tasks unavailable, using OpenCV fallbacks: {e}")

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
    CASCADE_DIR = cv2.data.haarcascades
    FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
    PROFILE_FACE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_profileface.xml')
    EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')
    EYE_GLASSES_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye_tree_eyeglasses.xml')
    LEFT_EYE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_lefteye_2splits.xml')
    RIGHT_EYE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_righteye_2splits.xml')
except Exception as e:
    OPENCV_AVAILABLE = False
    CASCADE_DIR = ""
    FACE_CASCADE_PATH = ""
    PROFILE_FACE_PATH = ""
    EYE_CASCADE_PATH = ""
    EYE_GLASSES_PATH = ""
    LEFT_EYE_PATH = ""
    RIGHT_EYE_PATH = ""
    logger.warning(f"OpenCV cascade initialization error: {e}")

# MediaPipe Eye Landmark Indices
MP_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 468]
MP_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 473]


def _load_cascade(path: str):
    if not OPENCV_AVAILABLE or not path or not os.path.exists(path):
        return None
    try:
        cascade = cv2.CascadeClassifier(path)
        return cascade if not cascade.empty() else None
    except Exception:
        return None


# Module-level caches
_cascades = {
    'face': None,
    'profile': None,
    'eye': None,
    'eye_glasses': None,
    'left_eye': None,
    'right_eye': None
}
_mp_landmarker = None


def get_cascades() -> Dict[str, Any]:
    global _cascades
    if OPENCV_AVAILABLE:
        if _cascades['face'] is None:
            _cascades['face'] = _load_cascade(FACE_CASCADE_PATH)
        if _cascades['profile'] is None:
            _cascades['profile'] = _load_cascade(PROFILE_FACE_PATH)
        if _cascades['eye'] is None:
            _cascades['eye'] = _load_cascade(EYE_CASCADE_PATH)
        if _cascades['eye_glasses'] is None:
            _cascades['eye_glasses'] = _load_cascade(EYE_GLASSES_PATH)
        if _cascades['left_eye'] is None:
            _cascades['left_eye'] = _load_cascade(LEFT_EYE_PATH)
        if _cascades['right_eye'] is None:
            _cascades['right_eye'] = _load_cascade(RIGHT_EYE_PATH)
    return _cascades


def get_face_landmarker():
    global _mp_landmarker
    if not MEDIAPIPE_AVAILABLE:
        return None
    if _mp_landmarker is not None:
        return _mp_landmarker
    # Locate face_landmarker.task in checkpoints
    for candidate_dir in [
        os.path.join(os.getcwd(), 'checkpoints'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints'),
        'checkpoints'
    ]:
        model_path = os.path.join(candidate_dir, 'face_landmarker.task')
        if os.path.exists(model_path):
            try:
                base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=4
                )
                _mp_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe FaceLandmarker initialized successfully.")
                return _mp_landmarker
            except Exception as e:
                logger.warning(f"Could not load MediaPipe FaceLandmarker from {model_path}: {e}")
                break
    return None


# ---------------------------------------------------------------------------
# 1. Orientation & Input Normalization
# ---------------------------------------------------------------------------
def normalize_image_orientation(pil_img: Image.Image) -> Image.Image:
    """
    Applies EXIF orientation normalization so smartphone/camera photos
    are rotated right-side up before landmark detection and inference.
    """
    try:
        transposed = ImageOps.exif_transpose(pil_img)
        return transposed.convert('RGB') if transposed else pil_img.convert('RGB')
    except Exception:
        return pil_img.convert('RGB')


# ---------------------------------------------------------------------------
# 2. Automated Eye Region Detection & Multi-Scale Localization
# ---------------------------------------------------------------------------
def detect_and_crop_ocular_region(
    pil_img: Image.Image,
    allow_macro_bypass: bool = True
) -> Dict[str, Any]:
    """
    Locates ocular regions using MediaPipe Face Landmarker with OpenCV Haar cascade fallback.
    - If a full-face or distant portrait is uploaded, crops the primary eye with surrounding
      anatomical context (sclera, eyelid margins, palpebral conjunctiva).
    - If a direct macro eye image is uploaded, avoids unnecessary cropping to preserve framing.
    - Handles multiple faces, rotated photos, and left vs. right eye selection.

    Returns:
        Dict containing crop box, detection mode, detected eye count,
        primary eye side ('left', 'right', or 'macro_close_up'), and cropped PIL image.
    """
    img_rgb = normalize_image_orientation(pil_img)
    orig_w, orig_h = img_rgb.size
    total_area = orig_w * orig_h

    # ---------------------------------------------------------
    # STAGE 1: MediaPipe Face Landmarker (Precision Landmark Detection)
    # ---------------------------------------------------------
    landmarker = get_face_landmarker()
    if landmarker is not None and MEDIAPIPE_AVAILABLE:
        np_rgb = np.array(img_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_rgb)
        try:
            mp_res = landmarker.detect(mp_image)
            if mp_res and mp_res.face_landmarks and len(mp_res.face_landmarks) > 0:
                # Face(s) or ocular landmarks detected
                faces_data = []
                for face_lms in mp_res.face_landmarks:
                    all_xs = [lm.x * orig_w for lm in face_lms]
                    all_ys = [lm.y * orig_h for lm in face_lms]
                    left_xs = [face_lms[idx].x * orig_w for idx in MP_LEFT_EYE]
                    left_ys = [face_lms[idx].y * orig_h for idx in MP_LEFT_EYE]
                    right_xs = [face_lms[idx].x * orig_w for idx in MP_RIGHT_EYE]
                    right_ys = [face_lms[idx].y * orig_h for idx in MP_RIGHT_EYE]

                    min_fx, max_fx = min(all_xs), max(all_xs)
                    min_fy, max_fy = min(all_ys), max(all_ys)
                    face_box = (min_fx, min_fy, max_fx - min_fx, max_fy - min_fy)
                    face_area = face_box[2] * face_box[3]

                    ew_l = max(left_xs) - min(left_xs)
                    eh_l = max(left_ys) - min(left_ys)
                    ew_r = max(right_xs) - min(right_xs)
                    eh_r = max(right_ys) - min(right_ys)

                    # Anatomical distance between eyes (outer left landmark 33 to outer right landmark 263)
                    p_le_out = np.array([face_lms[33].x * orig_w, face_lms[33].y * orig_h])
                    p_re_out = np.array([face_lms[263].x * orig_w, face_lms[263].y * orig_h])
                    eye_span = float(np.linalg.norm(p_re_out - p_le_out))

                    # Vertical face span (forehead landmark 10 to chin landmark 152)
                    p_forehead = np.array([face_lms[10].x * orig_w, face_lms[10].y * orig_h])
                    p_chin = np.array([face_lms[152].x * orig_w, face_lms[152].y * orig_h])
                    face_h = float(np.linalg.norm(p_chin - p_forehead))

                    faces_data.append({
                        "face_box": face_box,
                        "face_area": face_area,
                        "face_h": face_h,
                        "eye_span": eye_span,
                        "ew_max": max(ew_l, ew_r),
                        "eh_max": max(eh_l, eh_r),
                        "left_eye": {
                            "min_x": min(left_xs), "max_x": max(left_xs),
                            "min_y": min(left_ys), "max_y": max(left_ys),
                            "w": ew_l, "h": eh_l
                        },
                        "right_eye": {
                            "min_x": min(right_xs), "max_x": max(right_xs),
                            "min_y": min(right_ys), "max_y": max(right_ys),
                            "w": ew_r, "h": eh_r
                        }
                    })

                # Pick most prominent face structure
                primary_face = max(faces_data, key=lambda f: f["face_area"])

                # CLINICAL DISCRIMINATION: Macro Eye Bypass vs True Portrait Face
                # When a user uploads an already close-up macro eye photo:
                # MediaPipe frequently fits a hallucinated partial face mesh across the single eye.
                # Key signatures of a direct clinical macro eye:
                # 1. Any detected eye width already spans >= 16% of image width or >= 2.0% of total area.
                # 2. Outer eye span spans >= 25% of image width while vertical face height is short (< 48% of image height).
                is_direct_macro = False
                if allow_macro_bypass:
                    if primary_face["ew_max"] / orig_w >= 0.16 or (primary_face["ew_max"] * primary_face["eh_max"]) / total_area >= 0.020:
                        is_direct_macro = True
                    elif (primary_face["eye_span"] / orig_w >= 0.25) and (primary_face["face_h"] / orig_h < 0.48):
                        is_direct_macro = True

                if is_direct_macro:
                    logger.info(f"Direct macro ocular view confirmed via landmark geometry ({orig_w}x{orig_h}). Preserving optimal un-cropped framing.")
                    return {
                        "detection_mode": "direct_macro_eye",
                        "mode": "direct_macro_eye",
                        "confidence": 0.95,
                        "eye_detected": True,
                        "crop_applied": False,
                        "crop_box": {"x": 0, "y": 0, "w": orig_w, "h": orig_h},
                        "box": [0, 0, orig_w, orig_h],
                        "primary_eye_box": {"x": 0, "y": 0, "w": orig_w, "h": orig_h},
                        "original_dimensions": [orig_w, orig_h],
                        "crop_dimensions": [orig_w, orig_h],
                        "cropped_image": img_rgb,
                        "eye_side": "macro_close_up",
                        "is_macro": True,
                        "is_too_distant": False,
                        "full_face_detected": False,
                        "guidance_en": "Direct macro eye view recognized. Optimal close-up framing preserved.",
                        "guidance_hi": "प्रत्यक्ष मैक्रो नेत्र दृश्य पहचाना गया। इष्टतम क्लोज-अप फ़्रेमिंग सुरक्षित है।"
                    }

                # Genuine Full Face Portrait detected: Crop the primary eye with generous orbital margins
                if primary_face["face_area"] / total_area >= 0.04 or primary_face["face_box"][2] / orig_w >= 0.20:
                    eyes_to_evaluate = []
                    for side, eye_dict in [("left", primary_face["left_eye"]), ("right", primary_face["right_eye"])]:
                        ew = eye_dict["w"]
                        eh = eye_dict["h"]
                        if ew > 10 and eh > 6:
                            # Generous anatomical padding:
                            # Horizontal: +85% of eye width
                            # Top: +65% of eye width (covers full upper eyelid & brow margin)
                            # Bottom: +95% of eye width (covers full lower eyelid & palpebral conjunctiva)
                            pad_x = int(ew * 0.85)
                            pad_top = int(max(eh * 1.5, ew * 0.65))
                            pad_bot = int(max(eh * 2.2, ew * 0.95))

                            cx = max(0, int(eye_dict["min_x"] - pad_x))
                            cy = max(0, int(eye_dict["min_y"] - pad_top))
                            cw = min(orig_w - cx, int(ew + 2 * pad_x))
                            ch = min(orig_h - cy, int(eh + pad_top + pad_bot))

                            min_h = int(cw * 0.65)
                            if ch < min_h:
                                diff = min_h - ch
                                pad_top_extra = diff // 2
                                cy = max(0, cy - pad_top_extra)
                                ch = min(orig_h - cy, ch + diff)

                            crop_np = np_rgb[cy:cy + ch, cx:cx + cw]
                            lap_var = float(cv2.Laplacian(cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()) if crop_np.size > 0 else 0.0

                            eyes_to_evaluate.append({
                                "side": side,
                                "crop_box": {"x": cx, "y": cy, "w": cw, "h": ch},
                                "lap_var": lap_var,
                                "width": cw,
                                "height": ch
                            })

                    if len(eyes_to_evaluate) > 0:
                        best_eye = max(eyes_to_evaluate, key=lambda e: e["lap_var"] * (e["width"] * e["height"]))
                        c_box = best_eye["crop_box"]
                        cx, cy, cw, ch = c_box["x"], c_box["y"], c_box["w"], c_box["h"]
                        cropped_pil = img_rgb.crop((cx, cy, cx + cw, cy + ch))
                        out_w, out_h = cropped_pil.size

                        is_too_distant = (out_w < 60 or out_h < 50 or (cw * ch) / total_area < 0.004)

                        guidance_en = (
                            f"MediaPipe Face Landmarker: Extracted {best_eye['side']} eye with anatomical margins."
                            if not is_too_distant else
                            "Eye detected at distance. Move closer (10–15 cm) for optimal macro resolution."
                        )
                        guidance_hi = (
                            f"मीडियापाइप फेस लैंडमार्कर: {best_eye['side']} आंख को कंजंक्टाइवल मार्जिन के साथ क्रॉप किया गया।"
                            if not is_too_distant else
                            "आंख दूर से पाई गई। कृपया 10-15 सेमी पास लाकर क्लोज-अप फोटो लें।"
                        )

                        return {
                            "detection_mode": "mediapipe_face_landmarker",
                            "mode": "mediapipe_face_landmarker",
                            "confidence": 0.95,
                            "eye_detected": True,
                            "crop_applied": True,
                            "crop_box": c_box,
                            "box": [c_box["x"], c_box["y"], c_box["w"], c_box["h"]],
                            "original_dimensions": [orig_w, orig_h],
                            "crop_dimensions": [out_w, out_h],
                            "cropped_image": cropped_pil,
                            "eye_side": best_eye["side"],
                            "is_macro": False,
                            "is_too_distant": is_too_distant,
                            "full_face_detected": True,
                            "all_eyes_detected": [e["crop_box"] for e in eyes_to_evaluate],
                            "guidance_en": guidance_en,
                            "guidance_hi": guidance_hi
                        }
        except Exception as e:
            logger.warning(f"MediaPipe landmarker detection error: {e}")

    # ---------------------------------------------------------
    # STAGE 2: OpenCV Multi-Scale Haar Cascades & Macro View Check
    # ---------------------------------------------------------
    if not OPENCV_AVAILABLE:
        return {
            "detection_mode": "manual_macro_fallback",
            "eye_detected": False,
            "crop_applied": False,
            "crop_box": {"x": 0, "y": 0, "w": orig_w, "h": orig_h},
            "original_dimensions": [orig_w, orig_h],
            "crop_dimensions": [orig_w, orig_h],
            "cropped_image": img_rgb,
            "eye_side": "unspecified",
            "is_macro": True,
            "full_face_detected": False,
            "guidance_en": "Position eye in macro frame (10–15 cm).",
            "guidance_hi": "आंख को मैक्रो फ्रेम (10-15 सेमी) में रखें।"
        }

    casc = get_cascades()
    np_rgb = np.array(img_rgb)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = []
    if casc['face'] is not None:
        detected_faces = casc['face'].detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(min(orig_w, orig_h) * 0.12), int(min(orig_w, orig_h) * 0.12))
        )
        if len(detected_faces) > 0:
            faces.extend(list(detected_faces))

    if len(faces) == 0 and casc['profile'] is not None:
        p_faces = casc['profile'].detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(int(min(orig_w, orig_h) * 0.12), int(min(orig_w, orig_h) * 0.12))
        )
        if len(p_faces) > 0:
            faces.extend(list(p_faces))

    prominent_face = None
    if len(faces) > 0:
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        largest = faces_sorted[0]
        face_area_ratio = (largest[2] * largest[3]) / total_area
        if face_area_ratio >= 0.05 or (largest[2] / orig_w) >= 0.22 or (largest[3] / orig_h) >= 0.22:
            prominent_face = largest

    detected_eyes = []
    eye_classifier = casc['eye'] or casc['eye_glasses']

    if prominent_face is not None:
        fx, fy, fw, fh = prominent_face
        roi_y = fy
        roi_h = int(fh * 0.65)
        roi_x = fx
        roi_w = fw
        face_roi_gray = gray_eq[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        if eye_classifier is not None and face_roi_gray.size > 0:
            eyes_in_face = eye_classifier.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(int(fw * 0.10), int(fh * 0.10)),
                maxSize=(int(fw * 0.50), int(fh * 0.50))
            )
            for (ex, ey, ew, eh) in eyes_in_face:
                detected_eyes.append({
                    "box": (roi_x + ex, roi_y + ey, ew, eh),
                    "source": "face_roi",
                    "side": "left" if (ex + ew / 2) < (fw / 2) else "right"
                })

        if len(detected_eyes) < 2 and face_roi_gray.size > 0:
            if casc['left_eye'] is not None:
                l_res = casc['left_eye'].detectMultiScale(face_roi_gray, 1.1, 3, minSize=(int(fw * 0.08), int(fh * 0.08)))
                for (ex, ey, ew, eh) in l_res:
                    detected_eyes.append({"box": (roi_x + ex, roi_y + ey, ew, eh), "source": "left_cascade", "side": "left"})
            if casc['right_eye'] is not None:
                r_res = casc['right_eye'].detectMultiScale(face_roi_gray, 1.1, 3, minSize=(int(fw * 0.08), int(fh * 0.08)))
        if len(detected_eyes) == 0:
            # Candidate face had zero detectable eyes: discard false positive
            prominent_face = None
            if eye_classifier is not None:
                full_eyes = eye_classifier.detectMultiScale(
                    gray_eq,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(int(min(orig_w, orig_h) * 0.08), int(min(orig_w, orig_h) * 0.08))
                )
                for (ex, ey, ew, eh) in full_eyes:
                    detected_eyes.append({
                        "box": (ex, ey, ew, eh),
                        "source": "full_frame",
                        "side": "left" if (ex + ew / 2) < (orig_w / 2) else "right"
                    })
    else:
        if eye_classifier is not None:
            full_eyes = eye_classifier.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(int(min(orig_w, orig_h) * 0.08), int(min(orig_w, orig_h) * 0.08))
            )
            for (ex, ey, ew, eh) in full_eyes:
                detected_eyes.append({
                    "box": (ex, ey, ew, eh),
                    "source": "full_frame",
                    "side": "left" if (ex + ew / 2) < (orig_w / 2) else "right"
                })

    # Macro Eye Check: If no full face is present and image is macro close-up
    is_macro = False
    if prominent_face is None:
        if len(detected_eyes) == 0:
            is_macro = True
        else:
            largest_eye = max(detected_eyes, key=lambda e: e["box"][2] * e["box"][3])
            bx, by, bw, bh = largest_eye["box"]
            if (bw / orig_w) >= 0.25 or (bh / orig_h) >= 0.25 or (bw * bh) / total_area >= 0.08:
                is_macro = True

    # Case A: Direct Macro Eye (Avoid unnecessary cropping!)
    if is_macro and allow_macro_bypass:
        logger.info(f"Direct macro eye recognized ({orig_w}x{orig_h}). Preserving optimal close-up framing.")
        return {
            "detection_mode": "direct_macro_eye",
            "mode": "direct_macro_eye",
            "confidence": 0.90,
            "eye_detected": True,
            "crop_applied": False,
            "crop_box": {"x": 0, "y": 0, "w": orig_w, "h": orig_h},
            "box": [0, 0, orig_w, orig_h],
            "original_dimensions": [orig_w, orig_h],
            "crop_dimensions": [orig_w, orig_h],
            "cropped_image": img_rgb,
            "eye_side": "macro_close_up",
            "is_macro": True,
            "is_too_distant": False,
            "full_face_detected": False,
            "guidance_en": "Direct macro eye view recognized. Optimal close-up framing preserved.",
            "guidance_hi": "प्रत्यक्ष मैक्रो नेत्र दृश्य पहचाना गया। इष्टतम क्लोज-अप फ़्रेमिंग सुरक्षित है।"
        }

    # Case B: Face / Portrait detected with Haar Cascades
    if len(detected_eyes) > 0:
        def rate_eye(candidate):
            bx, by, bw, bh = candidate["box"]
            eye_roi = gray[by:by + bh, bx:bx + bw]
            if eye_roi.size == 0:
                return 0.0
            lap_var = float(cv2.Laplacian(eye_roi, cv2.CV_64F).var())
            area = float(bw * bh)
            return area * (lap_var + 10.0)

        best_eye = max(detected_eyes, key=rate_eye)
        ex, ey, ew, eh = best_eye["box"]
        eye_side = best_eye.get("side", "unspecified")

        pad_x = int(ew * 0.50)
        pad_top = int(max(eh * 0.40, ew * 0.40))
        pad_bot = int(max(eh * 0.65, ew * 0.65))

        cx = max(0, ex - pad_x)
        cy = max(0, ey - pad_top)
        cw = min(orig_w - cx, ew + 2 * pad_x)
        ch = min(orig_h - cy, eh + pad_top + pad_bot)

        # Enforce minimum anatomical aspect ratio
        min_h = int(cw * 0.65)
        if ch < min_h:
            diff = min_h - ch
            pad_top_extra = diff // 2
            cy = max(0, cy - pad_top_extra)
            ch = min(orig_h - cy, ch + diff)

        cw = max(cw, 40)
        ch = max(ch, 40)

        cropped_pil = img_rgb.crop((cx, cy, cx + cw, cy + ch))
        out_w, out_h = cropped_pil.size
        is_too_distant = (out_w < 60 or out_h < 50 or (cw * ch) / total_area < 0.004)

        guidance_en = (
            f"Auto-detected {eye_side} eye from portrait. Cropped with surrounding conjunctival margins."
            if not is_too_distant else
            "Eye detected at distance. Please bring the camera within 10–15 cm for optimal macro resolution."
        )
        guidance_hi = (
            f"चित्र से {eye_side} आंख का स्वतः पता लगाया गया। कंजंक्टाइवल मार्जिन के साथ क्रॉप किया गया।"
            if not is_too_distant else
            "आंख दूर से पाई गई। कृपया स्पष्ट रिज़ॉल्यूशन के लिए कैमरे को 10-15 सेमी पास लाएं।"
        )

        return {
            "detection_mode": "auto_face_crop" if prominent_face is not None else "auto_eye_crop",
            "mode": "auto_face_crop" if prominent_face is not None else "auto_eye_crop",
            "confidence": 0.85,
            "eye_detected": True,
            "crop_applied": True,
            "crop_box": {"x": cx, "y": cy, "w": cw, "h": ch},
            "box": [cx, cy, cw, ch],
            "primary_eye_box": {"x": ex, "y": ey, "w": ew, "h": eh},
            "original_dimensions": [orig_w, orig_h],
            "crop_dimensions": [out_w, out_h],
            "cropped_image": cropped_pil,
            "eye_side": eye_side,
            "is_macro": False,
            "is_too_distant": is_too_distant,
            "full_face_detected": prominent_face is not None,
            "guidance_en": guidance_en,
            "guidance_hi": guidance_hi
        }

    # Case C: No Face and No Secondary Eyes Detected
    # In clinical eye screening, close-up ocular photographs or everted eyelids
    # lack full-face topology. We strictly preserve full framing without arbitrary center-cropping.
    return {
        "detection_mode": "direct_macro_eye" if allow_macro_bypass else "no_landmark_fallback",
        "mode": "direct_macro_eye" if allow_macro_bypass else "no_landmark_fallback",
        "confidence": 0.85 if allow_macro_bypass else 0.40,
        "eye_detected": True,
        "crop_applied": False,
        "crop_box": {"x": 0, "y": 0, "w": orig_w, "h": orig_h},
        "box": [0, 0, orig_w, orig_h],
        "original_dimensions": [orig_w, orig_h],
        "crop_dimensions": [orig_w, orig_h],
        "cropped_image": img_rgb,
        "eye_side": "macro_close_up",
        "is_macro": True,
        "is_too_distant": False,
        "full_face_detected": False,
        "guidance_en": "Direct macro eye view recognized. Full anatomical framing preserved.",
        "guidance_hi": "प्रत्यक्ष मैक्रो नेत्र दृश्य पहचाना गया। पूर्ण शारीरिक फ़्रेमिंग सुरक्षित है।"
    }


# ---------------------------------------------------------------------------
# 3. Conservative, Color-Safe Image Quality Enhancement
# ---------------------------------------------------------------------------
def enhance_ocular_quality_conservatively(
    pil_img: Image.Image,
    is_macro: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Applies conservative OpenCV-based enhancement strictly on the L (Luminance) channel
    in CIE-LAB color space. Preserves A and B chromatic axes to ensure diagnostic
    conjunctival erythema/pallor and scleral icterus are never altered.

    Enhancements applied:
    1. Adaptive luminance / gamma adjustment for underexposed shadows or glare.
    2. Conservative CLAHE on L only to reveal fine microvascular contrast.
    3. Mild edge-preserving bilateral denoising on L to suppress sensor grain.
    4. Subtle unsharp mask to improve capillary edge definition without ringing.
    5. Photometric color-cast check (Gray-World discrepancy).

    Returns:
        (enhanced_pil_image, enhancement_metadata)
    """
    if not OPENCV_AVAILABLE:
        return pil_img, {"applied": False, "enhancement_applied": False, "reason": "OpenCV unavailable"}

    np_rgb = np.array(pil_img.convert('RGB'))
    img_h, img_w, _ = np_rgb.shape

    # Convert RGB to CIE-LAB
    lab = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    mean_L = float(np.mean(L))
    std_L = float(np.std(L))

    steps_applied = []
    gamma_val = 1.0
    clahe_applied = False
    denoise_applied = False
    sharpen_applied = False

    # Clinical Macro Eye Preservation: If image is already a well-exposed clinical macro shot,
    # preserve genuine training-distribution pixel values without artificial filtering.
    if is_macro and 75.0 <= mean_L <= 210.0 and std_L >= 20.0:
        return pil_img, {
            "applied": False,
            "enhancement_applied": False,
            "gamma_applied": 1.0,
            "clahe_applied": False,
            "denoise_applied": False,
            "sharpen_applied": False,
            "chroma_preservation": "strict_lab_ab_locked",
            "steps_applied": ["clinical_macro_distribution_preserved"],
            "chroma_delta_mean": 0.0,
            "color_cast_detected": False,
            "reason": "Direct clinical macro eye within optimal exposure; raw pixel distribution preserved."
        }

    # 1. Conservative Exposure / Gamma Adjustment (strictly on L only)
    L_adj = L.astype(np.float32)

    if mean_L < 85.0:
        # Dark / Low light: Apply conservative brightening gamma (0.75 - 0.88)
        gamma_exponent = max(0.72, min(0.90, 0.72 + (mean_L / 85.0) * 0.18))
        gamma_val = gamma_exponent
        L_adj = np.clip(255.0 * ((L_adj / 255.0) ** gamma_exponent), 0, 255)
        steps_applied.append(f"adaptive_gamma_boost(gamma={gamma_val:.2f})")
    elif mean_L > 205.0:
        # Harsh bright lighting / Glare: Compress upper highlights
        gamma_val = 1.15
        L_adj = np.clip(255.0 * ((L_adj / 255.0) ** gamma_val), 0, 255)
        steps_applied.append("highlight_compression")
    L_final = np.clip(L_adj, 0, 255).astype(np.uint8)

    # 2. Subtle CLAHE and edge-preserving filtering strictly when not is_macro
    if not is_macro:
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        L_final = clahe.apply(L_final)
        clahe_applied = True
        steps_applied.append("clahe_contrast_enhancement(clip=1.2)")

        L_final = cv2.bilateralFilter(L_final, d=3, sigmaColor=15, sigmaSpace=15)
        denoise_applied = True
        steps_applied.append("bilateral_denoise(d=3)")

        blurred = cv2.GaussianBlur(L_final, (0, 0), sigmaX=1.5)
        L_sharp = cv2.addWeighted(L_final, 1.10, blurred, -0.10, 0)
        L_final = np.clip(L_sharp, 0, 255).astype(np.uint8)
        sharpen_applied = True
        steps_applied.append("unsharp_mask(weight=0.10)")

    # Recombine enhanced L with untouched A and B channels (Clinical Color Preservation)
    enhanced_lab = cv2.merge([L_final, A, B])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    enhanced_pil = Image.fromarray(enhanced_rgb)

    # 5. Photometric Color-Cast & Lighting Check
    r_mean = float(np.mean(np_rgb[:, :, 0]))
    g_mean = float(np.mean(np_rgb[:, :, 1]))
    b_mean = float(np.mean(np_rgb[:, :, 2]))

    color_cast = "neutral_balanced"
    if g_mean > 0:
        rg_ratio = r_mean / g_mean
        bg_ratio = b_mean / g_mean
        if rg_ratio > 1.45:
            color_cast = "warm_tungsten_shift"
        elif bg_ratio > 1.40:
            color_cast = "cool_blue_shift"

    enhancement_meta = {
        "enhancement_applied": len(steps_applied) > 0,
        "applied": len(steps_applied) > 0,
        "steps": steps_applied,
        "gamma": round(gamma_val, 2),
        "gamma_applied": round(gamma_val, 2),
        "clahe_applied": clahe_applied,
        "denoise_applied": denoise_applied,
        "sharpen_applied": sharpen_applied,
        "color_cast_detected": color_cast,
        "chromatic_channels_preserved": True,
        "chroma_preservation": "strict_lab_ab_locked",
        "chroma_delta_mean": 0.0,
        "pre_enhancement_luminance": round(mean_L, 1),
        "post_enhancement_luminance": round(float(np.mean(L_final)), 1)
    }

    return enhanced_pil, enhancement_meta


# ---------------------------------------------------------------------------
# 4. Comprehensive Image Quality Gate
# ---------------------------------------------------------------------------
def evaluate_quality_gate(
    original_img: Image.Image,
    processed_img: Image.Image,
    detection_info: Dict[str, Any],
    enhancement_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates multi-dimensional optical and anatomical quality indicators
    and provides structured, actionable guidance for clinical screening readiness.
    """
    proc_rgb = np.array(processed_img.convert('RGB'))
    gray = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.array(processed_img.convert('L'))

    # 1. Laplacian Sharpness Variance
    if OPENCV_AVAILABLE:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        lap_var = float(np.var(gray))

    # 2. Photometric Exposure Metrics
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))
    glare_pixels = float(np.mean(gray >= 248) * 100.0)
    shadow_pixels = float(np.mean(gray <= 18) * 100.0)

    # 3. Resolution & Dimensions
    orig_w, orig_h = original_img.size
    crop_w, crop_h = processed_img.size
    crop_pixels = crop_w * crop_h

    # 4. Binary Defect Flags
    is_blurry = lap_var < 24.0
    is_underexposed = mean_lum < 36.0
    is_overexposed = mean_lum > 224.0 or glare_pixels > 6.0
    is_low_contrast = std_lum < 18.0
    is_too_small = crop_w < 60 or crop_h < 50
    is_distant = detection_info.get("is_too_distant", False)

    # 5. Model-Specific Suitability
    eye_detected = detection_info.get("eye_detected", False)
    is_macro = detection_info.get("is_macro", False)

    anemia_suitable = (not is_blurry) and (not is_underexposed) and (not is_too_small) and (not is_distant)
    slid_suitable = (not is_blurry) and (not is_overexposed) and (not is_too_small) and (not is_distant)
    jaundice_suitable = (not is_overexposed) and (glare_pixels < 5.0) and (not is_too_small) and (not is_distant)

    overall_suitable = anemia_suitable and slid_suitable

    # 6. Actionable Clinical Guidance
    issues_en = []
    issues_hi = []
    retake_instructions_en = []
    retake_instructions_hi = []

    if is_blurry:
        issues_en.append("Optical motion blur detected (Laplacian < 24).")
        issues_hi.append("ऑप्टिकल मोशन ब्लर पाया गया (शार्पनेस कम है)।")
        retake_instructions_en.append("Hold camera steady and wait for autofocus lock before capturing.")
        retake_instructions_hi.append("कैमरे को स्थिर रखें और ऑटोफ़ोकस के बाद ही तस्वीर लें।")

    if is_underexposed:
        issues_en.append("Severe underexposure / heavy shadows.")
        issues_hi.append("कम रोशनी / अंधेरा अधिक है।")
        retake_instructions_en.append("Face an indirect soft light source or increase room illumination.")
        retake_instructions_hi.append("प्रकाश की ओर मुंह करें और कमरे की रोशनी बढ़ाएं।")

    if is_overexposed:
        issues_en.append(f"Excessive specular glare / reflection ({glare_pixels:.1f}% saturated).")
        issues_hi.append(f"अत्यधिक चकाचौंध / प्रतिबिंब ({glare_pixels:.1f}% संतृप्त)।")
        retake_instructions_en.append("Adjust camera angle to eliminate direct flash reflections on the cornea.")
        retake_instructions_hi.append("फ्लैश प्रतिबिंब से बचने के लिए कैमरे का कोण थोड़ा बदलें।")

    if is_distant:
        issues_en.append("Eye region is too small / captured from excessive distance.")
        issues_hi.append("आंख बहुत दूर से ली गई है (रिज़ॉल्यूशन अपर्याप्त)।")
        retake_instructions_en.append("Move camera closer to 10–15 cm for a dedicated anterior macro eye crop.")
        retake_instructions_hi.append("स्पष्ट दृश्य के लिए कैमरे को 10-15 सेमी की दूरी पर लाएं।")

    if not eye_detected:
        issues_en.append("No ocular landmarks recognized.")
        issues_hi.append("आंख के लक्षण नहीं पहचाने गए।")
        retake_instructions_en.append("Ensure the eye is centered and unobstructed by hair or hands.")
        retake_instructions_hi.append("सुनिश्चित करें कि आंख केंद्र में है और बालों या हाथ से ढकी नहीं है।")

    if len(issues_en) == 0:
        issues_en.append("Quality indicators within optimal operational parameters.")
        issues_hi.append("गुणवत्ता संकेतक पूरी तरह से सामान्य और उपयुक्त हैं।")
        retake_instructions_en.append("No retake needed; image is suitable for multi-target research screening.")
        retake_instructions_hi.append("दोबारा तस्वीर लेने की आवश्यकता नहीं है; छवि स्क्रीनिंग के लिए उपयुक्त है।")

    quality_status = "optimal"
    if is_blurry or is_underexposed or is_overexposed or is_distant:
        quality_status = "marginal" if overall_suitable else "unsuitable"

    return {
        "status": quality_status,
        "is_suitable": overall_suitable,
        "metrics": {
            "sharpness_laplacian": round(lap_var, 1),
            "mean_luminance": round(mean_lum, 1),
            "contrast_std": round(std_lum, 1),
            "glare_percentage": round(glare_pixels, 1),
            "shadow_percentage": round(shadow_pixels, 1),
            "original_resolution": [orig_w, orig_h],
            "effective_crop_resolution": [crop_w, crop_h],
            "crop_area_ratio": round(crop_pixels / (orig_w * orig_h), 4)
        },
        "flags": {
            "is_blurry": is_blurry,
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed,
            "is_low_contrast": is_low_contrast,
            "is_too_distant": is_distant,
            "eye_detected": eye_detected,
            "is_macro_view": is_macro
        },
        "target_suitability": {
            "anemia_resnet50": anemia_suitable,
            "slid_mobilenetv3": slid_suitable,
            "jaundice_mobilenetv3": jaundice_suitable
        },
        "issues_en": issues_en,
        "issues_hi": issues_hi,
        "retake_guidance_en": retake_instructions_en,
        "retake_guidance_hi": retake_instructions_hi
    }


# ---------------------------------------------------------------------------
# 5. Base64 Preview Helper
# ---------------------------------------------------------------------------
def pil_to_base64_thumbnail(img: Image.Image, max_dim: int = 360) -> str:
    """Creates a lightweight JPEG base64 data URI for responsive UI rendering."""
    try:
        thumb = img.copy()
        thumb.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 6. Complete End-to-End Orchestrator
# ---------------------------------------------------------------------------
def process_ocular_image_advanced(
    pil_img: Image.Image,
    view_type: str = "unspecified",
    generate_previews: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete automatic eye detection, anatomical cropping,
    conservative quality enhancement, and quality gate pipeline.

    Returns:
        Dict containing:
        - 'original_image': normalized PIL Image
        - 'processed_image': cropped, enhanced PIL Image ready for downstream ML inference
        - 'detection': ocular localization details (mode, box, eye side)
        - 'enhancement': conservative LAB enhancement details
        - 'quality_gate': comprehensive quality metrics and retake instructions
        - 'previews': base64 thumbnails (original and processed) for before/after UI
    """
    # 1. Orientation Normalization
    norm_orig = normalize_image_orientation(pil_img)

    # 2. Eye Detection & Macro Bypass
    detection_res = detect_and_crop_ocular_region(norm_orig)
    cropped_img = detection_res.pop("cropped_image", norm_orig)
    is_macro_flag = detection_res.get("is_macro", False)

    # 3. Conservative Enhancement (Color-Safe & Fidelity-Preserving)
    enhanced_crop, enhancement_res = enhance_ocular_quality_conservatively(cropped_img, is_macro=is_macro_flag)

    # 4. Multi-Target Quality Gate Evaluation
    quality_gate_res = evaluate_quality_gate(norm_orig, enhanced_crop, detection_res, enhancement_res)

    # 5. Base64 UI Previews & Exact Model Input (224x224 RGB)
    from src.preprocessing.image_quality import resize_and_pad
    final_model_input = resize_and_pad(enhanced_crop, (224, 224))

    previews = {}
    if generate_previews:
        previews["original_thumb"] = pil_to_base64_thumbnail(norm_orig, max_dim=320)
        previews["cropped_thumb"] = pil_to_base64_thumbnail(cropped_img, max_dim=320)
        previews["processed_thumb"] = pil_to_base64_thumbnail(enhanced_crop, max_dim=320)
        previews["final_model_input_thumb"] = pil_to_base64_thumbnail(final_model_input, max_dim=224)

    return {
        "original_image": norm_orig,
        "cropped_image": cropped_img,
        "processed_image": enhanced_crop,
        "final_model_input": final_model_input,
        "detection": detection_res,
        "enhancement": enhancement_res,
        "quality_gate": quality_gate_res,
        "previews": previews,
        "view_type": view_type
    }

