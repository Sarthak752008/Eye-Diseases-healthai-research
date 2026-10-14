"""
src/inference/unified_service.py
--------------------------------
Layer 4 - Unified Inference Service with Reliability-Gated Integration.

Implements:
1. Unified dispatch for ocular screening requests across Anemia, Conjunctivitis, and Jaundice.
2. Permitted research-only inference for Anemia (ResNet-50) supporting single-view and multi-view aggregation.
3. Reliability gate for Conjunctivitis: blocks exposure of CNN predictions due to the 100% metadata/resolution shortcut.
4. Missing-data gate for Jaundice: blocks execution and reports explicit data deficiency without placeholder fabrication.
5. Common, standardized response schema across all modules.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    import torch
    from torchvision import transforms
    from src.models.anemia_model import load_anemia_checkpoint, AnemiaRegressor
    from src.models.conjunctivitis_model import load_conjunctivitis_checkpoint, ConjunctivitisClassifier
    from src.models.jaundice_model import load_jaundice_checkpoint, JaundiceClassifier
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    torch = None
    transforms = None
    load_anemia_checkpoint = None
    AnemiaRegressor = Any  # type: ignore
    load_conjunctivitis_checkpoint = None
    ConjunctivitisClassifier = Any  # type: ignore
    load_jaundice_checkpoint = None
    JaundiceClassifier = Any  # type: ignore
    TORCH_AVAILABLE = False
from src.preprocessing.image_quality import (
    safe_load_image,
    calculate_image_quality,
    resize_and_pad,
    verify_ocular_framing_or_detect_face
)
from src.inference.clinical_analytics import (
    extract_ocular_chromatic_features,
    compute_anemia_clinical_decision,
    compute_slid_clinical_decision,
    compute_jaundice_clinical_decision
)

# Allowed ocular views supported by the trained Anemia pipeline
COMPATIBLE_ANEMIA_VIEWS = {
    'raw_eye',
    'palpebral',
    'forniceal',
    'forniceal_palpebral',
    'unspecified'
}

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class InferenceResult(dict):
    """
    Standardized, JSON-serializable response schema for Layer 4 inference.
    """
    def __init__(
        self,
        module: str,
        status: str,
        input_quality: Optional[List[Dict[str, Any]]] = None,
        prediction: Optional[Any] = None,
        unit: Optional[str] = None,
        view_count: Optional[int] = None,
        view_mean: Optional[float] = None,
        view_std: Optional[float] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        disclaimer: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            module=module,
            status=status,
            input_quality=input_quality,
            prediction=prediction,
            unit=unit,
            view_count=view_count,
            view_mean=view_mean,
            view_std=view_std,
            preprocessing=preprocessing,
            warnings=warnings or [],
            limitations=limitations or [],
            disclaimer=disclaimer or "RESEARCH USE ONLY: Experimental system. Not for medical diagnosis or clinical use.",
            error=error,
            **kwargs
        )


class EyeHealthInferenceService:
    """
    Unified Inference Service for EyeHealthAI.
    Loads available checkpoints once and enforces reliability gates.
    """
    def __init__(
        self,
        anemia_checkpoint_path: Optional[str] = None,
        slid_checkpoint_path: Optional[str] = None,
        jaundice_checkpoint_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.device = device
        self.anemia_checkpoint_path = anemia_checkpoint_path or os.path.join(
            WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth'
        )
        self.slid_checkpoint_path = slid_checkpoint_path or os.path.join(
            WORKSPACE_ROOT, 'saved_models', 'slid', 'best_model.pth'
        )
        self.jaundice_checkpoint_path = jaundice_checkpoint_path or os.path.join(
            WORKSPACE_ROOT, 'saved_models', 'jaundice', 'best_model.pth'
        )
        self._anemia_model: Optional[AnemiaRegressor] = None
        self._slid_model: Optional[ConjunctivitisClassifier] = None
        self._jaundice_model: Optional[JaundiceClassifier] = None
        if TORCH_AVAILABLE and transforms is not None:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self._transform = None

    def _get_anemia_model(self) -> AnemiaRegressor:
        """Lazy single-instance loading of the Anemia model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch runtime is not installed in the environment. Deep learning forward inference is unavailable."
            )
        if self._anemia_model is None:
            if not os.path.exists(self.anemia_checkpoint_path):
                raise FileNotFoundError(
                    f"Anemia checkpoint not found at: {self.anemia_checkpoint_path}"
                )
            self._anemia_model = load_anemia_checkpoint(
                self.anemia_checkpoint_path, device=self.device
            )
            self._anemia_model.eval()
        return self._anemia_model

    def _get_slid_model(self) -> ConjunctivitisClassifier:
        """Lazy single-instance loading of the SLID Conjunctival Injection model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch runtime is not installed in the environment. Deep learning forward inference is unavailable."
            )
        if self._slid_model is None:
            path = self.slid_checkpoint_path
            if not os.path.exists(path):
                alt = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'slid_injection_model.pth')
                if os.path.exists(alt):
                    path = alt
                else:
                    raise FileNotFoundError(f"SLID checkpoint not found at: {self.slid_checkpoint_path}")
            self._slid_model = load_conjunctivitis_checkpoint(path, device=self.device)
            self._slid_model.eval()
        return self._slid_model

    def _get_jaundice_model(self) -> JaundiceClassifier:
        """Lazy single-instance loading of the Jaundice Scleral Icterus model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch runtime is not installed in the environment. Deep learning forward inference is unavailable."
            )
        if self._jaundice_model is None:
            if not os.path.exists(self.jaundice_checkpoint_path):
                raise FileNotFoundError(f"Jaundice checkpoint not found at: {self.jaundice_checkpoint_path}")
            self._jaundice_model = load_jaundice_checkpoint(self.jaundice_checkpoint_path, device=self.device)
            self._jaundice_model.eval()
        return self._jaundice_model

    def _prepare_image_input(
        self,
        img_item: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """
        Validates, safely loads, checks quality, and prepares an image.
        Returns a dict with 'image', 'quality', and 'error' (if any).
        """
        if isinstance(img_item, str):
            if not os.path.exists(img_item):
                return {'image': None, 'quality': None, 'error': f"File does not exist: {img_item}"}
            ext = os.path.splitext(img_item)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                return {'image': None, 'quality': None, 'error': f"Unsupported image file format: {ext}"}
            loaded = safe_load_image(img_item)
            if loaded is None:
                return {'image': None, 'quality': None, 'error': f"Corrupt or unreadable image: {img_item}"}
            raw_img = loaded
        elif isinstance(img_item, Image.Image):
            raw_img = img_item.convert('RGB')
        else:
            return {'image': None, 'quality': None, 'error': f"Invalid image type: {type(img_item)}"}

        quality = calculate_image_quality(raw_img)
        return {'image': raw_img, 'quality': quality, 'error': None}

    def predict_anemia(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        view_types: Optional[Union[str, List[str]]] = None,
        capture_source: str = "upload",
        experimental_override: bool = False,
        gender: str = "prefer_not_to_say"
    ) -> InferenceResult:
        """
        Runs Anemia hemoglobin regression (permitted for research use only).
        Supports single view or multiple views from the same session.
        Enforces camera-domain validation gate for laptop-webcam captures.
        """
        module_name = "anemia"
        warnings_list = [
            "EXPERIMENTAL ESTIMATE: Permitted for research use only. Not established as suitable for clinical use.",
            "Not validated for ordinary laptop-webcam images or uncalibrated ambient lighting conditions.",
            "Within-session standard deviation represents observational variation across anatomical crops, NOT a calibrated confidence interval."
        ]
        limitations_list = [
            "Patient-level test MAE is ~1.50 g/dL (view-level MAE is ~1.78 g/dL). Cannot distinguish borderline anemia without laboratory confirmation.",
            "Model performance relies on clinical color-preservation; variations in flash, white balance, or camera sensors can distort predictions.",
            "Image quality checks (Laplacian sharpness and luminance) assess pixel-level clarity only; they do not verify that ocular conjunctiva or sclera is present or visible."
        ]
        preprocessing_meta = {
            "target_size": [224, 224],
            "aspect_ratio_preserved": True,
            "padding": "center_pad",
            "color_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        }

        # Normalize images input to a list
        if not isinstance(images, list):
            image_list = [images]
        else:
            image_list = images

        if len(image_list) == 0:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error="No images provided for anemia inference."
            )

        # ---------------------------------------------------------
        # Layer 6 Camera-Domain Validation Gate for Webcam Captures
        # ---------------------------------------------------------
        clean_source = capture_source.strip().lower()
        if clean_source == 'webcam' and not experimental_override:
            # Audit input quality for the images provided
            quality_audits = []
            for item in image_list:
                prep = self._prepare_image_input(item)
                if prep['quality']:
                    quality_audits.append(prep['quality'])

            return InferenceResult(
                module=module_name,
                status="blocked_domain_validation",
                input_quality=quality_audits,
                prediction=None,
                unit=None,
                view_count=None,
                view_mean=None,
                view_std=None,
                preprocessing=preprocessing_meta,
                warnings=[
                    "BLOCKED (CAMERA-DOMAIN VALIDATION GATE): The anemia model was trained on clinical anterior-segment research photography and has NOT been validated for ordinary laptop-webcam captures.",
                    "Webcam sensors, auto-exposure, dynamic tone mapping, and compression introduce severe optical domain shift that distorts conjunctival erythema/pallor signals.",
                    "Webcam-to-model inference is blocked by default to prevent presenting unvalidated webcam inferences as hemoglobin measurements.",
                    "To evaluate for laboratory research only, an explicit researcher experimental override is required."
                ],
                limitations=[
                    "A prospective webcam validation protocol is defined in docs/WEBCAM_VALIDATION.md requiring paired laboratory ground truth.",
                    "Never treat webcam-derived outputs as confirmed hemoglobin values or medical diagnoses."
                ],
                error=None
            )

        if clean_source == 'webcam' and experimental_override:
            warnings_list.insert(0, "RESEARCHER EXPERIMENTAL OVERRIDE ACTIVE: Inference executed on uncalibrated laptop-webcam capture.")
            warnings_list.append(
                "UNVALIDATED WEBCAM ESTIMATE: Substantial error is expected due to webcam sensor auto-gain, color distortion, and compression artifacts. "
                "This output is unvalidated and completely unsuitable for medical decisions."
            )
            limitations_list.append(
                "Webcam estimate is completely unvalidated and unsuitable for medical decisions. Paired venous blood reference testing is required."
            )


        # Normalize view_types to a list
        if view_types is None:
            view_list = ['unspecified'] * len(image_list)
        elif isinstance(view_types, str):
            view_list = [view_types] * len(image_list)
        else:
            if len(view_types) != len(image_list):
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=f"Mismatched counts: {len(image_list)} images provided but {len(view_types)} view_types specified."
                )
            view_list = view_types

        # Validate view types against compatible ocular anatomical views
        for v in view_list:
            if v not in COMPATIBLE_ANEMIA_VIEWS:
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=(
                        f"Unsupported anatomical view: '{v}'. "
                        f"Compatible views are: {sorted(list(COMPATIBLE_ANEMIA_VIEWS))}. "
                        "Do not average arbitrary anatomical views unless the trained pipeline supports those views."
                    )
                )

        # Ensure checkpoint can be loaded
        try:
            model = self._get_anemia_model()
        except FileNotFoundError as e:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error=str(e)
            )
        except Exception as e:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error=f"Failed to load anemia model checkpoint: {e}"
            )

        # Process each image
        quality_records = []
        view_predictions = []
        hb_values = []

        for idx, (img_item, v_type) in enumerate(zip(image_list, view_list)):
            prep = self._prepare_image_input(img_item)
            if prep['error'] is not None:
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=f"Image index {idx} failed validation: {prep['error']}"
                )

            raw_img = prep['image']
            q_info = prep['quality']
            quality_records.append(q_info)

            # Framing & anatomical validation check
            framing = verify_ocular_framing_or_detect_face(raw_img, v_type)
            if not framing['is_valid']:
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=f"Framing validation failed for view {idx}: {framing['reason']}"
                )

            # Issue quality warnings if necessary
            if q_info.get('is_blurry'):
                warnings_list.append(f"View {idx} ({v_type}) flagged as potentially blurry (sharpness={q_info['sharpness']}).")
            if q_info.get('is_too_dark'):
                warnings_list.append(f"View {idx} ({v_type}) flagged as too dark (brightness={q_info['brightness']}).")
            if q_info.get('is_too_bright'):
                warnings_list.append(f"View {idx} ({v_type}) flagged as overexposed (brightness={q_info['brightness']}).")

            # Preprocess image
            resized_img = resize_and_pad(raw_img, (224, 224))
            tensor = self._transform(resized_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_val = float(model(tensor).item())

            hb_round = round(pred_val, 2)
            hb_values.append(pred_val)
            chromatic_data = extract_ocular_chromatic_features(raw_img)
            view_predictions.append({
                'view_index': idx,
                'view_type': v_type,
                'hb_pred_g_dl': hb_round,
                'quality': q_info,
                'chromatic': chromatic_data
            })

        # Calculate session aggregation
        view_count = len(hb_values)
        view_mean = round(float(np.mean(hb_values)), 2)
        VIEW_WEIGHTS = {
            'palpebral': 0.45,
            'forniceal_palpebral': 0.30,
            'forniceal': 0.15,
            'raw_eye': 0.10,
            'unspecified': 0.25
        }
        if view_count > 1:
            weights = [VIEW_WEIGHTS.get(v, 0.25) for v in view_list]
            total_w = sum(weights)
            weighted_mean = round(float(sum(w * val for w, val in zip(weights, hb_values)) / total_w), 2)
            view_std = round(float(np.std(hb_values, ddof=1)), 2)
            warnings_list.append(
                f"Aggregated {view_count} session views. Observed variation across views is {view_std} g/dL (reproducibility indicator)."
            )
        else:
            weighted_mean = view_mean
            view_std = 0.0

        clinical_decision = compute_anemia_clinical_decision(
            view_mean,
            gender=gender,
            view_count=view_count
        )
        clinical_decision["weighted_mean_g_dl"] = weighted_mean
        if clinical_decision["diagnostic_certainty"] == "borderline_threshold_equivocal":
            warnings_list.append(clinical_decision["certainty_description_en"])

        for p in view_predictions:
            p['clinical_decision'] = clinical_decision
            p['weighted_mean'] = weighted_mean

        return InferenceResult(
            module=module_name,
            status="available_research_only",
            input_quality=quality_records,
            prediction=view_predictions,
            unit="g/dL",
            view_count=view_count,
            view_mean=view_mean,
            view_std=view_std,
            preprocessing=preprocessing_meta,
            warnings=warnings_list,
            limitations=limitations_list,
            error=None
        )

    def predict_conjunctivitis(
        self,
        images: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None
    ) -> InferenceResult:
        """
        Conjunctivitis module - RELIABILITY GATE ENFORCED.
        Model output is blocked due to the 100% resolution/framing shortcut discovered in Layer 3.5.
        """
        # Audit input quality if images are provided
        quality_records = None
        if images is not None:
            img_list = images if isinstance(images, list) else [images]
            quality_records = []
            for item in img_list:
                prep = self._prepare_image_input(item)
                if prep['quality']:
                    quality_records.append(prep['quality'])

        return InferenceResult(
            module="conjunctivitis",
            status="blocked_validation",
            input_quality=quality_records,
            prediction=None,
            unit=None,
            view_count=None,
            view_mean=None,
            view_std=None,
            preprocessing=None,
            warnings=[
                "BLOCKED: Model validation audit detected a 100% resolution/framing shortcut in the upstream dataset.",
                "Normal images are 100% small square crops (mean 56x56 px), whereas Conjunctivitis images are rectangular macro photos (mean 352x252 px).",
                "A metadata classifier using ONLY image dimensions achieves 100% test accuracy without inspecting any image pixels.",
                "The convolutional model cannot be trusted as a biological hyperemia detector; model output is withheld to prevent presenting non-biological shortcut inferences as disease diagnoses."
            ],
            limitations=[
                "Model artifact is preserved offline for research evaluation (saved_models/conjunctivitis/best_model_v2_clean.pth).",
                "Revalidation requires acquiring matched-resolution anterior eye images with ophthalmologist grading (see docs/RELIABILITY_AND_DOMAIN_VALIDATION.md)."
            ],
            error=None
        )

    def predict_jaundice(
        self,
        images: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None
    ) -> InferenceResult:
        """
        Jaundice module - MISSING-DATA GATE ENFORCED.
        Dataset is absent; placeholder inference or simulated bilirubin is strictly prohibited.
        """
        return InferenceResult(
            module="jaundice",
            status="unavailable_missing_data",
            input_quality=None,
            prediction=None,
            unit=None,
            view_count=None,
            view_mean=None,
            view_std=None,
            preprocessing=None,
            warnings=[
                "UNAVAILABLE: No jaundice or scleral icterus dataset exists in the workspace.",
                "No clinical laboratory total serum bilirubin reference values are available.",
                "Clinical safety rules strictly forbid running placeholder inference, fabricating bilirubin values, or claiming the jaundice model is trained."
            ],
            limitations=[
                "Model architecture (JaundiceClassifier) and preprocessing scaffolds are preserved in src/models/jaundice_model.py.",
                "Clinical data ingestion is required before training or inference can be activated."
            ],
            error=None
        )

    def predict_conjunctival_injection(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        capture_source: str = "upload",
        experimental_override: bool = False
    ) -> InferenceResult:
        """
        Visible Conjunctival Injection / Hyperemia Screening (SLID MobileNetV3-Small).
        Separate from the confounded Mendeley dataset.
        Differentiates Normal ocular surface (Class 0: control_normal) vs. Visible Injection (Class 1: conjunctival_injection).
        """
        module_name = "slid"
        warnings_list = [
            "RESEARCH USE ONLY: Investigational visible conjunctival injection screening.",
            "Screening indicates visible conjunctival hyperemia on slit-lamp anterior ocular photography; it does NOT diagnose infectious etiology (viral, bacterial, or allergic conjunctivitis).",
            "Clinical diagnosis requires biomicroscopic slit-lamp examination by an ophthalmologist."
        ]
        limitations_list = [
            "Validated on SLID slit-lamp dataset (Zhejiang University, Frontiers in Digital Health 2025).",
            "Not certified as an independent medical device.",
            "Webcam captures are subject to camera-domain shifts."
        ]
        preprocessing_meta = {
            "target_size": [224, 224],
            "aspect_ratio_preserved": True,
            "padding": "center_pad",
            "color_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        }

        image_list = images if isinstance(images, list) else [images]
        if len(image_list) == 0:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error="No images provided for conjunctival injection inference."
            )

        clean_source = capture_source.strip().lower()
        if clean_source == 'webcam' and not experimental_override:
            quality_audits = []
            for item in image_list:
                prep = self._prepare_image_input(item)
                if prep['quality']:
                    quality_audits.append(prep['quality'])

            return InferenceResult(
                module=module_name,
                status="blocked_domain_validation",
                input_quality=quality_audits,
                prediction=None,
                unit=None,
                preprocessing=preprocessing_meta,
                warnings=[
                    "BLOCKED (CAMERA-DOMAIN VALIDATION GATE): The conjunctival injection model requires focused slit-lamp or calibrated macro photography.",
                    "Webcam sensors, auto-exposure, and compression introduce severe optical domain shift.",
                    "Webcam inference is blocked by default to prevent presenting unvalidated webcam outputs."
                ],
                limitations=limitations_list,
                error=None
            )

        if clean_source == 'webcam' and experimental_override:
            warnings_list.insert(0, "RESEARCHER EXPERIMENTAL OVERRIDE ACTIVE: Inference executed on uncalibrated laptop-webcam capture.")

        try:
            model = self._get_slid_model()
        except FileNotFoundError as e:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error=str(e)
            )

        quality_records = []
        view_predictions = []
        class_names = ["control_normal", "conjunctival_injection"]

        for idx, img_item in enumerate(image_list):
            prep = self._prepare_image_input(img_item)
            if prep['error'] is not None:
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=f"Image index {idx} failed validation: {prep['error']}"
                )

            raw_img = prep['image']
            q_info = prep['quality']
            quality_records.append(q_info)

            resized_img = resize_and_pad(raw_img, (224, 224))
            tensor = self._transform(resized_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

            p_norm = round(float(probs[0]), 4)
            p_inj = round(float(probs[1]), 4)
            pred_class = class_names[1] if p_inj >= 0.5 else class_names[0]
            conf = max(p_norm, p_inj)
            chromatic_data = extract_ocular_chromatic_features(raw_img)

            view_predictions.append({
                'view_index': idx,
                'predicted_class': pred_class,
                'confidence': conf,
                'probabilities': {
                    'control_normal': p_norm,
                    'conjunctival_injection': p_inj
                },
                'quality': q_info,
                'chromatic': chromatic_data
            })

        mean_inj_prob = round(float(np.mean([p['probabilities']['conjunctival_injection'] for p in view_predictions])), 4)
        session_chromatic = view_predictions[0].get('chromatic') if view_predictions else None
        clinical_decision = compute_slid_clinical_decision(
            predicted_class=view_predictions[0]['predicted_class'],
            confidence=view_predictions[0]['confidence'],
            chromatic=session_chromatic or {}
        )
        for p in view_predictions:
            p['clinical_decision'] = clinical_decision

        return InferenceResult(
            module=module_name,
            status="available_research_only",
            input_quality=quality_records,
            prediction=view_predictions,
            unit="categorical",
            view_count=len(view_predictions),
            view_mean=mean_inj_prob,
            view_std=round(float(np.std([p['probabilities']['conjunctival_injection'] for p in view_predictions])), 4) if len(view_predictions) > 1 else 0.0,
            preprocessing=preprocessing_meta,
            warnings=warnings_list,
            limitations=limitations_list,
            error=None
        )

    def predict_scleral_icterus(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        capture_source: str = "upload",
        experimental_override: bool = False
    ) -> InferenceResult:
        """
        Visible Scleral Icterus / Jaundice Screening (MobileNetV3-Small).
        Categorical screening: Normal (Class 0: normal) vs. Visibly Jaundiced (Class 1: visibly_jaundiced).
        NOTE: Total serum bilirubin estimation is strictly UNAVAILABLE due to missing clinical laboratory ground truth.
        """
        module_name = "jaundice"
        warnings_list = [
            "CATEGORICAL SCLERAL ICTERUS SCREENING ONLY: Permitted for research use under CC BY-NC 4.0 license.",
            "NUMERICAL BILIRUBIN UNAVAILABLE: No paired laboratory total serum bilirubin reference values exist in the workspace.",
            "Under AGENTS.md clinical safety standards, numerical bilirubin values are strictly withheld to avoid generating fabricated biochemical numbers.",
            "Categorical scleral yellowing indicates visible icterus; clinical diagnosis requires laboratory serum bilirubin testing."
        ]
        limitations_list = [
            "Evaluated on Diverse Jaundice Eye Dataset (Daniel Ekong, CC BY-NC 4.0).",
            "Categorical classification only (Normal vs. Visibly Jaundiced). Not quantitative laboratory bilirubin.",
            "Subject to ambient lighting and camera-sensor chromaticity shifts."
        ]
        preprocessing_meta = {
            "target_size": [224, 224],
            "aspect_ratio_preserved": True,
            "padding": "center_pad",
            "color_normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        }

        image_list = images if isinstance(images, list) else [images]
        if len(image_list) == 0:
            return self.predict_jaundice()

        clean_source = capture_source.strip().lower()
        if clean_source == 'webcam' and not experimental_override:
            quality_audits = []
            for item in image_list:
                prep = self._prepare_image_input(item)
                if prep['quality']:
                    quality_audits.append(prep['quality'])

            return InferenceResult(
                module=module_name,
                status="blocked_domain_validation",
                input_quality=quality_audits,
                prediction=None,
                unit=None,
                preprocessing=preprocessing_meta,
                warnings=[
                    "BLOCKED (CAMERA-DOMAIN VALIDATION GATE): Scleral icterus screening requires calibrated macro ocular photography.",
                    "Webcam auto-white-balance artificially shifts yellow-blue chromaticity, distorting scleral icterus detection.",
                    "Webcam-to-model inference is blocked by default."
                ],
                limitations=limitations_list,
                error=None
            )

        if clean_source == 'webcam' and experimental_override:
            warnings_list.insert(0, "RESEARCHER EXPERIMENTAL OVERRIDE ACTIVE: Inference executed on uncalibrated laptop-webcam capture.")

        try:
            model = self._get_jaundice_model()
        except FileNotFoundError as e:
            return InferenceResult(
                module=module_name,
                status="error",
                warnings=warnings_list,
                limitations=limitations_list,
                error=str(e)
            )

        quality_records = []
        view_predictions = []
        class_names = ["normal", "visibly_jaundiced"]

        for idx, img_item in enumerate(image_list):
            prep = self._prepare_image_input(img_item)
            if prep['error'] is not None:
                return InferenceResult(
                    module=module_name,
                    status="error",
                    warnings=warnings_list,
                    limitations=limitations_list,
                    error=f"Image index {idx} failed validation: {prep['error']}"
                )

            raw_img = prep['image']
            q_info = prep['quality']
            quality_records.append(q_info)

            resized_img = resize_and_pad(raw_img, (224, 224))
            tensor = self._transform(resized_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

            p_norm = round(float(probs[0]), 4)
            p_jaund = round(float(probs[1]), 4)
            pred_class = class_names[1] if p_jaund >= 0.5 else class_names[0]
            conf = max(p_norm, p_jaund)
            chromatic_data = extract_ocular_chromatic_features(raw_img)

            view_predictions.append({
                'view_index': idx,
                'predicted_class': pred_class,
                'confidence': conf,
                'probabilities': {
                    'normal': p_norm,
                    'visibly_jaundiced': p_jaund
                },
                'quality': q_info,
                'chromatic': chromatic_data
            })

        mean_jaund_prob = round(float(np.mean([p['probabilities']['visibly_jaundiced'] for p in view_predictions])), 4)
        session_chromatic = view_predictions[0].get('chromatic') if view_predictions else None
        clinical_decision = compute_jaundice_clinical_decision(
            predicted_class=view_predictions[0]['predicted_class'],
            confidence=view_predictions[0]['confidence'],
            chromatic=session_chromatic or {}
        )

        for p in view_predictions:
            p['clinical_decision'] = clinical_decision

        return InferenceResult(
            module=module_name,
            status="available_research_only",
            input_quality=quality_records,
            prediction=view_predictions,
            unit="categorical",
            view_count=len(view_predictions),
            view_mean=mean_jaund_prob,
            view_std=round(float(np.std([p['probabilities']['visibly_jaundiced'] for p in view_predictions])), 4) if len(view_predictions) > 1 else 0.0,
            preprocessing=preprocessing_meta,
            warnings=warnings_list,
            limitations=limitations_list,
            error=None
        )

    def predict_unified(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        view_types: Optional[Union[str, List[str]]] = None,
        capture_source: str = "upload",
        experimental_override: bool = False,
        gender: str = "prefer_not_to_say"
    ) -> Dict[str, Any]:
        """
        Runs unified, multi-target ocular screening across all three trained models:
        1. Model A: Anemia (Continuous Hb Regression, ResNet-50)
        2. Model B: Conjunctival Injection / Hyperemia (SLID, MobileNetV3-Small)
        3. Model C: Scleral Icterus (Diverse Jaundice, MobileNetV3-Small)
        """
        image_list = images if isinstance(images, list) else [images]
        if len(image_list) == 0:
            return {
                "module": "unified",
                "status": "error",
                "error": "No images provided for unified screening.",
                "anemia": None,
                "slid": None,
                "jaundice": None
            }

        anemia_res = self.predict_anemia(
            image_list,
            view_types=view_types,
            capture_source=capture_source,
            experimental_override=experimental_override,
            gender=gender
        )

        slid_res = self.predict_conjunctival_injection(
            image_list,
            capture_source=capture_source,
            experimental_override=experimental_override
        )

        jaund_res = self.predict_scleral_icterus(
            image_list,
            capture_source=capture_source,
            experimental_override=experimental_override
        )

        if anemia_res.get("status") == "blocked_domain_validation" or slid_res.get("status") == "blocked_domain_validation":
            overall_status = "blocked_domain_validation"
        elif anemia_res.get("status") == "available_research_only":
            overall_status = "available_research_only"
        else:
            overall_status = anemia_res.get("status", "unknown")

        all_warnings = list(dict.fromkeys(
            anemia_res.get("warnings", []) + slid_res.get("warnings", []) + jaund_res.get("warnings", [])
        ))
        all_limitations = list(dict.fromkeys(
            anemia_res.get("limitations", []) + slid_res.get("limitations", []) + jaund_res.get("limitations", [])
        ))

        # -------------------------------------------------------------
        # Advanced AI Integration: Segmentation, Secondary Vision,
        # Reference Retrieval, and Calibrated Multi-Target Fusion
        # -------------------------------------------------------------
        first_prep = self._prepare_image_input(image_list[0])
        first_pil_img = first_prep.get("image")
        first_quality = first_prep.get("quality") or {}

        # 1. Ocular Region Segmentation
        seg_data = None
        if first_pil_img is not None:
            try:
                from src.segmentation import get_ocular_segmenter
                seg_obj = get_ocular_segmenter().segment(first_pil_img)
                seg_data = seg_obj.to_dict(include_masks=False)
            except Exception as e:
                seg_data = {"status": "error", "error": str(e), "coverage_pct": {}, "quality_flags": {}}

        # 2. Secondary Visual Analysis (OpenAI Responses / Deterministic Fallback)
        secondary_vision_data = None
        chrom_data = None
        if anemia_res.get("prediction") and len(anemia_res["prediction"]) > 0:
            chrom_data = anemia_res["prediction"][0].get("chromatic")

        if first_pil_img is not None:
            try:
                from src.inference.secondary_vision import get_secondary_vision_service
                secondary_vision_data = get_secondary_vision_service().analyze_image(
                    first_pil_img,
                    chromatic_features=chrom_data,
                    segmentation_info=seg_data
                )
            except Exception as e:
                secondary_vision_data = {"status": "error", "error": str(e)}

        # 3. MedSigLIP-Style Reference Case Retrieval
        ref_retrieval_data = None
        if first_pil_img is not None:
            try:
                from src.inference.reference_retrieval import get_reference_retrieval_service
                ref_retrieval_data = get_reference_retrieval_service(device=self.device).query(
                    first_pil_img,
                    chromatic_features=chrom_data,
                    top_k=3
                )
            except Exception as e:
                ref_retrieval_data = {"status": "error", "error": str(e), "top_matches": []}

        # 4. Calibrated Multi-Modal Ensemble Fusion
        calibrated_ensemble_data = None
        try:
            from src.inference.calibrated_ensemble import get_calibrated_ensemble
            calibrated_ensemble_data = get_calibrated_ensemble().evaluate_fusion(
                anemia_result=anemia_res,
                slid_result=slid_res,
                jaundice_result=jaund_res,
                segmentation_result=seg_data,
                chromatic_features=chrom_data,
                quality_info=first_quality,
                secondary_vision=secondary_vision_data
            )
        except Exception as e:
            calibrated_ensemble_data = {"status": "error", "error": str(e)}

        return {
            "module": "unified",
            "status": overall_status,
            "input_quality": anemia_res.get("input_quality") or slid_res.get("input_quality") or jaund_res.get("input_quality"),
            "prediction": anemia_res.get("prediction"),
            "unit": "g/dL",
            "view_count": anemia_res.get("view_count"),
            "view_mean": anemia_res.get("view_mean"),
            "view_std": anemia_res.get("view_std"),
            "anemia": anemia_res,
            "slid": slid_res,
            "jaundice": jaund_res,
            "segmentation": seg_data,
            "secondary_vision": secondary_vision_data,
            "reference_retrieval": ref_retrieval_data,
            "calibrated_ensemble": calibrated_ensemble_data,
            "warnings": all_warnings,
            "limitations": all_limitations,
            "disclaimer": "RESEARCH USE ONLY: Multi-target ocular screening prototype. Not certified for medical diagnosis or clinical use.",
            "error": anemia_res.get("error") or slid_res.get("error") or jaund_res.get("error")
        }

    def predict(
        self,
        module: str,
        images: Optional[Union[str, Image.Image, List[Union[str, Image.Image]]]] = None,
        view_types: Optional[Union[str, List[str]]] = None,
        capture_source: str = "upload",
        experimental_override: bool = False
    ) -> InferenceResult:
        """
        Common routing gateway for inference requests.
        """
        mod = module.strip().lower()
        if mod in ('unified', 'all', 'multi', 'multi_target', 'unified_ocular_screening'):
            if images is None:
                return InferenceResult(
                    module='unified',
                    status='error',
                    error="Images must be provided for unified ocular screening."
                )
            u_res = self.predict_unified(
                images,
                view_types=view_types,
                capture_source=capture_source,
                experimental_override=experimental_override
            )
            return InferenceResult(**u_res)
        elif mod == 'anemia':
            if images is None:
                return InferenceResult(
                    module='anemia',
                    status='error',
                    error="Images must be provided for anemia prediction."
                )
            return self.predict_anemia(
                images,
                view_types=view_types,
                capture_source=capture_source,
                experimental_override=experimental_override
            )
        elif mod in ('slid', 'conjunctival_injection', 'injection'):
            if images is None:
                return InferenceResult(
                    module='slid',
                    status='error',
                    error="Images must be provided for conjunctival injection prediction."
                )
            return self.predict_conjunctival_injection(
                images,
                capture_source=capture_source,
                experimental_override=experimental_override
            )
        elif mod in ('conjunctivitis', 'mendeley', 'red_eye'):
            return self.predict_conjunctivitis(images)
        elif mod in ('jaundice', 'icterus', 'scleral_icterus'):
            if images is None or (isinstance(images, list) and len(images) == 0):
                return self.predict_jaundice(images)
            return self.predict_scleral_icterus(
                images,
                capture_source=capture_source,
                experimental_override=experimental_override
            )
        else:
            return InferenceResult(
                module=module,
                status="error",
                error=f"Unknown module '{module}'. Supported modules: ['unified', 'anemia', 'slid', 'conjunctivitis', 'jaundice']."
            )


# Module-level singleton helper
_DEFAULT_SERVICE: Optional[EyeHealthInferenceService] = None

def get_inference_service(device: str = 'cpu') -> EyeHealthInferenceService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = EyeHealthInferenceService(device=device)
    return _DEFAULT_SERVICE
