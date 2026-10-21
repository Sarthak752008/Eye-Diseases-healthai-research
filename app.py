"""
app.py
------
EyeHealthAI Research Dashboard & Unified API Server.

Serves the research dashboard and provides REST endpoints for:
- System and module reliability status (/api/status)
- System operational health verification (/api/health)
- Image quality pre-screening (/api/quality-check)
- Unified multi-view inference (/api/analyze)
- Standardized session audit export (/api/export)
"""

import os
import sys
import io
import json
import hashlib
import logging
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image, ImageFile

logger = logging.getLogger("eyehealthai.app")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime, timezone
from sqlalchemy import text

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.config import settings
from src.inference.unified_service import get_inference_service
from src.inference.session_export import create_session_export, validate_session_export_schema
from src.inference.copilot import chat as copilot_chat, get_copilot_status
from src.preprocessing.image_quality import calculate_image_quality
from src.database import (
    init_db,
    register_user,
    update_user_profile_age,
    get_user_by_id,
    get_user_by_email,
    create_session,
    save_model_result,
    get_session_by_id,
    get_user_sessions,
    save_report_metadata,
    delete_user_data,
    delete_session_record,
    SessionLocal
)
from src.inference.pdf_report import build_two_page_pdf

from contextlib import asynccontextmanager

ImageFile.LOAD_TRUNCATED_IMAGES = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema and pre-warms deep learning models to eliminate inference cold-start latency."""
    init_db()
    try:
        service = get_inference_service()
        service._get_anemia_model()
        service._get_slid_model()
        service._get_jaundice_model()
        logger.info("EyeHealthAI inference engines pre-warmed into memory successfully.")
    except Exception as e:
        logger.warning(f"Inference engine pre-warming warning: {e}")
    yield

app = FastAPI(
    title="EyeHealthAI Research Dashboard API",
    description="Multi-modal ocular computer vision screening pipeline with reliability-gated integration.",
    version="1.0.0",
    lifespan=lifespan
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response

# CORS Configuration
# In production, restrict to allowed origins; allow wildcard in development if configured
cors_origins = ["*"] if ("*" in settings.ALLOWED_ORIGINS or settings.ENV == "development") else settings.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class NoCacheStaticFiles(StaticFiles):
    """Static file handler that ensures browser doesn't cache JS/CSS during development."""
    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

# Mount static assets
STATIC_DIR = os.path.join(WORKSPACE_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


def validate_uploaded_image(file: UploadFile, content: bytes) -> None:
    """Validates file size, file extension, and MIME type."""
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail=f"Image file '{file.filename}' is empty.")

    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext and ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}' for file '{file.filename}'. Allowed extensions: {sorted(list(settings.ALLOWED_IMAGE_EXTENSIONS))}."
            )

    # Check Content-Type if present
    if file.content_type:
        mime = file.content_type.lower()
        if mime not in settings.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported MIME type '{file.content_type}'. Allowed image MIME types: {sorted(list(settings.ALLOWED_MIME_TYPES))}."
            )


def safe_decode_bytes(content: bytes) -> Optional[Image.Image]:
    """Decodes image bytes safely in-memory preserving clinical color & alpha masking."""
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.load()
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (0, 0, 0))
                rgb_img.paste(img, mask=img.split()[3])
                return rgb_img
            elif img.mode != 'RGB':
                return img.convert('RGB')
            else:
                return img.copy()
    except Exception:
        return None


@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serves the primary research dashboard HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(
        index_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/report", response_class=FileResponse)
async def serve_screening_report():
    """Serves the pixel-perfect Ocular Health Screening Report page."""
    report_path = os.path.join(STATIC_DIR, "report.html")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report UI not found")
    return FileResponse(
        report_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Lightweight operational health endpoint.
    Reports API and service status while explicitly distinguishing
    application health from clinical model and dataset readiness.
    """
    anemia_ckpt = os.path.exists(settings.ANEMIA_CHECKPOINT_PATH)
    conj_ckpt = os.path.exists(settings.CONJUNCTIVITIS_CHECKPOINT_PATH)
    slid_ckpt = os.path.exists(settings.SLID_CHECKPOINT_PATH)
    jaundice_ckpt = os.path.exists(settings.JAUNDICE_CHECKPOINT_PATH)
    jaundice_cfg = os.path.exists(settings.JAUNDICE_CONFIG_PATH)
    cataract_ckpt = os.path.exists(settings.CATARACT_CHECKPOINT_PATH)
    stye_ckpt = os.path.exists(settings.STYE_CHECKPOINT_PATH)

    return {
        "status": "healthy",
        "service": "EyeHealthAI Research Service",
        "version": "1.0.0",
        "environment": settings.ENV,
        "device": settings.DEVICE,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        "modules": {
            "anemia": {
                "checkpoint_present": anemia_ckpt,
                "operational_status": "available_research_only" if anemia_ckpt else "missing_checkpoint",
                "validation_status": "experimental_research_only",
                "notes": "Non-diagnostic research regressor. Paired venous reference testing required."
            },
            "slid_injection": {
                "checkpoint_present": slid_ckpt,
                "operational_status": "available_research_only" if slid_ckpt else "missing_checkpoint",
                "validation_status": "experimental_research_only",
                "notes": "Visible conjunctival injection screening on slit-lamp photography (SLID dataset)."
            },
            "conjunctivitis": {
                "checkpoint_present": conj_ckpt,
                "operational_status": "blocked_validation",
                "validation_status": "shortcut_confound_blocked",
                "notes": "Predictions withheld due to 100% resolution/framing dataset shortcut."
            },
            "jaundice": {
                "checkpoint_present": False,
                "operational_status": "unavailable_missing_data",
                "validation_status": "unimplemented_no_ground_truth",
                "notes": "No clinical dataset or laboratory bilirubin reference measurements exist."
            },
            "jaundice_icterus": {
                "checkpoint_present": jaundice_ckpt,
                "operational_status": "available_research_only" if jaundice_ckpt else "missing_checkpoint",
                "validation_status": "experimental_research_only",
                "notes": "Visible scleral icterus screening (Diverse Jaundice dataset). Numerical lab bilirubin unavailable."
            },
            "cataract": {
                "checkpoint_present": cataract_ckpt,
                "operational_status": "available_research_only" if cataract_ckpt else "missing_checkpoint",
                "validation_status": "experimental_research_only",
                "notes": "Nuclear cataract and LOCS III opacification screening on slit-lamp biomicroscopy."
            },
            "stye": {
                "checkpoint_present": stye_ckpt,
                "operational_status": "available_research_only" if stye_ckpt else "missing_checkpoint",
                "validation_status": "experimental_research_only",
                "notes": "Acute eyelid margin lesion and hordeolum detection on anterior ocular photography."
            }
        },
        "disclaimer": "RESEARCH USE ONLY: Prototype service. Not certified or validated for clinical diagnostic use."
    }


@app.get("/ready")
@app.get("/api/ready")
async def readiness_check():
    """
    Readiness probe for production orchestrators.
    Distinguishes operational server/database health from clinical model validation gates.
    """
    db_connected = False
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            db_connected = True
    except Exception:
        db_connected = False

    anemia_ckpt = os.path.exists(settings.ANEMIA_CHECKPOINT_PATH)
    conj_ckpt = os.path.exists(settings.CONJUNCTIVITIS_CHECKPOINT_PATH)
    slid_ckpt = os.path.exists(settings.SLID_CHECKPOINT_PATH)
    jaundice_ckpt = os.path.exists(settings.JAUNDICE_CHECKPOINT_PATH)

    # Core research prototype is ready when DB is connected and Anemia checkpoint is present
    is_ready = db_connected and anemia_ckpt

    return {
        "ready": is_ready,
        "database_connected": db_connected,
        "checkpoints": {
            "anemia_resnet50": anemia_ckpt,
            "slid_mobilenet": slid_ckpt,
            "conjunctivitis_mobilenet": conj_ckpt,
            "jaundice_mobilenet": jaundice_ckpt
        },
        "gates": {
            "conjunctivitis": "blocked_domain_validation",
            "jaundice": "unavailable_missing_data",
            "conjunctivitis_mendeley": "blocked_domain_validation",
            "jaundice_laboratory_bilirubin": "unavailable_missing_data"
        },
        "status": "ready" if is_ready else "degraded",
        "service": "EyeHealthAI Research Backend"
    }


@app.get("/api/status")
async def get_system_status():
    """Returns operational status for all clinical modules and checkpoints."""
    anemia_ckpt = os.path.exists(settings.ANEMIA_CHECKPOINT_PATH)
    conj_ckpt = os.path.exists(settings.CONJUNCTIVITIS_CHECKPOINT_PATH)
    slid_ckpt = os.path.exists(settings.SLID_CHECKPOINT_PATH)
    jaundice_ckpt = os.path.exists(settings.JAUNDICE_CHECKPOINT_PATH)

    return {
        "project": "EyeHealthAI",
        "system_version": "Deploy-Ready Research Prototype",
        "modules": {
            "anemia": {
                "name": "Anemia Hemoglobin Regressor",
                "status": "available_research_only",
                "model_architecture": "ResNet-50",
                "checkpoint_available": anemia_ckpt,
                "target_unit": "g/dL",
                "permitted_use": "Investigational research only (multi-view session aggregation)"
            },
            "slid_injection": {
                "name": "SLID Visible Conjunctival Injection Classifier",
                "status": "available_research_only",
                "model_architecture": "MobileNetV3-Small",
                "checkpoint_available": slid_ckpt,
                "target_classes": ["control_normal", "conjunctival_injection"],
                "permitted_use": "Investigational research only on slit-lamp photography (Acc 98.78%)"
            },
            "conjunctivitis": {
                "name": "Conjunctivitis Classifier",
                "status": "blocked_validation",
                "model_architecture": "MobileNetV3-Small",
                "checkpoint_available": conj_ckpt,
                "gate_reason": "100% resolution/framing confound between dataset classes. Raw predictions withheld."
            },
            "jaundice": {
                "name": "Jaundice Scleral Icterus Classifier",
                "status": "unavailable_missing_data",
                "model_architecture": "MobileNetV3-Small (Scaffold)",
                "checkpoint_available": False,
                "gate_reason": "No clinical dataset or laboratory bilirubin reference measurements exist."
            },
            "jaundice_icterus": {
                "name": "Visible Scleral Icterus Classifier",
                "status": "available_research_only",
                "model_architecture": "MobileNetV3-Small",
                "checkpoint_available": jaundice_ckpt,
                "target_classes": ["normal", "visibly_jaundiced"],
                "permitted_use": "Categorical scleral icterus research only (CC BY-NC 4.0; no lab bilirubin)"
            }
        },
        "disclaimer": "RESEARCH USE ONLY: Experimental system. Not for medical diagnosis or clinical use."
    }


@app.post("/api/quality-check")
async def quality_check(image: UploadFile = File(...)):
    """
    Pre-screens an uploaded or webcam-captured image for sharpness, exposure, dimensions,
    and returns actionable guidance.
    """
    try:
        content = await image.read()
        validate_uploaded_image(image, content)

        rgb_img = safe_decode_bytes(content)
        if rgb_img is None:
            return {
                "filename": image.filename,
                "dimensions": None,
                "quality": None,
                "is_valid": False,
                "error": "Corrupt or unreadable image bytes."
            }
        from src.preprocessing.advanced_pipeline import process_ocular_image_advanced
        proc = process_ocular_image_advanced(rgb_img, generate_previews=True)
        q_gate = proc["quality_gate"]
        det = proc["detection"]
        enh = proc["enhancement"]
        prev = proc["previews"]
        q_metrics = q_gate.get("metrics", {})
        q_flags = q_gate.get("flags", {})

        return {
            "filename": image.filename,
            "dimensions": {"width": rgb_img.width, "height": rgb_img.height},
            "crop_dimensions": {"width": proc["processed_image"].width, "height": proc["processed_image"].height},
            "quality": {
                "sharpness": q_metrics.get("sharpness_laplacian", 0.0),
                "brightness": q_metrics.get("mean_luminance", 0.0),
                "contrast": q_metrics.get("contrast_std", 0.0),
                "is_blurry": q_flags.get("is_blurry", False),
                "is_too_dark": q_flags.get("is_underexposed", False),
                "is_too_bright": q_flags.get("is_overexposed", False),
                "is_too_distant": q_flags.get("is_too_distant", False)
            },
            "quality_gate": q_gate,
            "ocular_detection": det,
            "enhancement": enh,
            "previews": prev,
            "usability_status": "usable" if q_gate.get("is_suitable", True) else "needs_adjustment",
            "actionable_guidance": (q_gate.get("retake_guidance_en", []) + ([det.get("guidance_en")] if det.get("guidance_en") else [])) or ["Optimal image quality and illumination verified."],
            "actionable_guidance_hi": (q_gate.get("retake_guidance_hi", []) + ([det.get("guidance_hi")] if det.get("guidance_hi") else [])) or ["छवि की गुणवत्ता और रोशनी उपयुक्त पाई गई।"],
            "anatomical_disclaimer": "Image-quality check verifies pixel sharpness, lighting uniformity, and anterior ocular framing. Automatic ocular detection locates the anterior eye region and applies conservative, color-safe illumination enhancement. It does not replace clinical in-person slit-lamp or laboratory testing.",
            "is_valid": True
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "filename": image.filename,
            "dimensions": None,
            "quality": None,
            "ocular_detection": None,
            "is_valid": False,
            "error": "Image processing error: unable to decode or inspect image."
        }


@app.post("/api/detect-eye")
async def detect_eye_endpoint(image: UploadFile = File(...)):
    """
    Dedicated endpoint for automated ocular region detection and eye-focused cropping.
    """
    content = await image.read()
    validate_uploaded_image(image, content)
    rgb_img = safe_decode_bytes(content)
    if rgb_img is None:
        raise HTTPException(status_code=400, detail="Corrupt or unreadable image bytes.")

    from src.preprocessing.eye_detector import detect_ocular_region
    detection = detect_ocular_region(rgb_img)
    return {
        "filename": image.filename,
        "dimensions": {"width": rgb_img.width, "height": rgb_img.height},
        "detection": detection
    }


@app.post("/api/analyze")
async def analyze_ocular_images(
    module: str = Form("anemia"),
    images: Union[List[UploadFile], UploadFile, None] = File(None),
    image: Optional[UploadFile] = File(None),
    view_types: Union[List[str], str, None] = Form(None),
    capture_source: str = Form("upload"),
    experimental_override: bool = Form(False),
    is_experimental_override: Optional[bool] = Form(None),
    user_id: Optional[str] = Form(None),
    age_years: Optional[int] = Form(None),
    clinical_sex: Optional[str] = Form(None),
    is_pregnant: Optional[bool] = Form(None),
    pregnancy_trimester: Optional[int] = Form(None),
    external_ai_consent: Optional[bool] = Form(None),
    language: str = Form("en")
):
    """
    Unified multi-view inference endpoint.
    Processes images strictly in-memory without persistent disk storage.
    Enforces camera-domain validation gates for laptop webcam inputs.
    Persists session metadata and results into the database if requested.
    """
    if is_experimental_override is not None:
        experimental_override = bool(experimental_override or is_experimental_override)

    # Normalize images into a List[UploadFile]
    if images is None:
        images_list = [image] if image is not None else []
    elif isinstance(images, list):
        images_list = list(images)
        if image is not None and image not in images_list:
            images_list.append(image)
    else:
        images_list = [images]
        if image is not None and image not in images_list:
            images_list.append(image)
    images = images_list

    # Normalize view_types into a List[str]
    if view_types is None:
        view_types = []
    elif not isinstance(view_types, list):
        view_types = [view_types]

    service = get_inference_service()
    clean_module = module.strip().lower()
    clean_source = capture_source.strip().lower()

    # 1. Validation & Empty Input Handling
    SUPPORTED_MODULE_ALIASES = {
        'unified', 'all', 'multi', 'multi_target', 'unified_ocular_screening',
        'anemia', 'hb', 'hemoglobin',
        'slid', 'conjunctival_injection', 'injection',
        'conjunctivitis', 'mendeley', 'red_eye',
        'jaundice', 'icterus', 'scleral_icterus',
        'cataract', 'nuclear_cataract',
        'stye', 'hordeolum', 'stye_hordeolum'
    }
    if clean_module not in SUPPORTED_MODULE_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown module '{module}'. Supported modules: ['unified', 'anemia', 'slid', 'conjunctivitis', 'jaundice', 'cataract', 'stye']."
        )

    # Empty image submissions trigger explicit diagnostic gate audits
    if not images or len(images) == 0:
        if clean_module in ('conjunctivitis', 'mendeley', 'red_eye'):
            res = service.predict_conjunctivitis()
            export_payload = create_session_export(
                module="conjunctivitis",
                inference_result=res,
                filenames=[],
                capture_source=clean_source
            )
            _persist_db_session(export_payload, clean_module, clean_source, user_id, language, [], res.get("gate_reason"))
            return export_payload
        elif clean_module in ('jaundice', 'icterus', 'scleral_icterus'):
            res = service.predict_jaundice()
            export_payload = create_session_export(
                module="jaundice",
                inference_result=res,
                filenames=[],
                capture_source=clean_source
            )
            _persist_db_session(export_payload, clean_module, clean_source, user_id, language, [], res.get("gate_reason"))
            return export_payload
        elif clean_module in ('cataract', 'nuclear_cataract'):
            res = service.predict_cataract([])
            export_payload = create_session_export(
                module="cataract",
                inference_result=res,
                filenames=[],
                capture_source=clean_source
            )
            _persist_db_session(export_payload, clean_module, clean_source, user_id, language, [], res.get("error"))
            return export_payload
        elif clean_module in ('stye', 'hordeolum', 'stye_hordeolum'):
            res = service.predict_stye([])
            export_payload = create_session_export(
                module="stye",
                inference_result=res,
                filenames=[],
                capture_source=clean_source
            )
            _persist_db_session(export_payload, clean_module, clean_source, user_id, language, [], res.get("error"))
            return export_payload
        else:
            raise HTTPException(status_code=400, detail=f"No images provided for {clean_module} analysis.")

    # Legacy Mendeley dataset shortcut is strictly blocked across all capture sources per AGENTS.md
    if clean_module in ('conjunctivitis', 'mendeley', 'red_eye'):
        res = service.predict_conjunctivitis(images=[])
        export_payload = create_session_export(
            module="conjunctivitis",
            inference_result=res,
            filenames=[f.filename for f in images],
            capture_source=clean_source
        )
        _persist_db_session(export_payload, clean_module, clean_source, user_id, language, [], res.get("gate_reason"))
        return export_payload

    # -----------------------------------------------------------------------
    # Centralized Complete Multi-Target Evaluation Pipeline
    # Runs whenever genuine images are submitted to any eligible module.
    # -----------------------------------------------------------------------
    pil_images = []
    filenames = []
    image_hashes = []
    quality_records = []

    for idx, f in enumerate(images):
        fname = f.filename or f"image_{idx}.jpg"
        filenames.append(fname)
        content = await f.read()
        validate_uploaded_image(f, content)
        img_h = hashlib.sha256(content).hexdigest()
        image_hashes.append(img_h)
        logger.info(f"Ingested image #{idx+1} ({fname}): SHA-256={img_h}, Size={len(content)} bytes")

        rgb_img = safe_decode_bytes(content)
        if rgb_img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode image #{idx} ({fname}). Corrupt file bytes.")
        pil_images.append(rgb_img)

        # Calculate genuine, un-faked image quality measurements
        q_info = calculate_image_quality(rgb_img)
        quality_records.append(q_info)

    # Normalize view types
    v_types = view_types if view_types else ['unspecified'] * len(pil_images)
    if len(v_types) != len(pil_images):
        if len(v_types) == 1:
            v_types = v_types * len(pil_images)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Mismatched counts: {len(pil_images)} images and {len(v_types)} view types."
            )

    user_gender = "prefer_not_to_say"
    user_age = age_years
    user_csex = clinical_sex
    user_preg = is_pregnant
    user_trim = pregnancy_trimester
    user_ai_consent = external_ai_consent

    if user_id and str(user_id).strip().lower() not in ("none", "null", "undefined", ""):
        with SessionLocal() as db:
            u = get_user_by_id(db, str(user_id).strip())
            if not u:
                raise HTTPException(
                    status_code=404,
                    detail=f"User profile '{user_id}' not found. Please register before starting screening."
                )
            if u.gender:
                user_gender = u.gender
            if user_age is None:
                user_age = u.age_years
            if user_csex is None and u.clinical_sex:
                user_csex = u.clinical_sex
            if user_preg is None and u.is_pregnant is not None:
                user_preg = u.is_pregnant
            if user_trim is None and u.pregnancy_trimester is not None:
                user_trim = u.pregnancy_trimester
            if user_ai_consent is None and u.external_ai_consent is not None:
                user_ai_consent = u.external_ai_consent

    # Profile validation: Age is mandatory before starting an analysis
    if user_id and user_age is None:
        raise HTTPException(
            status_code=422,
            detail="Mandatory profile field 'age_years' is missing for this account. Please update patient age (0-120) before starting screening."
        )

    # Validate age integer bounds if provided
    if user_age is not None:
        if isinstance(user_age, bool) or not isinstance(user_age, int) or user_age < 0 or user_age > 120:
            raise HTTPException(
                status_code=422,
                detail=f"Age must be a valid integer between 0 and 120 years. Received: {user_age}"
            )

    # Execute complete multi-target pipeline across all available clinical modules
    inference_res = service.predict_unified(
        pil_images,
        view_types=v_types,
        capture_source=clean_source,
        experimental_override=experimental_override,
        gender=user_gender,
        age_years=user_age,
        clinical_sex=user_csex,
        is_pregnant=bool(user_preg),
        pregnancy_trimester=user_trim,
        external_ai_consent=bool(user_ai_consent)
    )

    if inference_res.get('status') == 'error':
        raise HTTPException(status_code=400, detail=inference_res.get('error'))

    # Select module-specific primary output while retaining all other findings
    anemia_res = inference_res.get("anemia") or {}
    slid_res = inference_res.get("slid") or {}
    jaund_res = inference_res.get("jaundice") or {}
    cataract_res = inference_res.get("cataract") or {}
    stye_res = inference_res.get("stye") or {}

    if clean_module in ('slid', 'conjunctival_injection', 'injection'):
        chosen_output = slid_res.get("prediction")
        ckpt_id = "MobileNetV3_saved_models_slid_best_model.pth"
    elif clean_module in ('jaundice', 'icterus', 'scleral_icterus'):
        chosen_output = jaund_res.get("prediction")
        ckpt_id = "MobileNetV3_saved_models_jaundice_best_model.pth"
    elif clean_module in ('cataract', 'nuclear_cataract'):
        chosen_output = cataract_res.get("prediction")
        ckpt_id = "MobileNetV3_saved_models_cataract_best_model.pth"
    elif clean_module in ('stye', 'hordeolum', 'stye_hordeolum'):
        chosen_output = stye_res.get("prediction")
        ckpt_id = "MobileNetV3_saved_models_stye_best_model.pth"
    elif clean_module in ('conjunctivitis', 'mendeley', 'red_eye'):
        chosen_output = slid_res.get("prediction")
        ckpt_id = "SLID_Remediated_MobileNetV3_slid_best_model.pth"
    else:
        chosen_output = anemia_res.get("prediction")
        ckpt_id = "Unified_MultiTarget_ResNet50_MobileNetV3"

    export_payload = create_session_export(
        module=clean_module,
        inference_result=inference_res,
        filenames=filenames,
        checkpoint_id=ckpt_id,
        capture_source=clean_source
    )

    # Attach complete multi-target results to ensure every component appears in dashboard & reports
    export_payload["selected_module"] = clean_module
    export_payload["operational_status"] = inference_res.get("status")
    export_payload["model_output"] = chosen_output
    export_payload["input_hashes"] = image_hashes
    export_payload["input_provenance"] = [
        {"filename": fn, "sha256": h} for fn, h in zip(filenames, image_hashes)
    ]
    if isinstance(slid_res, dict) and "raw_prediction" not in slid_res:
        slid_res["raw_prediction"] = slid_res.get("prediction")
    if isinstance(jaund_res, dict) and "raw_prediction" not in jaund_res:
        jaund_res["raw_prediction"] = jaund_res.get("prediction")
    if isinstance(anemia_res, dict) and "raw_prediction" not in anemia_res:
        anemia_res["raw_prediction"] = anemia_res.get("prediction")
    if isinstance(cataract_res, dict) and "raw_prediction" not in cataract_res:
        cataract_res["raw_prediction"] = cataract_res.get("prediction")
    if isinstance(stye_res, dict) and "raw_prediction" not in stye_res:
        stye_res["raw_prediction"] = stye_res.get("prediction")

    export_payload["anemia"] = anemia_res
    export_payload["anemia_result"] = anemia_res
    export_payload["slid"] = slid_res
    export_payload["slid_result"] = slid_res
    export_payload["jaundice"] = jaund_res
    export_payload["jaundice_result"] = jaund_res
    export_payload["cataract"] = cataract_res
    export_payload["cataract_result"] = cataract_res
    export_payload["stye"] = stye_res
    export_payload["stye_result"] = stye_res
    export_payload["model_results"] = {
        "anemia": anemia_res,
        "slid": slid_res,
        "jaundice": jaund_res,
        "cataract": cataract_res,
        "stye": stye_res
    }
    export_payload["segmentation"] = inference_res.get("segmentation")
    export_payload["secondary_vision"] = inference_res.get("secondary_vision")
    export_payload["reference_retrieval"] = inference_res.get("reference_retrieval")
    export_payload["calibrated_ensemble"] = inference_res.get("calibrated_ensemble")
    export_payload["module_execution_audit"] = inference_res.get("module_execution_audit")
    export_payload["mendeley_gate"] = inference_res.get("mendeley_gate")
    export_payload["serum_bilirubin_gate"] = inference_res.get("serum_bilirubin_gate")
    export_payload["personalized_ai"] = inference_res.get("personalized_ai")
    export_payload["patient_age_years"] = user_age
    export_payload["patient_clinical_sex"] = user_csex or user_gender
    export_payload["is_pregnant"] = bool(user_preg)
    export_payload["pregnancy_trimester"] = user_trim
    export_payload["external_ai_consent"] = bool(user_ai_consent)

    # Preprocessing Pipeline & Previews (Eye Detection & Conservative Enhancement)
    prep_pipe = inference_res.get("preprocessing_pipeline") or {}
    q_gate = prep_pipe.get("quality_gate") or {}
    q_metrics = q_gate.get("metrics") or {}
    q_flags = q_gate.get("flags") or {}

    export_payload["preprocessing_pipeline"] = prep_pipe
    export_payload["previews"] = inference_res.get("previews") or {}
    export_payload["image_sequence"] = inference_res.get("image_sequence") or {}
    export_payload["visualizations"] = inference_res.get("visualizations") or {}

    # Real calculated quality summary (post-crop & post-enhancement)
    first_q = quality_records[0] if quality_records else {}
    export_payload["input_quality"] = quality_records
    export_payload["quality_summary"] = {
        "laplacian_var": q_metrics.get("sharpness_laplacian", first_q.get("sharpness", 0.0)),
        "luminance_mean": q_metrics.get("mean_luminance", first_q.get("brightness", 0.0)),
        "contrast_michelson": first_q.get("contrast_michelson", 0.0),
        "is_blurry": q_flags.get("is_blurry", first_q.get("is_blurry", False)),
        "is_underexposed": q_flags.get("is_underexposed", first_q.get("is_too_dark", False)),
        "is_overexposed": q_flags.get("is_overexposed", first_q.get("is_too_bright", False)),
        "resolution": q_metrics.get("effective_crop_resolution", first_q.get("resolution", [0, 0])),
        "original_resolution": q_metrics.get("original_resolution", first_q.get("resolution", [0, 0])),
        "detection_mode": (prep_pipe.get("detection") or {}).get("detection_mode", "direct_macro"),
        "eye_side": (prep_pipe.get("detection") or {}).get("eye_side", "unspecified"),
        "enhancement_steps": (prep_pipe.get("enhancement") or {}).get("steps", []),
        "segmentation": inference_res.get("segmentation"),
        "secondary_vision": inference_res.get("secondary_vision"),
        "reference_retrieval": inference_res.get("reference_retrieval"),
        "calibrated_ensemble": inference_res.get("calibrated_ensemble"),
        "image_sequence": inference_res.get("image_sequence"),
        "visualizations": inference_res.get("visualizations"),
        "previews": inference_res.get("previews")
    }

    if clean_module in ('conjunctivitis', 'mendeley', 'red_eye'):
        export_payload.setdefault("warnings", []).append(
            "REMEDIATED CLINICAL ROUTE: The legacy Mendeley dataset model is decommissioned due to resolution shortcut bias. Active screening is powered by the verified SLID Clinical Slit-Lamp Hyperemia Model (Acc: 98.78%)."
        )

    _persist_db_session(
        export_payload,
        clean_module,
        clean_source,
        user_id,
        language,
        v_types,
        inference_res.get("gate_reason")
    )
    return export_payload


def _persist_db_session(export_payload: dict, module: str, capture_source: str, user_id: Optional[str], language: str, view_types: list, gate_reason: Optional[str]):
    """Internal helper to save session audit log and real multi-module outputs to database with strict transaction integrity."""
    with SessionLocal() as db:
        valid_user_id = None
        if user_id and str(user_id).strip().lower() not in ("none", "null", "undefined", ""):
            user = get_user_by_id(db, str(user_id).strip())
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail=f"User profile '{user_id}' not found. Please register before starting screening."
                )
            valid_user_id = user.id

        try:
            q_sum = {
                "status": "processed",
                "segmentation": export_payload.get("segmentation"),
                "secondary_vision": export_payload.get("secondary_vision"),
                "reference_retrieval": export_payload.get("reference_retrieval"),
                "calibrated_ensemble": export_payload.get("calibrated_ensemble"),
                "input_provenance": export_payload.get("input_provenance"),
                "module_execution_audit": export_payload.get("module_execution_audit"),
                "quality_summary": export_payload.get("quality_summary"),
                "quality_metrics": export_payload.get("input_quality"),
                "image_sequence": export_payload.get("image_sequence"),
                "visualizations": export_payload.get("visualizations"),
                "previews": export_payload.get("previews")
            }
            db_sess = create_session(
                db=db,
                selected_module=module,
                capture_source=capture_source,
                user_id=valid_user_id,
                language=language or "en",
                view_types=view_types,
                quality_summary=q_sum,
                patient_age_years=export_payload.get("patient_age_years"),
                patient_clinical_sex=export_payload.get("patient_clinical_sex"),
                ai_explanation=export_payload.get("personalized_ai"),
                session_id=export_payload.get("session_id")
            )

            # Persist all available evaluated modules for comprehensive clinical provenance
            anemia_data = export_payload.get("anemia") or export_payload.get("anemia_result") or {}
            slid_data = export_payload.get("slid") or export_payload.get("slid_result") or {}
            jaund_data = export_payload.get("jaundice") or export_payload.get("jaundice_result") or {}

            # 1. Anemia Result
            if anemia_data and anemia_data.get("prediction") is not None:
                anemia_pred = anemia_data.get("prediction")
                anemia_val = anemia_data.get("view_mean")
                if anemia_val is None and isinstance(anemia_pred, list) and len(anemia_pred) > 0 and isinstance(anemia_pred[0], dict):
                    anemia_val = anemia_pred[0].get("hb_pred_g_dl")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module="anemia",
                    operational_status=anemia_data.get("status", "unknown"),
                    prediction_value=anemia_val,
                    unit="g/dL",
                    view_count=anemia_data.get("view_count"),
                    view_mean=anemia_data.get("view_mean"),
                    view_std=anemia_data.get("view_std"),
                    reliability_warnings=anemia_data.get("warnings", []),
                    documented_limitations=anemia_data.get("limitations", []),
                    gate_reason=anemia_data.get("gate_reason"),
                    raw_prediction=anemia_pred
                )

            # 2. SLID Hyperemia Result
            if slid_data and slid_data.get("prediction") is not None:
                slid_pred = slid_data.get("prediction") or []
                slid_conf = None
                if isinstance(slid_pred, list) and len(slid_pred) > 0 and isinstance(slid_pred[0], dict):
                    slid_conf = slid_pred[0].get("confidence")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module="slid",
                    operational_status=slid_data.get("status", "unknown"),
                    prediction_value=slid_conf,
                    unit="categorical",
                    view_count=slid_data.get("view_count"),
                    view_mean=slid_data.get("view_mean"),
                    view_std=slid_data.get("view_std"),
                    reliability_warnings=slid_data.get("warnings", []),
                    documented_limitations=slid_data.get("limitations", []),
                    gate_reason=slid_data.get("gate_reason"),
                    raw_prediction=slid_pred
                )

            # 3. Jaundice / Scleral Icterus Result
            if jaund_data and jaund_data.get("prediction") is not None:
                jaund_pred = jaund_data.get("prediction") or []
                jaund_conf = None
                if isinstance(jaund_pred, list) and len(jaund_pred) > 0 and isinstance(jaund_pred[0], dict):
                    jaund_conf = jaund_pred[0].get("confidence")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module="jaundice",
                    operational_status=jaund_data.get("status", "unknown"),
                    prediction_value=jaund_conf,
                    unit="categorical",
                    view_count=jaund_data.get("view_count"),
                    view_mean=jaund_data.get("view_mean"),
                    view_std=jaund_data.get("view_std"),
                    reliability_warnings=jaund_data.get("warnings", []),
                    documented_limitations=jaund_data.get("limitations", []),
                    gate_reason=jaund_data.get("gate_reason"),
                    raw_prediction=jaund_pred
                )

            # 4. Nuclear Cataract Result
            cataract_data = export_payload.get("cataract") or export_payload.get("cataract_result") or {}
            if cataract_data and (cataract_data.get("prediction") is not None or cataract_data.get("status") is not None):
                cat_pred = cataract_data.get("prediction") or []
                cat_conf = None
                if isinstance(cat_pred, list) and len(cat_pred) > 0 and isinstance(cat_pred[0], dict):
                    cat_conf = cat_pred[0].get("confidence") or cat_pred[0].get("locs_grade_estimate")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module="cataract",
                    operational_status=cataract_data.get("status", "unknown"),
                    prediction_value=cat_conf,
                    unit="categorical",
                    view_count=cataract_data.get("view_count"),
                    view_mean=cataract_data.get("view_mean"),
                    view_std=cataract_data.get("view_std"),
                    reliability_warnings=cataract_data.get("warnings", []),
                    documented_limitations=cataract_data.get("limitations", []),
                    gate_reason=cataract_data.get("gate_reason"),
                    raw_prediction=cat_pred
                )

            # 5. Stye / Hordeolum Result
            stye_data = export_payload.get("stye") or export_payload.get("stye_result") or {}
            if stye_data and (stye_data.get("prediction") is not None or stye_data.get("status") is not None):
                stye_pred = stye_data.get("prediction") or []
                stye_conf = None
                if isinstance(stye_pred, list) and len(stye_pred) > 0 and isinstance(stye_pred[0], dict):
                    stye_conf = stye_pred[0].get("confidence")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module="stye",
                    operational_status=stye_data.get("status", "unknown"),
                    prediction_value=stye_conf,
                    unit="categorical",
                    view_count=stye_data.get("view_count"),
                    view_mean=stye_data.get("view_mean"),
                    view_std=stye_data.get("view_std"),
                    reliability_warnings=stye_data.get("warnings", []),
                    documented_limitations=stye_data.get("limitations", []),
                    gate_reason=stye_data.get("gate_reason"),
                    raw_prediction=stye_pred
                )

            # If none of the 5 structured modules matched, persist single-module fallback
            if not anemia_data.get("prediction") and not slid_data.get("prediction") and not jaund_data.get("prediction") and not cataract_data.get("prediction") and not stye_data.get("prediction"):
                pred_val = export_payload.get("view_mean")
                if pred_val is None:
                    raw_out = export_payload.get("model_output")
                    if isinstance(raw_out, list) and len(raw_out) > 0 and isinstance(raw_out[0], dict):
                        pred_val = raw_out[0].get("hb_pred_g_dl") or raw_out[0].get("confidence")
                    elif isinstance(raw_out, (int, float)):
                        pred_val = float(raw_out)

                raw_p = export_payload.get("raw_prediction") or export_payload.get("model_output") or export_payload.get("prediction")
                save_model_result(
                    db=db,
                    session_id=db_sess.id,
                    module=module,
                    operational_status=export_payload.get("operational_status", "unknown"),
                    prediction_value=pred_val,
                    unit=export_payload.get("unit"),
                    view_count=export_payload.get("view_count"),
                    view_mean=export_payload.get("view_mean"),
                    view_std=export_payload.get("view_std"),
                    reliability_warnings=export_payload.get("reliability_warnings", []),
                    documented_limitations=export_payload.get("documented_limitations", []),
                    gate_reason=gate_reason,
                    raw_prediction=raw_p
                )

            # Save report metadata
            save_report_metadata(
                db=db,
                session_id=db_sess.id,
                report_type="ocular_health_screening",
                language=language or "en",
                pdf_filename=f"EyeHealthAI_Report_{db_sess.id}_{language or 'en'}.pdf"
            )

            db.commit()
            export_payload["session_id"] = db_sess.id
            export_payload["id"] = db_sess.id
            export_payload["persistence_status"] = "persisted"
            export_payload["persisted_user_id"] = valid_user_id
        except HTTPException:
            db.rollback()
            raise
        except Exception as exc:
            db.rollback()
            logger.error(f"Session persistence to database failed: {exc}", exc_info=True)
            export_payload["persistence_status"] = "failed"
            export_payload["persistence_error"] = str(exc)
            raise HTTPException(
                status_code=500,
                detail=f"Database persistence failure: Unable to save screening session {exc}"
            )



@app.post("/api/export")
async def export_session_report(session_data: dict):
    """
    Validates and exports a downloadable JSON session audit report.
    """
    if not validate_session_export_schema(session_data):
        raise HTTPException(
            status_code=400,
            detail="Invalid session report schema. Mandatory audit fields missing."
        )

    sid = session_data.get("session_id", "session")
    formatted_json = json.dumps(session_data, indent=2)

    return Response(
        content=formatted_json,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="eyehealthai_{sid}.json"'
        }
    )


# ---------------------------------------------------------------------------
# User Registration, Consent & Session History Endpoints
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    """Request model for patient registration and informed consent with mandatory age."""
    full_name: str = Field(..., min_length=2, max_length=128)
    phone: str = Field(..., min_length=7, max_length=24)
    email: str = Field(..., min_length=5, max_length=128)
    age_years: Optional[int] = Field(None, description="Mandatory age in completed years (0-120)")
    date_of_birth: Optional[str] = Field(None, max_length=16, description="Optional legacy DOB string")
    gender: Optional[str] = Field("prefer_not_to_say", description="Demographic gender identity: female, male, non-binary, self-describe, prefer_not_to_say")
    clinical_sex: Optional[str] = Field("not_disclosed", description="Biological reference variable for WHO 2024 cutoffs: female, male, not_disclosed")
    is_pregnant: bool = Field(False, description="Pregnancy status (relevant for female WHO reference interpretation)")
    pregnancy_trimester: Optional[int] = Field(None, ge=1, le=3, description="Trimester (1, 2, or 3)")
    external_ai_consent: bool = Field(False, description="Explicit consent for privacy-preserved external AI vision evaluation")
    demographic_consent: bool = Field(False, description="Explicit consent for demographic research data collection")
    research_consent: bool = Field(True, description="Explicit consent for research-stage screening")
    data_storage_consent: bool = Field(True, description="Explicit consent for privacy-aware profile storage")

    @model_validator(mode="before")
    @classmethod
    def resolve_age_and_dob(cls, values):
        if not isinstance(values, dict):
            return values
        age_val = values.get("age_years")
        dob_val = values.get("date_of_birth")

        if age_val is not None and age_val != "":
            if isinstance(age_val, bool):
                raise ValueError("Age must be an integer, not a boolean.")
            if isinstance(age_val, float) and not age_val.is_integer():
                raise ValueError("Age must be an integer number of years, not a decimal value.")
            try:
                val = int(age_val)
            except (ValueError, TypeError):
                raise ValueError("Age must be a valid integer between 0 and 120.")
            if val < 0 or val > 120:
                raise ValueError(f"Age must be between 0 and 120 years. Received: {val}")
            values["age_years"] = val
        elif dob_val:
            try:
                birth_year = int(str(dob_val).strip()[:4])
                values["age_years"] = max(0, min(120, 2026 - birth_year))
            except Exception:
                raise ValueError("Age is mandatory and must be a valid integer between 0 and 120.")
        else:
            raise ValueError("Age is mandatory and must be a valid integer between 0 and 120.")
        return values

    @field_validator("clinical_sex", mode="before")
    @classmethod
    def validate_clinical_sex(cls, v):
        if not v:
            return "not_disclosed"
        clean = str(v).strip().lower()
        if clean in {"male", "m"}:
            return "male"
        if clean in {"female", "f"}:
            return "female"
        return "not_disclosed"


class UserUpdateAgeRequest(BaseModel):
    """Request model for updating missing age on legacy patient profiles."""
    age_years: int = Field(..., description="Mandatory age in completed years (0-120)")
    clinical_sex: Optional[str] = Field("not_disclosed")
    is_pregnant: Optional[bool] = Field(False)
    pregnancy_trimester: Optional[int] = Field(None, ge=1, le=3)
    external_ai_consent: Optional[bool] = Field(False)

    @field_validator("age_years", mode="before")
    @classmethod
    def validate_age_years(cls, v):
        if v is None or v == "" or isinstance(v, bool):
            raise ValueError("Age is mandatory and must be a valid integer between 0 and 120.")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("Age must be an integer number of years, not a decimal value.")
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError("Age must be a valid integer between 0 and 120.")
        if val < 0 or val > 120:
            raise ValueError(f"Age must be between 0 and 120 years. Received: {val}")
        return val


@app.post("/api/register")
async def register_patient_user(payload: UserRegisterRequest):
    """
    Registers a new user and logs explicit informed consent.
    Enforces privacy validation rules before proceeding to screening.
    
    SAFETY RULE: Gender and demographic metadata are NEVER fed to computer-vision inference.
    """
    if not payload.research_consent:
        raise HTTPException(
            status_code=400,
            detail="Explicit consent for research screening is mandatory before proceeding."
        )
    if not payload.data_storage_consent:
        raise HTTPException(
            status_code=400,
            detail="Explicit consent for privacy-aware data storage is mandatory."
        )
    if "@" not in payload.email or "." not in payload.email:
        raise HTTPException(
            status_code=400,
            detail="Invalid email format provided."
        )

    clin_sex = payload.clinical_sex or "not_disclosed"
    if clin_sex == "not_disclosed" and payload.gender:
        gen_clean = payload.gender.strip().lower()
        if gen_clean in {"male", "female"}:
            clin_sex = gen_clean

    with SessionLocal() as db:
        user = register_user(
            db=db,
            full_name=payload.full_name,
            phone=payload.phone,
            email=payload.email,
            age_years=payload.age_years,
            date_of_birth=payload.date_of_birth,
            gender=payload.gender,
            clinical_sex=clin_sex,
            is_pregnant=payload.is_pregnant,
            pregnancy_trimester=payload.pregnancy_trimester,
            demographic_consent=payload.demographic_consent,
            research_consent=payload.research_consent,
            data_storage_consent=payload.data_storage_consent,
            external_ai_consent=payload.external_ai_consent
        )
        return {
            "status": "success",
            "user": user.to_dict(),
            "message": "User registered and informed consent logged successfully."
        }


@app.post("/api/user/{user_id}/update-age")
async def update_patient_age(user_id: str, payload: UserUpdateAgeRequest):
    """
    Updates missing age and clinical reference variables for existing legacy accounts.
    Allows unblocking screening while preserving complete historical session auditability.
    """
    with SessionLocal() as db:
        user = get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
        try:
            user = update_user_profile_age(
                db=db,
                user_id=user_id,
                age_years=payload.age_years,
                clinical_sex=payload.clinical_sex,
                is_pregnant=payload.is_pregnant,
                pregnancy_trimester=payload.pregnancy_trimester,
                external_ai_consent=payload.external_ai_consent
            )
            return {
                "status": "success",
                "user": user.to_dict(),
                "message": "User profile age and clinical reference variables updated successfully."
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/user/{user_id}/history")
async def get_user_session_history(user_id: str):
    """
    Returns session history for the specified user.
    """
    with SessionLocal() as db:
        user = get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
        sessions = get_user_sessions(db, user_id)
        return {
            "user_id": user.id,
            "user": user.to_dict(),
            "sessions": [s.to_dict() for s in sessions]
        }


@app.delete("/api/user/{user_id}")
async def delete_user_account(user_id: str):
    """
    Right to Be Forgotten: Permanently deletes user profile and cascades to all sessions.
    """
    with SessionLocal() as db:
        success = delete_user_data(db, user_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
        return {
            "status": "success",
            "message": f"User profile '{user_id}' and all associated sessions permanently deleted."
        }


@app.delete("/api/session/{session_id}")
async def delete_screening_session_endpoint(session_id: str):
    """
    Deletes an individual screening session.
    """
    with SessionLocal() as db:
        success = delete_session_record(db, session_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return {
            "status": "success",
            "message": f"Session '{session_id}' deleted."
        }


# ---------------------------------------------------------------------------
# PDF Report Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/reports/{session_id}/pdf")
@app.get("/reports/{session_id}/pdf")
async def download_session_pdf(session_id: str, lang: str = "en", type: str = "screening"):
    """
    Generates and downloads a clean two-page PDF report.
    type: 'screening' for the patient-friendly report, 'scorecard' for the research scorecard.
    Supports English ('en') and Hindi ('hi').
    """
    session_data = None
    user_data = None
    with SessionLocal() as db:
        sess = get_session_by_id(db, session_id)
        if sess:
            session_data = sess.to_dict()
            session_data["session_id"] = sess.id
            ts_str = sess.session_date.isoformat() if sess.session_date else (sess.created_at.isoformat() if sess.created_at else None)
            session_data["timestamp_utc"] = ts_str
            session_data["session_date"] = ts_str
            session_data["selected_module"] = sess.selected_module
            session_data["capture_source"] = sess.capture_source or "upload"
            session_data["view_types"] = sess.view_types or []
            session_data["quality_summary"] = sess.quality_summary or {}
            session_data["patient_age_years"] = getattr(sess, "patient_age_years", None)
            session_data["patient_clinical_sex"] = getattr(sess, "patient_clinical_sex", None)
            if sess.user:
                user_data = sess.user.to_dict()
                if user_data.get("age_years") is None and sess.patient_age_years is not None:
                    user_data["age_years"] = sess.patient_age_years
            else:
                user_data = {
                    "full_name": "Anonymous Research Participant",
                    "email": "Not Recorded",
                    "phone": "Not Recorded",
                    "age_years": getattr(sess, "patient_age_years", None),
                    "clinical_sex": getattr(sess, "patient_clinical_sex", "not_disclosed"),
                    "gender": getattr(sess, "patient_clinical_sex", "not_disclosed")
                }

            session_data["model_results"] = {}
            for r in sess.model_results:
                r_dict = r.to_dict()
                if "status" not in r_dict or r_dict["status"] is None:
                    r_dict["status"] = r_dict.get("operational_status")
                if "prediction" not in r_dict or r_dict["prediction"] is None:
                    r_dict["prediction"] = r_dict.get("raw_prediction")
                session_data["model_results"][r.module] = r_dict
                if r.module == "anemia":
                    session_data["model_output"] = r.prediction_value
                    session_data["unit"] = r.unit or "g/dL"
                    session_data["view_count"] = r.view_count or 1
                    session_data["view_mean"] = r.view_mean or r.prediction_value
                    session_data["view_std"] = r.view_std
                    session_data["anemia_result"] = r_dict
                    if isinstance(r.raw_prediction, list) and len(r.raw_prediction) > 0 and isinstance(r.raw_prediction[0], dict):
                        session_data["clinical_decision"] = r.raw_prediction[0].get("clinical_decision")
                elif r.module in ("slid", "conjunctival_injection"):
                    session_data["slid_result"] = r_dict
                elif r.module in ("jaundice", "scleral_icterus"):
                    session_data["jaundice_result"] = r_dict
                elif r.module in ("cataract", "nuclear_cataract"):
                    session_data["cataract_result"] = r_dict
                elif r.module in ("stye", "hordeolum", "stye_hordeolum"):
                    session_data["stye_result"] = r_dict
                if r.operational_status:
                    session_data["operational_status"] = r.operational_status
                if r.reliability_warnings:
                    session_data["reliability_warnings"] = r.reliability_warnings
                if r.documented_limitations:
                    session_data["documented_limitations"] = r.documented_limitations

            session_data["anemia"] = session_data.get("anemia_result")
            session_data["slid"] = session_data.get("slid_result")
            session_data["jaundice"] = session_data.get("jaundice_result")
            session_data["cataract"] = session_data.get("cataract_result")
            session_data["stye"] = session_data.get("stye_result")

            # Ensure dynamic clinical decision is calculated if not present
            if session_data.get("view_mean") is not None and not session_data.get("clinical_decision"):
                try:
                    from src.inference.clinical_analytics import compute_clinical_decision
                    ugen = (user_data.get("gender") if user_data else "prefer_not_to_say") or "prefer_not_to_say"
                    session_data["clinical_decision"] = compute_clinical_decision(
                        hb_val=float(session_data["view_mean"]),
                        gender=ugen
                    )
                except Exception as c_err:
                    logger.warning(f"Failed to compute clinical decision for session {sess.id}: {c_err}")

            q_sum = sess.quality_summary or {}
            session_data["segmentation"] = q_sum.get("segmentation")
            session_data["secondary_vision"] = q_sum.get("secondary_vision")
            session_data["reference_retrieval"] = q_sum.get("reference_retrieval")
            session_data["calibrated_ensemble"] = q_sum.get("calibrated_ensemble")
            session_data["input_provenance"] = q_sum.get("input_provenance")
            session_data["personalized_ai"] = q_sum.get("personalized_ai")
            session_data["image_sequence"] = q_sum.get("image_sequence")
            session_data["visualizations"] = q_sum.get("visualizations")
            session_data["previews"] = q_sum.get("previews")

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=f"Screening session '{session_id}' not found. Cannot generate report for nonexistent session."
        )

    is_scorecard = str(type).lower() in ("scorecard", "technical", "research")
    pdf_bytes = build_two_page_pdf(
        session_data=session_data,
        user_data=user_data,
        language=lang,
        doc_type="scorecard" if is_scorecard else "screening"
    )

    clean_lang = "hi" if lang.lower().startswith("hi") else "en"
    if is_scorecard:
        filename = f"OcuScan_Technical_Scorecard_{session_id}_{clean_lang}.pdf"
    else:
        filename = f"EyeHealthAI_Report_{session_id}_{clean_lang}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/api/reports/{session_id}/screening-pdf")
@app.get("/reports/{session_id}/screening-pdf")
async def download_screening_report_pdf(session_id: str, lang: str = "en"):
    """Dedicated endpoint for 2-Page Patient Screening Report."""
    return await download_session_pdf(session_id=session_id, lang=lang, type="screening")


@app.get("/api/reports/{session_id}/scorecard-pdf")
@app.get("/reports/{session_id}/scorecard-pdf")
async def download_scorecard_pdf(session_id: str, lang: str = "en"):
    """Dedicated endpoint for 2-Page Technical Research Scorecard."""
    return await download_session_pdf(session_id=session_id, lang=lang, type="scorecard")


@app.get("/api/download-report")
@app.get("/download-report")
async def download_report_shortcut(session_id: str, lang: str = "en", type: str = "screening"):
    """Shortcut endpoint for downloading reports."""
    return await download_session_pdf(session_id=session_id, lang=lang, type=type)


@app.get("/api/download-scorecard")
@app.get("/download-scorecard")
async def download_scorecard_shortcut(session_id: str, lang: str = "en"):
    """Shortcut endpoint for downloading technical scorecard."""
    return await download_session_pdf(session_id=session_id, lang=lang, type="scorecard")



@app.get("/api/reports/{session_id}")
@app.get("/reports/{session_id}")
async def get_report_data_endpoint(session_id: str):
    """
    Retrieves session and user metadata for rendering in the HTML report view.
    """
    with SessionLocal() as db:
        sess = get_session_by_id(db, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        s_dict = sess.to_dict()
        s_dict["session_id"] = sess.id
        ts_str = sess.session_date.isoformat() if sess.session_date else (sess.created_at.isoformat() if sess.created_at else None)
        s_dict["timestamp_utc"] = ts_str
        s_dict["timestamp"] = ts_str
        s_dict["model_results"] = {}
        user_data = sess.user.to_dict() if sess.user else None
        for r in sess.model_results:
            r_dict = r.to_dict()
            if "status" not in r_dict or r_dict["status"] is None:
                r_dict["status"] = r_dict.get("operational_status")
            if "prediction" not in r_dict or r_dict["prediction"] is None:
                r_dict["prediction"] = r_dict.get("raw_prediction")
            s_dict["model_results"][r.module] = r_dict
            if r.module == "anemia":
                s_dict["anemia_result"] = r_dict
                s_dict["view_mean"] = r.view_mean or r.prediction_value
                s_dict["model_output"] = r.prediction_value
                s_dict["unit"] = r.unit or "g/dL"
                if isinstance(r.raw_prediction, list) and len(r.raw_prediction) > 0 and isinstance(r.raw_prediction[0], dict):
                    s_dict["clinical_decision"] = r.raw_prediction[0].get("clinical_decision")
            elif r.module in ("slid", "conjunctival_injection"):
                s_dict["slid_result"] = r_dict
            elif r.module in ("jaundice", "scleral_icterus"):
                s_dict["jaundice_result"] = r_dict
            elif r.module in ("cataract", "nuclear_cataract"):
                s_dict["cataract_result"] = r_dict
            elif r.module in ("stye", "hordeolum", "stye_hordeolum"):
                s_dict["stye_result"] = r_dict

        if s_dict.get("view_mean") is not None and not s_dict.get("clinical_decision"):
            try:
                from src.inference.clinical_analytics import compute_clinical_decision
                ugen = (user_data.get("gender") if user_data else "prefer_not_to_say") or "prefer_not_to_say"
                s_dict["clinical_decision"] = compute_clinical_decision(
                    hb_val=float(s_dict["view_mean"]),
                    gender=ugen
                )
            except Exception:
                pass

        q_sum = sess.quality_summary or {}
        s_dict["segmentation"] = q_sum.get("segmentation")
        s_dict["secondary_vision"] = q_sum.get("secondary_vision")
        s_dict["reference_retrieval"] = q_sum.get("reference_retrieval")
        s_dict["calibrated_ensemble"] = q_sum.get("calibrated_ensemble")
        s_dict["input_provenance"] = q_sum.get("input_provenance")
        s_dict["image_sequence"] = q_sum.get("image_sequence")
        s_dict["visualizations"] = q_sum.get("visualizations")
        s_dict["previews"] = q_sum.get("previews")

        return {
            "session_id": sess.id,
            "id": sess.id,
            "session": s_dict,
            "user": user_data,
            "model_results": s_dict.get("model_results", {}),
            "anemia": s_dict.get("anemia_result"),
            "anemia_result": s_dict.get("anemia_result"),
            "slid": s_dict.get("slid_result"),
            "slid_result": s_dict.get("slid_result"),
            "jaundice": s_dict.get("jaundice_result"),
            "jaundice_result": s_dict.get("jaundice_result"),
            "cataract": s_dict.get("cataract_result"),
            "cataract_result": s_dict.get("cataract_result"),
            "stye": s_dict.get("stye_result"),
            "stye_result": s_dict.get("stye_result"),
            "view_mean": s_dict.get("view_mean"),
            "segmentation": s_dict.get("segmentation"),
            "secondary_vision": s_dict.get("secondary_vision"),
            "reference_retrieval": s_dict.get("reference_retrieval"),
            "calibrated_ensemble": s_dict.get("calibrated_ensemble"),
            "personalized_ai": s_dict.get("personalized_ai"),
            "image_sequence": s_dict.get("image_sequence"),
            "visualizations": s_dict.get("visualizations"),
            "previews": s_dict.get("previews")
        }



# ---------------------------------------------------------------------------
# Research Copilot Endpoints (Gemini-powered)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for the Research Copilot chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="The research question or message.")
    history: Optional[List[dict]] = Field(default=None, description="Optional conversation history for context.")


@app.get("/api/copilot-status")
async def copilot_status():
    """
    Returns the availability status of the Gemini Research Copilot,
    including active safety gates.
    """
    return get_copilot_status()


@app.post("/api/chat")
async def research_copilot_chat(request: ChatRequest):
    """
    Research Copilot chat endpoint.
    Processes research questions through the Gemini API with full
    PII sanitization and medical-advice blocking.

    SAFETY: No patient images, clinical results, or PII are ever
    forwarded to the LLM. All inputs are sanitized server-side.
    """
    result = await copilot_chat(
        message=request.message,
        conversation_history=request.history
    )
    return result


# ---------------------------------------------------------------------------
# AI Doctor Buddy Mascot Endpoints
# ---------------------------------------------------------------------------

class DoctorBuddyEventPayload(BaseModel):
    """Payload for notifying Doctor Buddy of UI and camera workflow events."""
    event_code: str = Field(..., max_length=64, description="Application event code")
    language: str = Field("en", max_length=5, description="'en' or 'hi'")
    context_note: Optional[str] = Field(None, max_length=256, description="Non-PII context tag")


class DoctorBuddyAskPayload(BaseModel):
    """Payload for user queries to Doctor Buddy."""
    question: str = Field(..., min_length=1, max_length=500, description="User question")
    language: str = Field("en", max_length=5, description="'en' or 'hi'")
    current_step: int = Field(1, ge=1, le=7, description="Current wizard step index")
    session_id: Optional[str] = Field(None, description="Active session ID for session-grounded answers")
    history: Optional[List[dict]] = Field(default=None, description="Optional chat history")


@app.post("/api/doctor-buddy/event")
async def doctor_buddy_event_endpoint(payload: DoctorBuddyEventPayload):
    """
    Returns context-aware, deterministic guidance for camera, quality, and navigation events.
    Guarantees clinical consistency and instant response time.
    """
    from src.inference.doctor_buddy import get_doctor_buddy_guidance
    guidance = get_doctor_buddy_guidance(
        event_code=payload.event_code,
        language=payload.language,
        context_note=payload.context_note
    )
    return guidance


@app.post("/api/doctor-buddy/ask")
async def doctor_buddy_ask_endpoint(payload: DoctorBuddyAskPayload):
    """
    Processes user questions through safe LLM enrichment with deterministic fallback.
    PII sanitization and medical diagnostic advice blocking strictly enforced.
    """
    from src.inference.doctor_buddy import ask_doctor_buddy_question
    response = await ask_doctor_buddy_question(
        question=payload.question,
        language=payload.language,
        current_step=payload.current_step,
        session_id=payload.session_id,
        history=payload.history
    )
    return response


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(f"EyeHealthAI Research Dashboard Server ({settings.ENV})")
    print(f"Dashboard URL: http://{settings.HOST}:{settings.PORT}")
    print(f"Health Check:  http://{settings.HOST}:{settings.PORT}/api/health")
    print(f"API Docs:      http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Research Copilot: {'Enabled' if get_copilot_status()['available'] else 'Offline (set GEMINI_API_KEY)'}")
    print("=======================================================\n")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
