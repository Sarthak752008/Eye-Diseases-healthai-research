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
from typing import List, Optional
from PIL import Image, ImageFile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.config import settings
from src.inference.unified_service import get_inference_service
from src.inference.session_export import create_session_export, validate_session_export_schema
from src.preprocessing.image_quality import calculate_image_quality

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(
    title="EyeHealthAI Research Dashboard API",
    description="Multi-modal ocular computer vision screening pipeline with reliability-gated integration.",
    version="1.0.0"
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

# Mount static assets
STATIC_DIR = os.path.join(WORKSPACE_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    return FileResponse(index_path)


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
    jaundice_cfg = os.path.exists(settings.JAUNDICE_CONFIG_PATH)

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
            }
        },
        "disclaimer": "RESEARCH USE ONLY: Prototype service. Not certified or validated for clinical diagnostic use."
    }


@app.get("/api/status")
async def get_system_status():
    """Returns operational status for all clinical modules and checkpoints."""
    anemia_ckpt = os.path.exists(settings.ANEMIA_CHECKPOINT_PATH)
    conj_ckpt = os.path.exists(settings.CONJUNCTIVITIS_CHECKPOINT_PATH)

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
        quality = calculate_image_quality(rgb_img)
        dims = {"width": rgb_img.width, "height": rgb_img.height}

        # Actionable guidance generation
        guidance = []
        is_usable = True
        if quality.get("is_blurry"):
            guidance.append("Capture again holding the camera steady to reduce motion blur.")
            is_usable = False
        if quality.get("is_too_dark"):
            guidance.append("Improve lighting: face a diffuse light source or increase room illumination.")
            is_usable = False
        if quality.get("is_too_bright"):
            guidance.append("Reduce glare: adjust angle to prevent direct bright reflections on the ocular surface.")
            is_usable = False
        if not guidance:
            guidance.append("Exposure and sharpness are within operational bounds.")

        guidance.append("Position the eye within 10–15 cm of the camera lens for optimal focal clarity.")
        guidance.append("Macro framing required: Capture a close-up ocular view of the conjunctiva, not a full-face selfie.")

        return {
            "filename": image.filename,
            "dimensions": dims,
            "quality": quality,
            "usability_status": "usable" if is_usable else "needs_adjustment",
            "actionable_guidance": guidance,
            "anatomical_disclaimer": "Image-quality check verifies pixel sharpness and luminance only. It does not verify that the inner conjunctiva or sclera is properly everted or visible. A full-face image is not a valid conjunctiva image.",
            "is_valid": True
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "filename": image.filename,
            "dimensions": None,
            "quality": None,
            "is_valid": False,
            "error": "Image processing error: unable to decode or inspect image."
        }


@app.post("/api/analyze")
async def analyze_ocular_images(
    module: str = Form("anemia"),
    images: Optional[List[UploadFile]] = File(None),
    view_types: Optional[List[str]] = Form(None),
    capture_source: str = Form("upload"),
    experimental_override: bool = Form(False)
):
    """
    Unified multi-view inference endpoint.
    Processes images strictly in-memory without persistent disk storage.
    Enforces camera-domain validation gates for laptop webcam inputs.
    """
    service = get_inference_service()
    clean_module = module.strip().lower()
    clean_source = capture_source.strip().lower()

    # Route blocked or unavailable modules immediately
    if clean_module in ('conjunctivitis', 'red_eye'):
        res = service.predict_conjunctivitis()
        export_payload = create_session_export(
            module="conjunctivitis",
            inference_result=res,
            filenames=[f.filename for f in (images or [])],
            capture_source=clean_source
        )
        return export_payload

    if clean_module in ('jaundice', 'icterus'):
        res = service.predict_jaundice()
        export_payload = create_session_export(
            module="jaundice",
            inference_result=res,
            filenames=[f.filename for f in (images or [])],
            capture_source=clean_source
        )
        return export_payload

    if clean_module != 'anemia':
        raise HTTPException(
            status_code=400,
            detail=f"Unknown module '{module}'. Supported modules: ['anemia', 'conjunctivitis', 'jaundice']."
        )

    # Anemia validation
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided for anemia analysis.")

    # Convert uploaded files to PIL Images in-memory
    pil_images = []
    filenames = []
    for idx, f in enumerate(images):
        filenames.append(f.filename or f"image_{idx}")
        content = await f.read()
        validate_uploaded_image(f, content)

        rgb_img = safe_decode_bytes(content)
        if rgb_img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode image #{idx} ({f.filename}). Corrupt file bytes.")
        pil_images.append(rgb_img)

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

    # Execute inference with camera-domain gate
    inference_res = service.predict_anemia(
        pil_images,
        view_types=v_types,
        capture_source=clean_source,
        experimental_override=experimental_override
    )

    if inference_res.get('status') == 'error':
        raise HTTPException(status_code=400, detail=inference_res.get('error'))

    # Package as structured session export
    export_payload = create_session_export(
        module="anemia",
        inference_result=inference_res,
        filenames=filenames,
        checkpoint_id="ResNet50_saved_models_anemia_best_model.pth",
        capture_source=clean_source
    )

    return export_payload


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


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(f"EyeHealthAI Research Dashboard Server ({settings.ENV})")
    print(f"Dashboard URL: http://{settings.HOST}:{settings.PORT}")
    print(f"Health Check:  http://{settings.HOST}:{settings.PORT}/api/health")
    print(f"API Docs:      http://{settings.HOST}:{settings.PORT}/docs")
    print("=======================================================\n")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
