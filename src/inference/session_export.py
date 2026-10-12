"""
src/inference/session_export.py
-------------------------------
Layer 5 & 6 - Session Audit Export Generator and Validator.

Creates standardized, anonymized JSON audit reports for inference sessions.
Strictly excludes fabricated fields and personal identifying information.
Records capture source (upload vs webcam) and quality verification metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

VALID_CAPTURE_SOURCES = {"upload", "webcam"}


def create_session_export(
    module: str,
    inference_result: Dict[str, Any],
    filenames: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    capture_source: str = "upload"
) -> Dict[str, Any]:
    """
    Constructs a standardized session audit report.
    """
    sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
    ts = datetime.now(timezone.utc).isoformat()
    
    # Non-identifying basename references
    clean_filenames = [
        f.replace("\\", "/").split("/")[-1] for f in (filenames or [])
    ]
    
    status = inference_result.get('status', 'unknown')
    permitted = (status == 'available_research_only')
    clean_source = capture_source.lower() if capture_source.lower() in VALID_CAPTURE_SOURCES else "upload"
    
    export_payload = {
        "session_id": sid,
        "timestamp_utc": ts,
        "project": "EyeHealthAI",
        "system_version": "Layers 1-6 (Research Build)",
        "selected_module": module,
        "operational_status": status,
        "inference_permitted": permitted,
        "capture_source": clean_source,
        "checkpoint_identifier": checkpoint_id if permitted else None,
        "input_files": clean_filenames,
        "input_quality_audit": inference_result.get('input_quality'),
        "model_output": inference_result.get('prediction') if permitted else None,
        "unit": inference_result.get('unit') if permitted else None,
        "view_count": inference_result.get('view_count') if permitted else None,
        "view_mean": inference_result.get('view_mean') if permitted else None,
        "view_std": inference_result.get('view_std') if permitted else None,
        "preprocessing_applied": inference_result.get('preprocessing') if permitted else None,
        "reliability_warnings": inference_result.get('warnings', []),
        "documented_limitations": inference_result.get('limitations', []),
        "research_disclaimer": inference_result.get(
            'disclaimer',
            "RESEARCH USE ONLY: Experimental system. Not for medical diagnosis or clinical use."
        ),
        "clinical_decision": (
            inference_result.get('prediction', [{}])[0].get('clinical_decision')
            if (permitted and isinstance(inference_result.get('prediction'), list) and len(inference_result.get('prediction')) > 0 and isinstance(inference_result.get('prediction')[0], dict))
            else None
        ),
        "chromatic_biomarkers": (
            inference_result.get('prediction', [{}])[0].get('chromatic')
            if (permitted and isinstance(inference_result.get('prediction'), list) and len(inference_result.get('prediction')) > 0 and isinstance(inference_result.get('prediction')[0], dict))
            else None
        ),
        "error": inference_result.get('error')
    }
    
    return export_payload


def validate_session_export_schema(payload: Dict[str, Any]) -> bool:
    """
    Validates that all mandatory fields are present in a session export.
    """
    required_keys = {
        "session_id",
        "timestamp_utc",
        "project",
        "selected_module",
        "operational_status",
        "inference_permitted",
        "capture_source",
        "input_files",
        "reliability_warnings",
        "documented_limitations",
        "research_disclaimer"
    }
    return required_keys.issubset(set(payload.keys()))
