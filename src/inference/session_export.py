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
        "status": status,
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
        "error": inference_result.get('error'),
        "raw_prediction": inference_result.get('prediction')
    }
    
    if module in ("unified", "all", "multi", "multi_target", "unified_ocular_screening"):
        anemia_data = inference_result.get("anemia")
        slid_data = inference_result.get("slid")
        jaund_data = inference_result.get("jaundice")
        cataract_data = inference_result.get("cataract")
        stye_data = inference_result.get("stye")
        if anemia_data and "operational_status" not in anemia_data:
            anemia_data["operational_status"] = anemia_data.get("status", "available_research_only")
        if slid_data and "operational_status" not in slid_data:
            slid_data["operational_status"] = slid_data.get("status", "available_research_only")
        if jaund_data and "operational_status" not in jaund_data:
            jaund_data["operational_status"] = jaund_data.get("status", "available_research_only")
        if cataract_data and "operational_status" not in cataract_data:
            cataract_data["operational_status"] = cataract_data.get("status", "available_research_only")
        if stye_data and "operational_status" not in stye_data:
            stye_data["operational_status"] = stye_data.get("status", "available_research_only")
        export_payload["anemia"] = anemia_data
        export_payload["slid"] = slid_data
        export_payload["jaundice"] = jaund_data
        export_payload["cataract"] = cataract_data
        export_payload["stye"] = stye_data
        export_payload["anemia_result"] = anemia_data
        export_payload["slid_result"] = slid_data
        export_payload["jaundice_result"] = jaund_data
        export_payload["cataract_result"] = cataract_data
        export_payload["stye_result"] = stye_data
        export_payload["model_results"] = {
            "anemia": anemia_data,
            "slid": slid_data,
            "jaundice": jaund_data,
            "cataract": cataract_data,
            "stye": stye_data
        }
        export_payload["segmentation"] = inference_result.get("segmentation")
        export_payload["secondary_vision"] = inference_result.get("secondary_vision")
        export_payload["reference_retrieval"] = inference_result.get("reference_retrieval")
        export_payload["calibrated_ensemble"] = inference_result.get("calibrated_ensemble")
        export_payload["personalized_ai"] = inference_result.get("personalized_ai")
        if anemia_data and isinstance(anemia_data.get("prediction"), list) and len(anemia_data["prediction"]) > 0:
            export_payload["clinical_decision"] = anemia_data["prediction"][0].get("clinical_decision")
            export_payload["chromatic_biomarkers"] = anemia_data["prediction"][0].get("chromatic")

    if "patient_age_years" in inference_result:
        export_payload["patient_age_years"] = inference_result["patient_age_years"]
    if "patient_clinical_sex" in inference_result:
        export_payload["patient_clinical_sex"] = inference_result["patient_clinical_sex"]
    if "personalized_ai" in inference_result and "personalized_ai" not in export_payload:
        export_payload["personalized_ai"] = inference_result["personalized_ai"]

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
