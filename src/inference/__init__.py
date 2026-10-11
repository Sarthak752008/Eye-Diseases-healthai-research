"""
src/inference/__init__.py
Layer 4 - Unified Inference Service for EyeHealthAI
"""

from .unified_service import (
    EyeHealthInferenceService,
    InferenceResult,
    COMPATIBLE_ANEMIA_VIEWS,
    get_inference_service,
)
from .session_export import (
    create_session_export,
    validate_session_export_schema,
)

__all__ = [
    'EyeHealthInferenceService',
    'InferenceResult',
    'COMPATIBLE_ANEMIA_VIEWS',
    'get_inference_service',
    'create_session_export',
    'validate_session_export_schema',
]

