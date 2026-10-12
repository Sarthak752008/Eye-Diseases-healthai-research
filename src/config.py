"""
src/config.py
-------------
Centralized configuration management for EyeHealthAI.
Loads environment variables with robust defaults for development and production.
"""

import os
from typing import List, Set

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Settings:
    # Environment
    ENV: str = os.getenv("ENV", "development").lower()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Server binding
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security & CORS
    # In production, set ALLOWED_ORIGINS to specific domains (e.g., https://eyehealthai.research.org)
    _raw_origins: str = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    ALLOWED_ORIGINS: List[str] = [orig.strip() for orig in _raw_origins.split(",") if orig.strip()]

    # Upload constraints
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    ALLOWED_MIME_TYPES: Set[str] = {
        'image/jpeg',
        'image/png',
        'image/webp',
        'image/bmp',
        'image/tiff',
        'application/octet-stream'  # Fallback for some browser canvas blobs
    }

    # Model & Checkpoint Paths
    ANEMIA_CHECKPOINT_PATH: str = os.getenv(
        "ANEMIA_CHECKPOINT_PATH",
        os.path.join(WORKSPACE_ROOT, "saved_models", "anemia", "best_model.pth")
    )
    CONJUNCTIVITIS_CHECKPOINT_PATH: str = os.getenv(
        "CONJUNCTIVITIS_CHECKPOINT_PATH",
        os.path.join(WORKSPACE_ROOT, "saved_models", "conjunctivitis", "best_model_v2_clean.pth")
    )
    SLID_CHECKPOINT_PATH: str = os.getenv(
        "SLID_CHECKPOINT_PATH",
        os.path.join(WORKSPACE_ROOT, "saved_models", "slid", "best_model.pth")
    )
    JAUNDICE_CHECKPOINT_PATH: str = os.getenv(
        "JAUNDICE_CHECKPOINT_PATH",
        os.path.join(WORKSPACE_ROOT, "saved_models", "jaundice", "best_model.pth")
    )
    JAUNDICE_CONFIG_PATH: str = os.getenv(
        "JAUNDICE_CONFIG_PATH",
        os.path.join(WORKSPACE_ROOT, "saved_models", "jaundice", "jaundice_config.json")
    )

    # Compute Device
    DEVICE: str = os.getenv("DEVICE", "cpu")

settings = Settings()
