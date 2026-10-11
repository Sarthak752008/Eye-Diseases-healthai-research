"""
src/database/__init__.py
------------------------
EyeHealthAI database package initialization.
"""

from src.database.models import (
    Base,
    UserProfile,
    ConsentRecord,
    ScreeningSession,
    ModelResult,
    ReportMetadata,
)
from src.database.session import (
    init_db,
    get_db,
    get_db_context,
    register_user,
    get_user_by_id,
    get_user_by_email,
    create_session,
    save_model_result,
    get_session_by_id,
    get_user_sessions,
    save_report_metadata,
    delete_user_data,
    delete_session_record,
    engine,
    SessionLocal
)

__all__ = [
    "Base",
    "UserProfile",
    "ConsentRecord",
    "ScreeningSession",
    "ModelResult",
    "ReportMetadata",
    "init_db",
    "get_db",
    "get_db_context",
    "register_user",
    "get_user_by_id",
    "get_user_by_email",
    "create_session",
    "save_model_result",
    "get_session_by_id",
    "get_user_sessions",
    "save_report_metadata",
    "delete_user_data",
    "delete_session_record",
    "engine",
    "SessionLocal"
]
