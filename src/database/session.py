"""
src/database/session.py
-----------------------
Database engine, session management, and CRUD helper operations.
Supports local SQLite development and PostgreSQL production deployment.
"""

import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import (
    Base,
    UserProfile,
    ConsentRecord,
    ScreeningSession,
    ModelResult,
    ReportMetadata,
    generate_uuid
)

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(WORKSPACE_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_SQLITE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'eyehealthai.db')}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# Configure engine with connection pooling and thread safety for SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_db() -> None:
    """Creates database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for standalone scripts or non-request jobs."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CRUD Service Operations
# ---------------------------------------------------------------------------

def register_user(
    db: Session,
    full_name: str,
    phone: str,
    email: str,
    date_of_birth: str,
    research_consent: bool = True,
    data_storage_consent: bool = True,
    consent_version: str = "1.0"
) -> UserProfile:
    """
    Registers a new user profile and records an explicit, timestamped consent record.
    """
    # Check if existing user by email
    existing_user = db.query(UserProfile).filter(UserProfile.email == email.strip().lower()).first()
    if existing_user:
        # Update details
        existing_user.full_name = full_name.strip()
        existing_user.phone = phone.strip()
        existing_user.date_of_birth = date_of_birth.strip()
        user = existing_user
    else:
        user = UserProfile(
            id=f"usr_{generate_uuid()[:12]}",
            full_name=full_name.strip(),
            phone=phone.strip(),
            email=email.strip().lower(),
            date_of_birth=date_of_birth.strip()
        )
        db.add(user)
        db.flush()

    # Record consent
    consent = ConsentRecord(
        id=f"cns_{generate_uuid()[:12]}",
        user_id=user.id,
        consent_given=True,
        research_screening_consent=research_consent,
        data_storage_consent=data_storage_consent,
        consent_version=consent_version
    )
    db.add(consent)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[UserProfile]:
    """Retrieves a user profile by ID."""
    return db.query(UserProfile).filter(UserProfile.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[UserProfile]:
    """Retrieves a user profile by email."""
    return db.query(UserProfile).filter(UserProfile.email == email.strip().lower()).first()


def create_session(
    db: Session,
    selected_module: str,
    capture_source: str = "upload",
    user_id: Optional[str] = None,
    language: str = "en",
    view_types: Optional[List[str]] = None,
    quality_summary: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> ScreeningSession:
    """
    Creates a new screening session record.
    """
    sid = session_id or f"ses_{generate_uuid()[:12]}"
    session = ScreeningSession(
        id=sid,
        user_id=user_id,
        language=language,
        selected_module=selected_module,
        capture_source=capture_source,
        view_types=view_types or [],
        quality_summary=quality_summary or {}
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def save_model_result(
    db: Session,
    session_id: str,
    module: str,
    operational_status: str,
    prediction_value: Optional[float] = None,
    unit: Optional[str] = None,
    view_count: Optional[int] = None,
    view_mean: Optional[float] = None,
    view_std: Optional[float] = None,
    reliability_warnings: Optional[List[str]] = None,
    documented_limitations: Optional[List[str]] = None,
    gate_reason: Optional[str] = None
) -> ModelResult:
    """
    Records a structured model output or gated status for a session.
    """
    res = ModelResult(
        id=f"res_{generate_uuid()[:12]}",
        session_id=session_id,
        module=module,
        operational_status=operational_status,
        prediction_value=prediction_value,
        unit=unit,
        view_count=view_count,
        view_mean=view_mean,
        view_std=view_std,
        reliability_warnings=reliability_warnings or [],
        documented_limitations=documented_limitations or [],
        gate_reason=gate_reason
    )
    db.add(res)
    db.commit()
    db.refresh(res)
    return res


def get_session_by_id(db: Session, session_id: str) -> Optional[ScreeningSession]:
    """Retrieves session details by session ID."""
    return db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()


def get_user_sessions(db: Session, user_id: str) -> List[ScreeningSession]:
    """Retrieves all sessions belonging to a specific user, ordered newest first."""
    return (
        db.query(ScreeningSession)
        .filter(ScreeningSession.user_id == user_id)
        .order_by(ScreeningSession.created_at.desc())
        .all()
    )


def save_report_metadata(
    db: Session,
    session_id: str,
    report_type: str = "ocular_health_screening",
    language: str = "en",
    pdf_filename: Optional[str] = None
) -> ReportMetadata:
    """Logs report generation for a session."""
    meta = ReportMetadata(
        id=f"rep_{generate_uuid()[:12]}",
        session_id=session_id,
        report_type=report_type,
        language=language,
        pdf_filename=pdf_filename,
        status="active"
    )
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return meta


def delete_user_data(db: Session, user_id: str) -> bool:
    """
    Right to Be Forgotten: Permanently deletes a user profile and cascades
    to all associated consents, screening sessions, and report records.
    """
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True


def delete_session_record(db: Session, session_id: str) -> bool:
    """
    Deletes an individual session record and associated results/reports.
    """
    sess = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
    if not sess:
        return False
    db.delete(sess)
    db.commit()
    return True
