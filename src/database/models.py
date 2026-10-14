"""
src/database/models.py
----------------------
SQLAlchemy 2.0 ORM models for EyeHealthAI.

Provides structured persistence for:
1. UserProfile: Non-identifying / privacy-aware user registration.
2. ConsentRecord: Timestamped explicit informed consent records.
3. ScreeningSession: Individual ocular screening sessions.
4. ModelResult: Structured inference outputs and reliability gate statuses.
5. ReportMetadata: Generated screening report records.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserProfile(Base):
    """
    User registration profile.
    Supports privacy-aware screening and session history tracking.
    """
    __tablename__ = "user_profiles"

    id = Column(String(64), primary_key=True, default=generate_uuid, index=True)
    full_name = Column(String(128), nullable=False)
    phone = Column(String(32), nullable=False)
    email = Column(String(128), nullable=False, index=True)
    date_of_birth = Column(String(16), nullable=False)  # Format: YYYY-MM-DD
    gender = Column(String(32), nullable=True, default="prefer_not_to_say")
    demographic_consent = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    consents = relationship("ConsentRecord", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ScreeningSession", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "phone": self.phone,
            "email": self.email,
            "date_of_birth": self.date_of_birth,
            "gender": self.gender,
            "demographic_consent": self.demographic_consent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ConsentRecord(Base):
    """
    Audit-compliant informed consent record.
    Logs explicit user agreement before screening or data storage begins.
    """
    __tablename__ = "consent_records"

    id = Column(String(64), primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String(64), ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    consent_given = Column(Boolean, nullable=False, default=True)
    research_screening_consent = Column(Boolean, nullable=False, default=True)
    data_storage_consent = Column(Boolean, nullable=False, default=True)
    demographic_consent = Column(Boolean, nullable=False, default=False)
    consent_version = Column(String(16), nullable=False, default="1.0")
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)

    # Relationship
    user = relationship("UserProfile", back_populates="consents")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "consent_given": self.consent_given,
            "research_screening_consent": self.research_screening_consent,
            "data_storage_consent": self.data_storage_consent,
            "demographic_consent": self.demographic_consent,
            "consent_version": self.consent_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class ScreeningSession(Base):
    """
    Represents an individual ocular screening session.
    Images are NOT stored permanently; only quality metadata, filenames, and results are retained.
    """
    __tablename__ = "screening_sessions"

    id = Column(String(64), primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String(64), ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=True, index=True)
    session_date = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    language = Column(String(8), nullable=False, default="en")
    selected_module = Column(String(32), nullable=False)
    capture_source = Column(String(16), nullable=False, default="upload")  # "upload" or "webcam"
    view_types = Column(JSON, nullable=True)  # List of view names (e.g. ['palpebral', 'forniceal'])
    quality_summary = Column(JSON, nullable=True)  # Aggregated quality flags
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    user = relationship("UserProfile", back_populates="sessions")
    model_results = relationship("ModelResult", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("ReportMetadata", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        ts = self.session_date.isoformat() if self.session_date else (self.created_at.isoformat() if self.created_at else None)
        return {
            "id": self.id,
            "session_id": self.id,
            "user_id": self.user_id,
            "session_date": ts,
            "timestamp_utc": ts,
            "language": self.language,
            "selected_module": self.selected_module,
            "capture_source": self.capture_source,
            "view_types": self.view_types or [],
            "quality_summary": self.quality_summary or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "results": [r.to_dict() for r in (self.model_results or [])],
        }


class ModelResult(Base):
    """
    Structured model outputs and operational availability statuses.
    """
    __tablename__ = "model_results"

    id = Column(String(64), primary_key=True, default=generate_uuid, index=True)
    session_id = Column(String(64), ForeignKey("screening_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    module = Column(String(32), nullable=False, index=True)
    operational_status = Column(String(32), nullable=False)  # available_research_only, blocked_validation, unavailable_missing_data
    prediction_value = Column(Float, nullable=True)
    unit = Column(String(16), nullable=True)
    view_count = Column(Integer, nullable=True)
    view_mean = Column(Float, nullable=True)
    view_std = Column(Float, nullable=True)
    reliability_warnings = Column(JSON, nullable=True)
    documented_limitations = Column(JSON, nullable=True)
    gate_reason = Column(Text, nullable=True)
    raw_prediction = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationship
    session = relationship("ScreeningSession", back_populates="model_results")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "module": self.module,
            "operational_status": self.operational_status,
            "prediction_value": self.prediction_value,
            "unit": self.unit,
            "view_count": self.view_count,
            "view_mean": self.view_mean,
            "view_std": self.view_std,
            "raw_prediction": self.raw_prediction,
            "model_output": self.raw_prediction,
            "reliability_warnings": self.reliability_warnings or [],
            "documented_limitations": self.documented_limitations or [],
            "gate_reason": self.gate_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ReportMetadata(Base):
    """
    Audit log of generated PDF and HTML reports.
    """
    __tablename__ = "report_metadata"

    id = Column(String(64), primary_key=True, default=generate_uuid, index=True)
    session_id = Column(String(64), ForeignKey("screening_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    report_type = Column(String(32), nullable=False, default="ocular_health_screening")
    language = Column(String(8), nullable=False, default="en")
    generated_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    pdf_filename = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False, default="active")  # active, purged

    # Relationship
    session = relationship("ScreeningSession", back_populates="reports")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "report_type": self.report_type,
            "language": self.language,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "pdf_filename": self.pdf_filename,
            "status": self.status,
        }
