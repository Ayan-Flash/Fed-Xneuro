"""
Assessment Database Model for Fed-XNeuro Platform.
Persists hospital clinical cognitive evaluations and neuroimaging findings permanently.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime
from backend.app.db.session import Base


class AssessmentEntity(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True, index=True)
    patient_name = Column(String, nullable=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    mmse = Column(Float, nullable=True)
    cdr = Column(Float, nullable=True)
    risk = Column(String, nullable=True, index=True)
    probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model = Column(String, nullable=True)
    badge = Column(String, nullable=True)
    has_imaging = Column(Integer, default=0)
    biomarkers = Column(JSON, nullable=True)
    factors = Column(JSON, nullable=True)
    recommendation = Column(String, nullable=True)
    timestamp = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
