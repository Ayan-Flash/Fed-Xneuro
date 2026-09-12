"""
Experiment SQL model.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from backend.app.db.session import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    config_path = Column(String(500), nullable=True)
    status = Column(String(50), default="idle")
    created_at = Column(DateTime, default=datetime.utcnow)
