"""
Project SQL model.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from backend.app.db.session import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
