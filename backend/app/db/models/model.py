"""
Model SQL model.
"""

from sqlalchemy import Column, Integer, String, Text
from backend.app.db.session import Base


class ModelEntity(Base):
    __tablename__ = "registered_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    architecture_type = Column(String(50), default="cnn")
