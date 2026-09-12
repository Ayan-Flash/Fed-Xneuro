"""
Algorithm SQL model.
"""

from sqlalchemy import Column, Integer, String, Text
from backend.app.db.session import Base


class Algorithm(Base):
    __tablename__ = "algorithms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), default="baseline")  # baseline, proposed
