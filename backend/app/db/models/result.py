"""
Result SQL model.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from backend.app.db.session import Base


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), unique=True, index=True, nullable=False)
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    metrics_path = Column(String(500), nullable=True)
    plot_path = Column(String(500), nullable=True)
    summary_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
