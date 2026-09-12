"""
Metric SQL model.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from backend.app.db.session import Base


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), index=True, nullable=False)
    round_num = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    communication_bytes = Column(Float, default=0.0)
    client_metrics_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
