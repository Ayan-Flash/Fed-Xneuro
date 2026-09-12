"""
Client SQL model.
"""

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from backend.app.db.session import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), index=True, nullable=False)
    client_id = Column(String(50), nullable=False)
    num_samples = Column(Integer, default=0)
    last_accuracy = Column(Float, nullable=True)
    last_loss = Column(Float, nullable=True)
