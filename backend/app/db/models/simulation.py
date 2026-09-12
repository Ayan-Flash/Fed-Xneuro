"""
Simulation SQL model.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.app.db.session import Base


class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False, default="Simulation")
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="created")  # created, running, completed, failed, stopped

    # Configuration
    dataset = Column(String(100), nullable=False, default="mnist")
    model = Column(String(100), nullable=False, default="cnn")
    algorithm = Column(String(100), nullable=False, default="fedavg")
    num_clients = Column(Integer, nullable=False, default=5)
    client_fraction = Column(Float, nullable=False, default=1.0)
    num_rounds = Column(Integer, nullable=False, default=10)
    local_epochs = Column(Integer, nullable=False, default=1)
    batch_size = Column(Integer, nullable=False, default=32)
    learning_rate = Column(Float, nullable=False, default=0.01)
    partition_type = Column(String(50), nullable=False, default="iid")
    partition_alpha = Column(Float, nullable=False, default=0.5)
    seed = Column(Integer, nullable=False, default=42)
    device = Column(String(50), nullable=False, default="auto")

    # Tracking progress
    current_round = Column(Integer, nullable=False, default=0)
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    results_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
