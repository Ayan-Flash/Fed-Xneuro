"""
Simulation schemas.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SimulationBase(BaseModel):
    name: str = "Simulation"
    description: Optional[str] = None
    dataset: str = "mnist"
    model: str = "cnn"
    algorithm: str = "fedavg"
    num_clients: int = Field(default=5, ge=1)
    client_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    num_rounds: int = Field(default=10, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    partition_type: str = "iid"
    partition_alpha: float = 0.5
    seed: int = 42
    device: str = "auto"


class SimulationCreate(SimulationBase):
    pass


class SimulationUpdate(BaseModel):
    status: Optional[str] = None
    current_round: Optional[int] = None
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None


class SimulationResponse(SimulationBase):
    id: int
    run_id: str
    status: str
    current_round: int
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SimulationStatus(BaseModel):
    simulation_id: int
    run_id: str
    status: str
    current_round: int
    total_rounds: int
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
