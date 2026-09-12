"""
Metric schemas.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class MetricBase(BaseModel):
    round_num: int
    loss: float
    accuracy: float
    communication_bytes: float = 0.0
    client_metrics_json: Optional[str] = None


class MetricCreate(MetricBase):
    simulation_id: int


class MetricResponse(MetricBase):
    id: int
    simulation_id: int
    timestamp: datetime

    class Config:
        from_attributes = True
