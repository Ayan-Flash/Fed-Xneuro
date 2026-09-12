"""
Result schemas.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel


class ResultResponse(BaseModel):
    id: int
    simulation_id: int
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    metrics_path: Optional[str] = None
    plot_path: Optional[str] = None
    summary_json: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
