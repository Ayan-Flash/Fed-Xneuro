"""
Experiment schemas.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config_path: Optional[str] = None


class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    config_path: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
