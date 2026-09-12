"""
Algorithm and architecture schemas.
"""

from typing import Optional, List
from pydantic import BaseModel


class AlgorithmInfo(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    category: str = "baseline"


class ModelInfo(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    architecture_type: str = "cnn"


class DatasetInfo(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    num_classes: int = 10
    data_type: str = "image"
