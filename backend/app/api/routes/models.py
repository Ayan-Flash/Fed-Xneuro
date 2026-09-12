"""
Model architecture API routes.
"""

from typing import List
from fastapi import APIRouter
from backend.app.schemas.algorithm import ModelInfo
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelInfo])
def list_models():
    """Lists all available neural network architectures."""
    return ModelService.list_models()
