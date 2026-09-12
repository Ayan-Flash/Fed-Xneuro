"""
Algorithm metadata API routes.
"""

from typing import List
from fastapi import APIRouter
from backend.app.schemas.algorithm import AlgorithmInfo
from backend.app.services.algorithm_service import AlgorithmService

router = APIRouter(prefix="/algorithms", tags=["algorithms"])


@router.get("", response_model=List[AlgorithmInfo])
def list_algorithms():
    """Lists all available federated learning algorithms."""
    return AlgorithmService.list_algorithms()
