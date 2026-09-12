"""
Dataset API routes.
"""

from typing import List
from fastapi import APIRouter
from backend.app.schemas.algorithm import DatasetInfo
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetInfo])
def list_datasets():
    """Lists all available datasets."""
    return DatasetService.list_datasets()
