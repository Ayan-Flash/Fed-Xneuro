"""
Metrics API routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.metric import MetricResponse
from backend.app.services.metrics_service import MetricsService

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/{simulation_id}", response_model=List[MetricResponse])
def get_metrics_for_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Retrieves round-by-round metrics for a given simulation."""
    return MetricsService.get_metrics_for_simulation(db, simulation_id)
