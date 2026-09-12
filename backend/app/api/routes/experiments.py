"""
Experiment API routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.experiment import ExperimentCreate, ExperimentResponse
from backend.app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse)
def create_experiment(exp_in: ExperimentCreate, db: Session = Depends(get_db)):
    return ExperimentService.create_experiment(db, exp_in)


@router.get("", response_model=List[ExperimentResponse])
def list_experiments(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return ExperimentService.list_experiments(db, skip=skip, limit=limit)
