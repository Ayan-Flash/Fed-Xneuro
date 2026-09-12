"""
Experiment service.
"""

from typing import List
from sqlalchemy.orm import Session
from backend.app.db.models.experiment import Experiment
from backend.app.schemas.experiment import ExperimentCreate


class ExperimentService:
    @staticmethod
    def create_experiment(db: Session, exp_in: ExperimentCreate) -> Experiment:
        exp = Experiment(
            name=exp_in.name,
            description=exp_in.description,
            config_path=exp_in.config_path,
            status="idle",
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        return exp

    @staticmethod
    def list_experiments(db: Session, skip: int = 0, limit: int = 50) -> List[Experiment]:
        return db.query(Experiment).offset(skip).limit(limit).all()
