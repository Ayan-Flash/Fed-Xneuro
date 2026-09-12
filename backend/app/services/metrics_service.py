"""
Metrics and results services.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.db.models.metric import Metric
from backend.app.db.models.result import Result


class MetricsService:
    @staticmethod
    def get_metrics_for_simulation(db: Session, simulation_id: int) -> List[Metric]:
        return db.query(Metric).filter(Metric.simulation_id == simulation_id).order_by(Metric.round_num.asc()).all()

    @staticmethod
    def add_round_metric(
        db: Session,
        simulation_id: int,
        round_num: int,
        loss: float,
        accuracy: float,
        communication_bytes: float = 0.0,
        client_metrics: Optional[Dict[str, Any]] = None,
    ) -> Metric:
        import json
        m = Metric(
            simulation_id=simulation_id,
            round_num=round_num,
            loss=loss,
            accuracy=accuracy,
            communication_bytes=communication_bytes,
            client_metrics_json=json.dumps(client_metrics) if client_metrics else None,
        )
        db.add(m)
        db.commit()
        db.refresh(m)
        return m


class ResultService:
    @staticmethod
    def get_result(db: Session, simulation_id: int) -> Optional[Result]:
        return db.query(Result).filter(Result.simulation_id == simulation_id).first()
