"""
Simulation management and execution service.
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from backend.app.db.models.simulation import Simulation
from backend.app.db.models.metric import Metric
from backend.app.schemas.simulation import SimulationCreate
from backend.app.core.exceptions import NotFoundException
from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine


class SimulationService:
    @staticmethod
    def create_simulation(db: Session, sim_in: SimulationCreate) -> Simulation:
        run_id = f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        sim = Simulation(
            run_id=run_id,
            name=sim_in.name,
            description=sim_in.description,
            status="created",
            dataset=sim_in.dataset,
            model=sim_in.model,
            algorithm=sim_in.algorithm,
            num_clients=sim_in.num_clients,
            client_fraction=sim_in.client_fraction,
            num_rounds=sim_in.num_rounds,
            local_epochs=sim_in.local_epochs,
            batch_size=sim_in.batch_size,
            learning_rate=sim_in.learning_rate,
            partition_type=sim_in.partition_type,
            partition_alpha=sim_in.partition_alpha,
            seed=sim_in.seed,
            device=sim_in.device,
            current_round=0,
        )
        db.add(sim)
        db.commit()
        db.refresh(sim)
        return sim

    @staticmethod
    def get_simulation(db: Session, sim_id: int) -> Simulation:
        sim = db.query(Simulation).filter(Simulation.id == sim_id).first()
        if not sim:
            raise NotFoundException(f"Simulation with id {sim_id} not found")
        return sim

    @staticmethod
    def get_simulation_by_run_id(db: Session, run_id: str) -> Simulation:
        sim = db.query(Simulation).filter(Simulation.run_id == run_id).first()
        if not sim:
            raise NotFoundException(f"Simulation with run_id {run_id} not found")
        return sim

    @staticmethod
    def list_simulations(db: Session, skip: int = 0, limit: int = 50) -> List[Simulation]:
        return db.query(Simulation).order_by(Simulation.id.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def delete_simulation(db: Session, sim_id: int) -> bool:
        sim = SimulationService.get_simulation(db, sim_id)
        db.query(Metric).filter(Metric.simulation_id == sim_id).delete()
        from backend.app.db.models.result import Result
        db.query(Result).filter(Result.simulation_id == sim_id).delete()
        db.delete(sim)
        db.commit()
        return True

    @staticmethod
    def update_simulation_status(
        db: Session,
        sim_id: int,
        status: str,
        current_round: Optional[int] = None,
        final_loss: Optional[float] = None,
        final_accuracy: Optional[float] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> Simulation:
        sim = SimulationService.get_simulation(db, sim_id)
        sim.status = status
        if current_round is not None:
            sim.current_round = current_round
        if final_loss is not None:
            sim.final_loss = final_loss
        if final_accuracy is not None:
            sim.final_accuracy = final_accuracy
        if results is not None:
            sim.results_json = json.dumps(results)
        db.commit()
        db.refresh(sim)
        return sim
