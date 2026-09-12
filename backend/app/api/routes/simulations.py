"""
Simulation management API routes.
"""

from typing import List
from fastapi import APIRouter, Depends, BackgroundTasks, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.simulation import (
    SimulationCreate,
    SimulationResponse,
    SimulationStatus,
)
from backend.app.services.simulation_service import SimulationService
from backend.app.workers.simulation_worker import SimulationWorker

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.post("", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
def create_simulation(sim_in: SimulationCreate, db: Session = Depends(get_db)):
    """Creates a new federated learning simulation experiment."""
    return SimulationService.create_simulation(db, sim_in)


@router.get("", response_model=List[SimulationResponse])
def list_simulations(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Lists recent simulations."""
    return SimulationService.list_simulations(db, skip=skip, limit=limit)


@router.get("/{simulation_id}", response_model=SimulationResponse)
def get_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Fetches details of a specific simulation."""
    return SimulationService.get_simulation(db, simulation_id)


@router.get("/{simulation_id}/status", response_model=SimulationStatus)
def get_simulation_status(simulation_id: int, db: Session = Depends(get_db)):
    """Fetches the current progress and status of a simulation."""
    sim = SimulationService.get_simulation(db, simulation_id)
    return SimulationStatus(
        simulation_id=sim.id,
        run_id=sim.run_id,
        status=sim.status,
        current_round=sim.current_round,
        total_rounds=sim.num_rounds,
        final_accuracy=sim.final_accuracy,
        final_loss=sim.final_loss,
    )


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Deletes a simulation."""
    SimulationService.delete_simulation(db, simulation_id)
    return None
