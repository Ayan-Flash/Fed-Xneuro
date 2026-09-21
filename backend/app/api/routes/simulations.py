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


from typing import List, Optional

@router.get("", response_model=List[SimulationResponse])
def list_simulations(skip: int = 0, limit: int = 50, role: Optional[str] = None, db: Session = Depends(get_db)):
    """Lists recent simulations."""
    return SimulationService.list_simulations(db, skip=skip, limit=limit, role=role)


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


@router.post("/{simulation_id}/start", response_model=SimulationStatus)
def start_simulation(
    simulation_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Starts asynchronous background execution of a simulation."""
    from fastapi import HTTPException
    sim = SimulationService.get_simulation(db, simulation_id)
    if sim.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation {simulation_id} is already running",
        )
    background_tasks.add_task(SimulationWorker.start_simulation_task, simulation_id)
    return SimulationStatus(
        simulation_id=sim.id,
        run_id=sim.run_id,
        status="starting",
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
