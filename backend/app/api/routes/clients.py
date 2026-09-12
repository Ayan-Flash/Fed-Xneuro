"""
Clients API routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.db.models.client import Client
from backend.app.schemas.client import ClientResponse

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("/{simulation_id}", response_model=List[ClientResponse])
def get_clients_for_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Lists client nodes registered for a simulation."""
    return db.query(Client).filter(Client.simulation_id == simulation_id).all()
