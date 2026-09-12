"""
Client schemas.
"""

from typing import Optional
from pydantic import BaseModel


class ClientResponse(BaseModel):
    id: int
    simulation_id: int
    client_id: str
    num_samples: int
    last_accuracy: Optional[float] = None
    last_loss: Optional[float] = None

    class Config:
        from_attributes = True
