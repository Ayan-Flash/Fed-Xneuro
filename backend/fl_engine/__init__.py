"""
PS32 Federated Learning Simulation Engine.
Core simulation package for distributed PyTorch training, FedAvg aggregation,
data partitioning, and comprehensive experiment metrics.
"""

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine
from backend.fl_engine.simulation.state import SimulationState
from backend.fl_engine.core.client import FederatedClient
from backend.fl_engine.core.server import FederatedServer

__version__ = "0.1.0"
__all__ = [
    "SimulationConfig",
    "SimulationEngine",
    "SimulationState",
    "FederatedClient",
    "FederatedServer",
]
