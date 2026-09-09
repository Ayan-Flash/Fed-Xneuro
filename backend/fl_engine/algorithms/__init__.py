from typing import Dict, Type, Any
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.algorithms.baselines.fedavg import FedAvg
from backend.fl_engine.algorithms.baselines.fedprox import FedProx
from backend.fl_engine.algorithms.baselines.fedavgm import FedAvgM
from backend.fl_engine.algorithms.baselines.scaffold import SCAFFOLD

ALGORITHM_REGISTRY: Dict[str, Type[BaseFederatedAlgorithm]] = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedavgm": FedAvgM,
    "scaffold": SCAFFOLD,
}


def create_algorithm(name: str, **kwargs: Any) -> BaseFederatedAlgorithm:
    """Factory function for FL algorithms."""
    key = name.lower()
    if key not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{name}' not recognized. Available: {available}")
    return ALGORITHM_REGISTRY[key](**kwargs)


__all__ = [
    "BaseFederatedAlgorithm",
    "ClientUpdate",
    "FedAvg",
    "FedProx",
    "FedAvgM",
    "SCAFFOLD",
    "create_algorithm",
    "ALGORITHM_REGISTRY",
]
