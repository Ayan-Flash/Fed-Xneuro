from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch


@dataclass
class ClientUpdate:
    """Encapsulates the update received from a client after local training."""
    client_id: str
    parameters: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, Any]


class BaseFederatedAlgorithm(ABC):
    """
    Abstract base class for all Federated Learning algorithms in PS32.
    Decouples server orchestration from aggregation math and algorithm-specific routines.
    """

    def __init__(self, name: str = "BaseAlgorithm") -> None:
        self.name = name

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client updates into a new global parameter state_dict.

        Args:
            client_updates: List of updates from participating clients.
            global_parameters: The current global model state_dict before aggregation.

        Returns:
            Dict containing the updated global state_dict.
        """
        pass

    def on_round_start(self, round_num: int) -> None:
        """Hook called at the start of each federated round."""
        pass

    def on_round_end(self, round_num: int, round_metrics: Dict[str, Any]) -> None:
        """Hook called at the end of each federated round."""
        pass
