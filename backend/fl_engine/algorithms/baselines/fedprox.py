from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator


class FedProx(BaseFederatedAlgorithm):
    """
    Federated Proximal (FedProx) algorithm (Li et al., 2020).
    Adds a proximal term (mu / 2) * ||w - w_global||^2 to client loss to stabilize non-IID training.
    
    Note: Full local proximal loss regularization loop scheduled for Phase 2.
    """

    def __init__(self, mu: float = 0.01) -> None:
        super().__init__(name="FedProx")
        self.mu = mu
        self.aggregator = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("No client updates to aggregate.")
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]
        return self.aggregator.aggregate(client_parameters, client_sample_counts)
