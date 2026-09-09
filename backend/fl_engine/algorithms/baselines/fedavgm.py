from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator


class FedAvgM(BaseFederatedAlgorithm):
    """
    FedAvg with Server Momentum (FedAvgM) algorithm (Hsu et al., 2019).
    Applies momentum to the global pseudo-gradients calculated across rounds.
    """

    def __init__(self, server_momentum: float = 0.9, server_lr: float = 1.0) -> None:
        super().__init__(name="FedAvgM")
        self.server_momentum = server_momentum
        self.server_lr = server_lr
        self.momentum_buffer: Dict[str, torch.Tensor] = {}
        self.aggregator = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]
        avg_params = self.aggregator.aggregate(client_parameters, client_sample_counts)

        if global_parameters is None or self.server_momentum == 0.0:
            return avg_params

        # Compute pseudo-gradient: delta = global_parameters - avg_params
        new_params: Dict[str, torch.Tensor] = {}
        for key in avg_params.keys():
            delta = global_parameters[key] - avg_params[key]
            if key not in self.momentum_buffer:
                self.momentum_buffer[key] = torch.zeros_like(delta)
            self.momentum_buffer[key] = self.server_momentum * self.momentum_buffer[key] + delta
            new_params[key] = global_parameters[key] - self.server_lr * self.momentum_buffer[key]

        return new_params
