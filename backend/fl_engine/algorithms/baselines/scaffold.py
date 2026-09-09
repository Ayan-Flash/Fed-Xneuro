from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator


class SCAFFOLD(BaseFederatedAlgorithm):
    """
    SCAFFOLD algorithm (Karimireddy et al., 2020).
    Uses control variates to correct for client drift under non-IID data distributions.
    """

    def __init__(self, server_lr: float = 1.0) -> None:
        super().__init__(name="SCAFFOLD")
        self.server_lr = server_lr
        self.server_control: Dict[str, torch.Tensor] = {}
        self.aggregator = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("No client updates to aggregate.")

        # 1. Aggregate client model parameters (weighted average)
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]
        avg_params = self.aggregator.aggregate(client_parameters, client_sample_counts)

        # 2. Server learning rate scaling if configured
        if global_parameters is not None and self.server_lr != 1.0:
            new_params: Dict[str, torch.Tensor] = {}
            for k in avg_params.keys():
                new_params[k] = global_parameters[k] + self.server_lr * (avg_params[k] - global_parameters[k])
        else:
            new_params = avg_params

        # 3. Accumulate control variate deltas: c = c + (1 / |S|) * sum(delta_c_i)
        num_updates = len(client_updates)
        has_deltas = any(u.control_variate_delta is not None for u in client_updates)
        if has_deltas:
            for update in client_updates:
                if update.control_variate_delta:
                    for k, delta in update.control_variate_delta.items():
                        delta_cpu = delta.detach().cpu()
                        if k not in self.server_control:
                            self.server_control[k] = torch.zeros_like(delta_cpu)
                        self.server_control[k] += delta_cpu / num_updates

        return new_params
