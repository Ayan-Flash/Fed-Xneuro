from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator


class FedAvg(BaseFederatedAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm (McMahan et al., 2017).
    Aggregates client model updates via sample-weighted parameter averaging.
    """

    def __init__(self) -> None:
        super().__init__(name="FedAvg")
        self.aggregator = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client models using FedAvg.
        """
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]

        return self.aggregator.aggregate(client_parameters, client_sample_counts)
