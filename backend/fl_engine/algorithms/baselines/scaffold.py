from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator


class SCAFFOLD(BaseFederatedAlgorithm):
    """
    SCAFFOLD algorithm (Karimireddy et al., 2020).
    Uses control variates to correct for client drift under non-IID data.
    
    Scheduled for full control variate implementation in Phase 2.
    """

    def __init__(self) -> None:
        super().__init__(name="SCAFFOLD")
        self.server_control: Dict[str, torch.Tensor] = {}
        self.aggregator = FedAvgAggregator()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]
        return self.aggregator.aggregate(client_parameters, client_sample_counts)
