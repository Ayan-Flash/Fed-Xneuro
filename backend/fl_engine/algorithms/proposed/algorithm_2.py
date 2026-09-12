"""
Algorithm 2: Fed-XNeuro-Personalized (Personalized Multimodal Federated Learning).

Implements personalized federated learning for heterogeneous medical centers:
- Aggregates shared neuroimaging and multimodal temporal representation backbones
- Preserves hospital-specific personalized risk prediction heads
- Handles client demographic shifts and scanner-specific biases
"""

from typing import Dict, List, Optional
import torch
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator
from backend.fl_engine.algorithms.proposed.privacy import DifferentialPrivacyGuard


class FedXNeuroPersonalized(BaseFederatedAlgorithm):
    """
    Personalized Fed-XNeuro Algorithm (Fed-XNeuro-P).
    Performs federated averaging exclusively on shared feature extraction and
    longitudinal temporal transformer layers, leaving personalized local risk heads
    on local hospital client devices.
    """

    def __init__(
        self,
        personalized_prefix: str = "risk_head",
        use_dp: bool = True,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
    ) -> None:
        super().__init__(name="Fed-XNeuro-Personalized")
        self.personalized_prefix = personalized_prefix
        self.use_dp = use_dp
        self.aggregator = FedAvgAggregator()
        self.privacy_accountant = DifferentialPrivacyGuard(
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
        )

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates only shared representation backbone layers across hospital clients.
        Personalized layers (e.g. risk_head) are retained locally.
        """
        # Filter out personalized layer parameters before server aggregation
        shared_client_params: List[Dict[str, torch.Tensor]] = []
        for update in client_updates:
            shared = {
                k: v for k, v in update.parameters.items()
                if not k.startswith(self.personalized_prefix)
            }
            shared_client_params.append(shared)

        client_counts = [u.num_samples for u in client_updates]
        aggregated_shared = self.aggregator.aggregate(shared_client_params, client_counts)

        # Merge with global parameters
        new_global_parameters: Dict[str, torch.Tensor] = {}
        if global_parameters is not None:
            new_global_parameters.update(global_parameters)

        new_global_parameters.update(aggregated_shared)
        return new_global_parameters


__all__ = ["FedXNeuroPersonalized"]
