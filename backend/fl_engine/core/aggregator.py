from typing import Dict, List, Tuple
import torch


class FedAvgAggregator:
    """
    Implements weighted model aggregation (Federated Averaging).
    
    Formula:
        W_global = sum_{k=1}^K (n_k / N) * W_k
    where:
        W_k = model parameters (state_dict) from client k
        n_k = number of training samples for client k
        N = sum_{k=1}^K n_k (total samples across all participating clients)
    """

    @staticmethod
    def aggregate(
        client_parameters: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client state_dicts using sample-weighted averaging.

        Args:
            client_parameters: List of state_dicts from participating clients.
            client_sample_counts: List of sample counts corresponding to each client.

        Returns:
            Dict containing the aggregated global state_dict.
        """
        if not client_parameters:
            raise ValueError("Cannot aggregate empty list of client parameters.")
        if len(client_parameters) != len(client_sample_counts):
            raise ValueError("Length of client_parameters and client_sample_counts must match.")

        total_samples = sum(client_sample_counts)
        if total_samples <= 0:
            raise ValueError("Total number of samples across participating clients must be > 0.")

        # Normalize weights
        weights = [n / total_samples for n in client_sample_counts]

        aggregated_dict: Dict[str, torch.Tensor] = {}
        first_dict = client_parameters[0]

        for key, ref_tensor in first_dict.items():
            if not torch.is_floating_point(ref_tensor):
                # Non-floating point tensors (e.g. BatchNorm num_batches_tracked)
                # Weighted average rounded, or selection from primary client
                # Safely round weighted sum or adopt the dominant client's counter
                weighted_sum = sum(
                    w * client_parameters[i][key].to(torch.float64)
                    for i, w in enumerate(weights)
                )
                aggregated_dict[key] = torch.round(weighted_sum).to(ref_tensor.dtype)
            else:
                # Floating point weights: weighted sum
                weighted_tensor = torch.zeros_like(ref_tensor, dtype=ref_tensor.dtype)
                for i, w in enumerate(weights):
                    param_tensor = client_parameters[i][key]
                    if param_tensor.shape != ref_tensor.shape:
                        raise ValueError(
                            f"Tensor shape mismatch for '{key}': expected {ref_tensor.shape}, got {param_tensor.shape}"
                        )
                    weighted_tensor += (param_tensor * w).to(ref_tensor.dtype)
                aggregated_dict[key] = weighted_tensor

        return aggregated_dict
