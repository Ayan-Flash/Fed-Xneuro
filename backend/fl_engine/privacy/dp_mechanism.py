import math
from typing import Dict, List, Tuple, Optional
import torch
from backend.fl_engine.algorithms.base import ClientUpdate


def compute_param_l2_norm(parameters: Dict[str, torch.Tensor]) -> float:
    """
    Computes the total L2 norm across all floating-point tensors in a parameter dictionary.
    """
    total_sq_norm = 0.0
    for param in parameters.values():
        if param.is_floating_point():
            total_sq_norm += param.detach().pow(2).sum().item()
    return math.sqrt(total_sq_norm)


def clip_parameter_update(
    update_dict: Dict[str, torch.Tensor],
    max_norm: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Clips parameter update tensors to a maximum L2 norm threshold C.
    
    Args:
        update_dict: Dict of parameter deltas or weights.
        max_norm: Maximum allowed L2 norm (C).

    Returns:
        Tuple of (clipped_parameters_dict, original_l2_norm).
    """
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be strictly positive, got {max_norm}")

    original_norm = compute_param_l2_norm(update_dict)
    clip_coef = min(1.0, max_norm / (original_norm + 1e-12))

    clipped_dict: Dict[str, torch.Tensor] = {}
    for k, v in update_dict.items():
        if v.is_floating_point():
            clipped_dict[k] = v * clip_coef
        else:
            clipped_dict[k] = v.clone()

    return clipped_dict, original_norm


def add_gaussian_noise(
    parameters: Dict[str, torch.Tensor],
    noise_std: float,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Injects calibrated zero-mean Gaussian noise N(0, noise_std^2) to floating-point tensors.
    """
    if noise_std <= 0.0:
        return {k: v.clone() for k, v in parameters.items()}

    noised_dict: Dict[str, torch.Tensor] = {}
    for k, v in parameters.items():
        if v.is_floating_point():
            target_device = device or v.device
            noise = torch.randn_like(v, device=target_device) * noise_std
            noised_dict[k] = v.to(target_device) + noise
        else:
            noised_dict[k] = v.clone()

    return noised_dict


class DifferentialPrivacyMechanism:
    """
    Federated Differential Privacy (DP-FL) Mechanism.
    Implements Client-Level (User-Level) Differential Privacy:
    1. Clips client updates to maximum L2 sensitivity threshold C.
    2. Injects calibrated Gaussian noise scaled to C * noise_multiplier / K.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
    ) -> None:
        if clip_norm <= 0.0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if noise_multiplier < 0.0:
            raise ValueError(f"noise_multiplier cannot be negative, got {noise_multiplier}")

        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta

    def clip_client_updates(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Dict[str, torch.Tensor],
    ) -> Tuple[List[ClientUpdate], List[float]]:
        """
        Clips the pseudo-gradient update Delta_i = w_i - w_global of each client to self.clip_norm.
        
        Returns:
            Tuple of (clipped_client_updates, list_of_original_norms).
        """
        clipped_updates: List[ClientUpdate] = []
        original_norms: List[float] = []

        for update in client_updates:
            # Compute update delta: Delta_i = w_i - w_global
            delta: Dict[str, torch.Tensor] = {}
            for k, w_client in update.parameters.items():
                w_global = global_parameters[k].to(w_client.device)
                delta[k] = w_client - w_global

            clipped_delta, orig_norm = clip_parameter_update(delta, self.clip_norm)
            original_norms.append(orig_norm)

            # Reconstruct clipped client parameters: w_clipped = w_global + clipped_delta
            clipped_params: Dict[str, torch.Tensor] = {}
            for k, w_global in global_parameters.items():
                clipped_params[k] = (w_global.to(clipped_delta[k].device) + clipped_delta[k]).detach().cpu()

            clipped_updates.append(
                ClientUpdate(
                    client_id=update.client_id,
                    parameters=clipped_params,
                    num_samples=update.num_samples,
                    metrics=update.metrics,
                    control_variate_delta=update.control_variate_delta,
                )
            )

        return clipped_updates, original_norms

    def perturb_aggregated_parameters(
        self,
        aggregated_parameters: Dict[str, torch.Tensor],
        num_participating_clients: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Adds Gaussian noise to the aggregated model parameters.
        Effective noise scale = (clip_norm * noise_multiplier) / num_participating_clients.
        """
        if self.noise_multiplier == 0.0 or num_participating_clients <= 0:
            return aggregated_parameters

        effective_std = (self.clip_norm * self.noise_multiplier) / float(num_participating_clients)
        return add_gaussian_noise(aggregated_parameters, effective_std)
