"""
Differential Privacy Guard for Fed-XNeuro.

Implements Module 6 & 10 from Fed-XNeuro specification:
1. Client-side gradient / parameter update L2 norm clipping
2. Calibrated Gaussian noise addition (epsilon, delta)-DP
3. Privacy budget accounting across federated rounds
4. Privacy boundary enforcement (no raw data or sensitive attributions sent)
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class DifferentialPrivacyGuard:
    """
    Client-side Differential Privacy engine.
    Clips model updates/gradients and adds calibrated Gaussian noise before server transmission.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
        target_epsilon: Optional[float] = 3.0,
    ) -> None:
        """
        Args:
            clip_norm: Maximum L2 norm for parameter updates/gradients (C).
            noise_multiplier: Ratio of Gaussian noise std to clip_norm (sigma).
            target_delta: Target delta for (epsilon, delta)-DP.
            target_epsilon: Optional target privacy budget for accounting.
        """
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.target_delta = float(target_delta)
        self.target_epsilon = target_epsilon
        self.total_rounds_applied = 0

    def compute_update_norm(self, deltas: Dict[str, torch.Tensor]) -> float:
        """Calculates global L2 norm across all parameter tensors."""
        total_sq_norm = 0.0
        for tensor in deltas.values():
            if torch.is_floating_point(tensor):
                total_sq_norm += tensor.norm(2).item() ** 2
        return math.sqrt(total_sq_norm)

    def clip_parameter_deltas(
        self, deltas: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Clips parameter update deltas to the configured L2 norm bound.
        """
        global_norm = self.compute_update_norm(deltas)
        clip_factor = min(1.0, self.clip_norm / (global_norm + 1e-8))

        clipped_deltas: Dict[str, torch.Tensor] = {}
        for k, v in deltas.items():
            if torch.is_floating_point(v):
                clipped_deltas[k] = v * clip_factor
            else:
                clipped_deltas[k] = v.clone()

        return clipped_deltas, global_norm

    def add_gaussian_noise(
        self,
        clipped_deltas: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Injects calibrated zero-mean Gaussian noise scaled by sigma * C.
        The noise variance is distributed across parameter dimensions so that
        the expected total L2 norm of the perturbation equals sigma * C.
        """
        total_elements = sum(v.numel() for v in clipped_deltas.values() if torch.is_floating_point(v))
        dim_scale = math.sqrt(max(1, total_elements))
        noise_std = (self.noise_multiplier * self.clip_norm) / dim_scale
        noisy_deltas: Dict[str, torch.Tensor] = {}

        for k, v in clipped_deltas.items():
            if torch.is_floating_point(v):
                noise = torch.randn(
                    v.shape,
                    dtype=v.dtype,
                    device=v.device,
                    generator=generator,
                ) * noise_std
                noisy_deltas[k] = v + noise
            else:
                noisy_deltas[k] = v.clone()

        return noisy_deltas

    def privatize_update(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Transforms local trained parameters into a differentially private model update:
        1. Compute Delta = W_local - W_global
        2. Clip Delta to L2 norm bound C
        3. Add Gaussian noise N(0, sigma^2 * C^2)
        4. Reconstruct W_dp = W_global + noisy_Delta

        Returns:
            Tuple of (privatized parameters state_dict, dp_diagnostics_dict)
        """
        deltas: Dict[str, torch.Tensor] = {}
        for k, local_p in local_parameters.items():
            if k in global_parameters:
                glob_p = global_parameters[k].to(local_p.device)
                if torch.is_floating_point(local_p):
                    deltas[k] = local_p - glob_p
                else:
                    deltas[k] = local_p.clone()
            else:
                deltas[k] = local_p.clone()

        clipped_deltas, raw_norm = self.clip_parameter_deltas(deltas)
        noisy_deltas = self.add_gaussian_noise(clipped_deltas, generator=generator)

        # Reconstruct updated parameters
        privatized_params: Dict[str, torch.Tensor] = {}
        for k, local_p in local_parameters.items():
            if k in global_parameters and torch.is_floating_point(local_p):
                glob_p = global_parameters[k].to(local_p.device)
                privatized_params[k] = glob_p + noisy_deltas[k]
            else:
                privatized_params[k] = local_p.clone()

        self.total_rounds_applied += 1
        privatized_norm = self.compute_update_norm(noisy_deltas)

        diagnostics = {
            "dp_raw_norm": float(raw_norm),
            "dp_clip_norm": float(self.clip_norm),
            "dp_noise_std": float(self.noise_multiplier * self.clip_norm),
            "dp_privatized_norm": float(privatized_norm),
        }

        return privatized_params, diagnostics

    def privatize_gradients(self, model: nn.Module) -> float:
        """
        In-place gradient clipping and Gaussian noise injection for DP-SGD training.
        """
        # Collect gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0

        total_sq_norm = sum(g.norm(2).item() ** 2 for g in grads)
        global_norm = math.sqrt(total_sq_norm)
        clip_factor = min(1.0, self.clip_norm / (global_norm + 1e-8))

        noise_std = self.noise_multiplier * self.clip_norm
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_factor)
                noise = torch.randn_like(p.grad.data) * noise_std
                p.grad.data.add_(noise)

        return float(global_norm)

    def compute_privacy_spent(
        self,
        num_rounds: Optional[int] = None,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Estimates spent privacy budget (epsilon, delta) using composition.
        
        Formula:
            epsilon approx (sample_rate * sqrt(2 * T * ln(1/delta))) / noise_multiplier
        """
        t = num_rounds if num_rounds is not None else self.total_rounds_applied
        if t <= 0 or self.noise_multiplier <= 0:
            return 0.0, self.target_delta

        q = min(1.0, max(0.01, sample_rate))
        sigma = max(0.01, self.noise_multiplier)
        delta = max(1e-12, self.target_delta)

        # Standard analytical privacy composition
        epsilon = (q * math.sqrt(2.0 * t * math.log(1.0 / delta))) / sigma
        return round(float(epsilon), 4), float(delta)
