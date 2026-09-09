from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InversionAttackSimulator:
    """
    Simulates a Gradient Inversion / Reconstruction Attack (DLG style: Zhu et al., 2019)
    to empirically demonstrate vulnerability of raw client updates and the defense
    efficacy of Differential Privacy noise injection.
    """

    @staticmethod
    def simulate_attack(
        model: nn.Module,
        true_input: torch.Tensor,
        true_label: torch.Tensor,
        enable_dp: bool = False,
        noise_std: float = 0.05,
        iterations: int = 60,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Executes a gradient reconstruction attack.
        
        Args:
            model: PyTorch classification model.
            true_input: Ground truth input sample tensor (e.g. 1x1x28x28 or 1xFeatures).
            true_label: Ground truth label tensor.
            enable_dp: If True, injects Gaussian DP noise to true gradient.
            noise_std: Standard deviation of DP noise if enabled.
            iterations: Number of reconstruction optimization iterations.
            learning_rate: Learning rate for dummy input optimization.

        Returns:
            Dictionary containing initial_mse, final_mse, defense_effective boolean.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # 1. Compute true gradient on private data
        model.zero_grad()
        out = model(true_input)
        loss = criterion(out, true_label)
        true_grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        true_grads = [g.detach().clone() for g in true_grads]

        # 2. Apply DP noise defense if requested
        if enable_dp and noise_std > 0.0:
            defended_grads = []
            for g in true_grads:
                noise = torch.randn_like(g) * noise_std
                defended_grads.append(g + noise)
            target_grads = defended_grads
        else:
            target_grads = true_grads

        # 3. Attacker initializes dummy input to optimize
        dummy_input = torch.randn_like(true_input, requires_grad=True)
        initial_mse = F.mse_loss(dummy_input.detach(), true_input).item()

        optimizer = torch.optim.LBFGS([dummy_input], lr=learning_rate) if iterations < 20 else torch.optim.Adam([dummy_input], lr=learning_rate)

        # 4. Attacker attempts to reconstruct input by matching gradients
        for _ in range(iterations):
            def closure():
                optimizer.zero_grad()
                model.zero_grad()
                dummy_out = model(dummy_input)
                dummy_loss = criterion(dummy_out, true_label)
                dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

                grad_diff = 0.0
                for dg, tg in zip(dummy_grads, target_grads):
                    grad_diff = grad_diff + ((dg - tg) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()

        final_mse = F.mse_loss(dummy_input.detach(), true_input).item()

        # Defense is effective if MSE remains elevated (e.g. > 0.05) or does not collapse to zero
        defense_effective = (final_mse > 0.05) if enable_dp else (final_mse < initial_mse)

        return {
            "enable_dp": enable_dp,
            "noise_std": noise_std if enable_dp else 0.0,
            "initial_mse": float(initial_mse),
            "final_mse": float(final_mse),
            "defense_effective": bool(defense_effective),
            "iterations": iterations,
        }
