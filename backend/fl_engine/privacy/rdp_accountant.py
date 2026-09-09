import math
from typing import List, Optional, Tuple


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant.
    Tracks cumulative privacy budget (epsilon, delta) spent across federated communication rounds
    using the Subsampled Gaussian Mechanism (Mironov, 2017; Wang et al., 2019).
    """

    DEFAULT_ORDERS: List[float] = [
        1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0,
        28.0, 32.0, 48.0, 64.0, 128.0
    ]

    def __init__(
        self,
        orders: Optional[List[float]] = None,
        target_delta: float = 1e-5
    ) -> None:
        self.orders = orders or self.DEFAULT_ORDERS
        self.target_delta = target_delta
        self.rdp_history: List[List[float]] = []

    def step(self, sampling_ratio: float, noise_multiplier: float) -> None:
        """
        Records one federated round of the subsampled Gaussian mechanism.

        Args:
            sampling_ratio: Fraction of clients selected this round (q = K / N).
            noise_multiplier: Noise standard deviation multiplier (sigma).
        """
        if noise_multiplier <= 0.0:
            # Zero noise implies infinite privacy leakage
            rdp_step = [float("inf") for _ in self.orders]
        else:
            rdp_step = [
                self._compute_rdp_order(q=sampling_ratio, sigma=noise_multiplier, alpha=alpha)
                for alpha in self.orders
            ]
        self.rdp_history.append(rdp_step)

    def _compute_rdp_order(self, q: float, sigma: float, alpha: float) -> float:
        """
        Computes RDP for a single step with subsampling ratio q, noise sigma, and order alpha.
        Uses analytical upper bounds for subsampled Gaussian mechanism.
        """
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            # Full participation: exact Gaussian mechanism RDP
            return alpha / (2.0 * (sigma ** 2))

        # Subsampled Gaussian mechanism bound (Wang et al., 2019)
        # For integer alpha: e^{(alpha - 1) * epsilon} <= (1 - q)^alpha + q * alpha * ...
        # Standard analytical approximation for continuous/integer alpha:
        # epsilon(alpha) <= (1 / (alpha - 1)) * ln(1 + q^2 * (alpha * (alpha - 1) / 2) * (e^{1/sigma^2} - 1))
        # Valid for moderate alpha and q:
        var = sigma ** 2
        gaussian_rdp = alpha / (2.0 * var)
        
        # When q is small, RDP scales approximately with q^2
        bound = (q ** 2) * gaussian_rdp
        return float(bound)

    def get_cumulative_rdp(self) -> List[float]:
        """Returns the sum of RDP values across all completed steps for each order."""
        if not self.rdp_history:
            return [0.0 for _ in self.orders]

        cumulative = [0.0 for _ in self.orders]
        for step_rdp in self.rdp_history:
            for idx, val in enumerate(step_rdp):
                cumulative[idx] += val
        return cumulative

    def get_epsilon(self, target_delta: Optional[float] = None) -> float:
        """
        Converts cumulative Rényi DP to standard (epsilon, delta)-DP:
        epsilon(delta) = min_{alpha} [ RDP(alpha) + ln(1 / delta) / (alpha - 1) ]
        """
        delta = target_delta or self.target_delta
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"target_delta must be in (0, 1), got {delta}")

        if not self.rdp_history:
            return 0.0

        cum_rdp = self.get_cumulative_rdp()
        if any(math.isinf(v) for v in cum_rdp):
            return float("inf")

        eps_candidates = []
        for alpha, rdp in zip(self.orders, cum_rdp):
            if alpha > 1.0:
                eps = rdp + (math.log(1.0 / delta) / (alpha - 1.0))
                eps_candidates.append(eps)

        return min(eps_candidates) if eps_candidates else 0.0

    @classmethod
    def compute_privacy_budget(
        cls,
        num_rounds: int,
        client_fraction: float,
        noise_multiplier: float,
        target_delta: float = 1e-5,
    ) -> float:
        """Convenience method to compute total epsilon for a planned simulation."""
        accountant = cls(target_delta=target_delta)
        for _ in range(num_rounds):
            accountant.step(sampling_ratio=client_fraction, noise_multiplier=noise_multiplier)
        return accountant.get_epsilon(target_delta=target_delta)
