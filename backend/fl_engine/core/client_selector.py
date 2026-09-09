from typing import List
import numpy as np


class RandomClientSelector:
    """
    Selects a fraction of participating clients uniformly at random per round
    with reproducible pseudo-random state.
    """

    def __init__(self, client_fraction: float = 1.0, seed: int = 42) -> None:
        if not (0.0 < client_fraction <= 1.0):
            raise ValueError(f"client_fraction must be in (0.0, 1.0], got {client_fraction}")
        self.client_fraction = client_fraction
        self.rng = np.random.default_rng(seed)

    def select_clients(self, client_ids: List[str]) -> List[str]:
        """
        Selects a random subset of client IDs.

        Args:
            client_ids: List of available client identifiers.

        Returns:
            List of selected client identifiers.
        """
        num_total = len(client_ids)
        if num_total == 0:
            return []

        # Number of clients to select: at least 1, up to num_total
        num_selected = max(1, int(round(num_total * self.client_fraction)))
        num_selected = min(num_selected, num_total)

        selected = self.rng.choice(client_ids, size=num_selected, replace=False)
        return selected.tolist()
