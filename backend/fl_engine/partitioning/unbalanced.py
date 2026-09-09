from typing import Dict, List
import numpy as np
from backend.fl_engine.partitioning.base import BasePartitioner


class UnbalancedPartitioner(BasePartitioner):
    """
    Unbalanced Partitioner.
    Distributes different amounts of training data to different clients.
    Uses a log-normal distribution or Dirichlet weights to generate uneven partition sizes.
    """

    def __init__(self, num_clients: int, imbalance_factor: float = 0.5, seed: int = 42) -> None:
        super().__init__(num_clients=num_clients, seed=seed)
        self.imbalance_factor = imbalance_factor

    def partition(self, targets: np.ndarray) -> Dict[int, List[int]]:
        num_samples = len(targets)
        rng = np.random.default_rng(self.seed)

        # Generate uneven proportions using log-normal distribution
        raw_weights = rng.lognormal(mean=0.0, sigma=self.imbalance_factor, size=self.num_clients)
        proportions = raw_weights / np.sum(raw_weights)

        # Compute sample counts per client ensuring at least 1 sample per client
        sample_counts = np.maximum(1, np.floor(proportions * num_samples).astype(int))
        # Adjust remainder
        diff = num_samples - np.sum(sample_counts)
        for i in range(abs(diff)):
            idx = i % self.num_clients
            sample_counts[idx] += 1 if diff > 0 else -1

        # Shuffle indices
        shuffled_indices = rng.permutation(num_samples)

        # Split
        split_points = np.cumsum(sample_counts)[:-1]
        splits = np.split(shuffled_indices, split_points)

        client_dict: Dict[int, List[int]] = {}
        for client_id in range(self.num_clients):
            client_dict[client_id] = splits[client_id].tolist()

        return client_dict
