from typing import Dict, List
import numpy as np
from backend.fl_engine.partitioning.base import BasePartitioner


class DirichletPartitioner(BasePartitioner):
    """
    Dirichlet Non-IID Partitioner.
    Partitions class samples across clients according to a Dirichlet distribution Dir(alpha).
    Smaller alpha leads to higher data heterogeneity (fewer classes per client).
    Larger alpha approaches IID distribution.
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        min_samples_per_client: int = 10,
        seed: int = 42
    ) -> None:
        super().__init__(num_clients=num_clients, seed=seed)
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client

    def partition(self, targets: np.ndarray) -> Dict[int, List[int]]:
        num_samples = len(targets)
        unique_classes = np.unique(targets)
        rng = np.random.default_rng(self.seed)

        client_dict: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

        # Group indices by class
        class_indices: Dict[int, np.ndarray] = {}
        for c in unique_classes:
            idx = np.where(targets == c)[0]
            rng.shuffle(idx)
            class_indices[int(c)] = idx

        # Repeat sampling if any client receives less than min_samples_per_client (or if dataset permits)
        max_retries = 50
        retry = 0
        while retry < max_retries:
            temp_dict: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

            for c in unique_classes:
                idx = class_indices[int(c)]
                # Draw Dirichlet distribution proportions
                proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))
                # Balance slightly to prevent empty slices
                proportions = np.array([
                    p * (len(temp_dict[i]) < num_samples / self.num_clients)
                    for i, p in enumerate(proportions)
                ])
                if proportions.sum() == 0:
                    proportions = rng.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = proportions / proportions.sum()

                # Calculate split points
                split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
                splits = np.split(idx, split_points)

                for client_id in range(self.num_clients):
                    temp_dict[client_id].extend(splits[client_id].tolist())

            # Check if all clients satisfy minimum requirement
            min_count = min(len(temp_dict[i]) for i in range(self.num_clients))
            if min_count >= min(self.min_samples_per_client, num_samples // (self.num_clients * 2)):
                client_dict = temp_dict
                break
            retry += 1
        else:
            # Fallback to last partition
            client_dict = temp_dict

        # Shuffle each client's indices
        for client_id in range(self.num_clients):
            rng.shuffle(client_dict[client_id])

        return client_dict
