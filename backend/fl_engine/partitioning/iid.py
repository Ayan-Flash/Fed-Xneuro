from typing import Dict, List
import numpy as np
from backend.fl_engine.partitioning.base import BasePartitioner


class IIDPartitioner(BasePartitioner):
    """
    Independent and Identically Distributed (IID) Partitioner.
    Distributes dataset indices uniformly and evenly among clients.
    """

    def partition(self, targets: np.ndarray) -> Dict[int, List[int]]:
        num_samples = len(targets)
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(num_samples)

        # Split into approximately equal slices
        client_slices = np.array_split(shuffled_indices, self.num_clients)
        client_dict: Dict[int, List[int]] = {}
        for client_id, split in enumerate(client_slices):
            client_dict[client_id] = split.tolist()

        return client_dict
