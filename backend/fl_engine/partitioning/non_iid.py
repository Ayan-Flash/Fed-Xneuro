from typing import Dict, List
import numpy as np
from backend.fl_engine.partitioning.base import BasePartitioner


class LabelNonIIDPartitioner(BasePartitioner):
    """
    Label-based Non-IID Partitioner.
    Sorts samples by class label and divides them into shards, assigning
    a limited number of shards (e.g. 2 shards) to each client to create
    strong class distribution skew.
    """

    def __init__(self, num_clients: int, shards_per_client: int = 2, seed: int = 42) -> None:
        super().__init__(num_clients=num_clients, seed=seed)
        self.shards_per_client = shards_per_client

    def partition(self, targets: np.ndarray) -> Dict[int, List[int]]:
        num_samples = len(targets)
        total_shards = self.num_clients * self.shards_per_client
        shard_size = num_samples // total_shards

        # Sort indices by their class label
        sorted_indices = np.argsort(targets)

        # Truncate slightly if num_samples is not divisible by total_shards
        usable_samples = total_shards * shard_size
        truncated_indices = sorted_indices[:usable_samples]

        # Divide into shards
        shards = [
            truncated_indices[i * shard_size : (i + 1) * shard_size]
            for i in range(total_shards)
        ]

        # Permute shards deterministically
        rng = np.random.default_rng(self.seed)
        shard_ids = rng.permutation(total_shards)

        client_dict: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}
        for client_id in range(self.num_clients):
            assigned_shard_ids = shard_ids[
                client_id * self.shards_per_client : (client_id + 1) * self.shards_per_client
            ]
            for shard_id in assigned_shard_ids:
                client_dict[client_id].extend(shards[shard_id].tolist())

        # Distribute any leftover samples to clients
        leftovers = sorted_indices[usable_samples:].tolist()
        for idx, sample_idx in enumerate(leftovers):
            client_dict[idx % self.num_clients].append(sample_idx)

        return client_dict
