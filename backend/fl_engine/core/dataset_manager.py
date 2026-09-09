from typing import Dict, List, Any, Tuple
import numpy as np
from torch.utils.data import Dataset, Subset
from backend.fl_engine.datasets.registry import DatasetRegistry
from backend.fl_engine.datasets.base import BaseDatasetManager
from backend.fl_engine.partitioning import create_partitioner


class DatasetManager:
    """
    High-level Dataset Manager orchestrating dataset loading,
    partitioning among clients, and generating distribution statistics.
    """

    def __init__(
        self,
        dataset_name: str = "mnist",
        num_clients: int = 5,
        partition_type: str = "iid",
        data_dir: str = "data/raw/mnist",
        seed: int = 42,
        **partition_kwargs: Any
    ) -> None:
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.data_dir = data_dir
        self.seed = seed
        self.partition_kwargs = partition_kwargs

        self.underlying_manager: BaseDatasetManager = None
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.client_indices: Dict[int, List[int]] = {}
        self.client_datasets: Dict[int, Subset] = {}

    def setup(self) -> Tuple[Dict[int, Subset], Dataset]:
        """
        Loads the dataset, partitions training indices, and generates client subsets.

        Returns:
            Tuple of (dict of client subsets, global test dataset).
        """
        # Load dataset via registry
        self.underlying_manager = DatasetRegistry.get(self.dataset_name, data_dir=self.data_dir)
        self.train_dataset, self.test_dataset = self.underlying_manager.load_data()
        targets = self.underlying_manager.get_targets()

        # Create partitioner
        partitioner = create_partitioner(
            partition_type=self.partition_type,
            num_clients=self.num_clients,
            seed=self.seed,
            **self.partition_kwargs
        )
        self.client_indices = partitioner.partition(targets)

        # Wrap into Subsets
        self.client_datasets = {
            client_id: Subset(self.train_dataset, indices)
            for client_id, indices in self.client_indices.items()
        }

        return self.client_datasets, self.test_dataset

    def get_client_dataset(self, client_id: int) -> Subset:
        """Returns the local Dataset for the given client ID."""
        if client_id not in self.client_datasets:
            raise KeyError(f"Client {client_id} not found in client datasets.")
        return self.client_datasets[client_id]

    def get_test_dataset(self) -> Dataset:
        """Returns the global test dataset."""
        return self.test_dataset

    def get_statistics(self) -> Dict[str, Any]:
        """
        Computes detailed dataset and partition statistics.
        Includes sample count and class distribution per client.
        """
        if not self.client_indices:
            self.setup()

        targets = self.underlying_manager.get_targets()
        meta = self.underlying_manager.get_metadata()
        num_classes = meta.get("num_classes", 10)

        samples_per_client = {cid: len(indices) for cid, indices in self.client_indices.items()}
        class_distribution_per_client: Dict[int, Dict[int, int]] = {}

        for cid, indices in self.client_indices.items():
            client_targets = targets[indices]
            unique, counts = np.unique(client_targets, return_counts=True)
            dist = {int(c): 0 for c in range(num_classes)}
            for u, cnt in zip(unique, counts):
                dist[int(u)] = int(cnt)
            class_distribution_per_client[cid] = dist

        return {
            "dataset": self.dataset_name,
            "partition_type": self.partition_type,
            "total_train_samples": len(self.train_dataset),
            "total_test_samples": len(self.test_dataset),
            "num_classes": num_classes,
            "samples_per_client": samples_per_client,
            "class_distribution_per_client": class_distribution_per_client,
        }
