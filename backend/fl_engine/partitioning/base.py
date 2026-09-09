from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class BasePartitioner(ABC):
    """Abstract base class for dataset partitioning strategies in PS32."""

    def __init__(self, num_clients: int, seed: int = 42) -> None:
        self.num_clients = num_clients
        self.seed = seed

    @abstractmethod
    def partition(self, targets: np.ndarray) -> Dict[int, List[int]]:
        """
        Partitions sample indices among clients.

        Args:
            targets: 1D array of class labels for the training set.

        Returns:
            Dict mapping client_id (0 to num_clients-1) to list of sample indices.
        """
        pass
