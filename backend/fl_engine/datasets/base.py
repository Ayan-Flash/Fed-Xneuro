from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from torch.utils.data import Dataset


class BaseDatasetManager(ABC):
    """Abstract base class for dataset managers in PS32 FL Simulator."""

    def __init__(self, data_dir: str = "data/raw") -> None:
        self.data_dir = data_dir
        self.train_dataset: Dataset = None
        self.test_dataset: Dataset = None

    @abstractmethod
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Loads and returns (train_dataset, test_dataset)."""
        pass

    @abstractmethod
    def get_targets(self) -> np.ndarray:
        """Returns the class labels / targets for the training dataset as a 1D NumPy array."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata regarding the dataset."""
        pass
