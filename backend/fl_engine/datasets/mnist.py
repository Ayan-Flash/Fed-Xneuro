import os
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from backend.fl_engine.datasets.base import BaseDatasetManager


class MNISTDatasetManager(BaseDatasetManager):
    """
    Manager for the MNIST dataset.
    Downloads to data/raw/mnist/ and provides train/test splits with normalization.
    """

    def __init__(self, data_dir: str = "data/raw/mnist") -> None:
        super().__init__(data_dir=data_dir)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Downloads (if necessary) and loads the MNIST dataset."""
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        return self.train_dataset, self.test_dataset

    def get_targets(self) -> np.ndarray:
        """Returns training set targets as a NumPy array."""
        if self.train_dataset is None:
            self.load_data()
        targets = self.train_dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the MNIST dataset."""
        if self.train_dataset is None or self.test_dataset is None:
            self.load_data()
        return {
            "name": "MNIST",
            "num_classes": self.num_classes,
            "train_samples": len(self.train_dataset),
            "test_samples": len(self.test_dataset),
            "input_shape": (1, 28, 28),
            "data_dir": os.path.abspath(self.data_dir)
        }
