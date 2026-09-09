import os
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from backend.fl_engine.datasets.base import BaseDatasetManager


class FashionMNISTDatasetManager(BaseDatasetManager):
    """Manager for the Fashion-MNIST dataset."""

    def __init__(self, data_dir: str = "data/raw/fashion_mnist") -> None:
        super().__init__(data_dir=data_dir)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

    def load_data(self) -> Tuple[Dataset, Dataset]:
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_dataset = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        return self.train_dataset, self.test_dataset

    def get_targets(self) -> np.ndarray:
        if self.train_dataset is None:
            self.load_data()
        targets = self.train_dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)

    def get_metadata(self) -> Dict[str, Any]:
        if self.train_dataset is None or self.test_dataset is None:
            self.load_data()
        return {
            "name": "FashionMNIST",
            "num_classes": self.num_classes,
            "train_samples": len(self.train_dataset),
            "test_samples": len(self.test_dataset),
            "input_shape": (1, 28, 28),
            "data_dir": os.path.abspath(self.data_dir)
        }
