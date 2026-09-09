import os
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from backend.fl_engine.datasets.base import BaseDatasetManager


class CIFAR10DatasetManager(BaseDatasetManager):
    """Manager for the CIFAR-10 dataset."""

    def __init__(self, data_dir: str = "data/raw/cifar10") -> None:
        super().__init__(data_dir=data_dir)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def load_data(self) -> Tuple[Dataset, Dataset]:
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)
        return self.train_dataset, self.test_dataset

    def get_targets(self) -> np.ndarray:
        if self.train_dataset is None:
            self.load_data()
        return np.array(self.train_dataset.targets)

    def get_metadata(self) -> Dict[str, Any]:
        if self.train_dataset is None or self.test_dataset is None:
            self.load_data()
        return {
            "name": "CIFAR10",
            "num_classes": self.num_classes,
            "train_samples": len(self.train_dataset),
            "test_samples": len(self.test_dataset),
            "input_shape": (3, 32, 32),
            "data_dir": os.path.abspath(self.data_dir)
        }
