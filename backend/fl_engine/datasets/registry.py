from typing import Dict, Type, Any
from backend.fl_engine.datasets.base import BaseDatasetManager
from backend.fl_engine.datasets.mnist import MNISTDatasetManager
from backend.fl_engine.datasets.cifar10 import CIFAR10DatasetManager
from backend.fl_engine.datasets.fashion_mnist import FashionMNISTDatasetManager
from backend.fl_engine.datasets.multimodal import MultimodalDatasetManager


class DatasetRegistry:
    """Registry for datasets available in the FL simulator."""

    _registry: Dict[str, Type[BaseDatasetManager]] = {}

    @classmethod
    def register(cls, name: str, dataset_cls: Type[BaseDatasetManager]) -> None:
        cls._registry[name.lower()] = dataset_cls

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseDatasetManager:
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Dataset '{name}' not found in registry. Available datasets: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())


# Pre-register supported datasets
DatasetRegistry.register("mnist", MNISTDatasetManager)
DatasetRegistry.register("cifar10", CIFAR10DatasetManager)
DatasetRegistry.register("fashion_mnist", FashionMNISTDatasetManager)
DatasetRegistry.register("multimodal", MultimodalDatasetManager)
DatasetRegistry.register("adni_mci", MultimodalDatasetManager)
