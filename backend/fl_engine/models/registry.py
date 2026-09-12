from typing import Dict, Type, Any
from backend.fl_engine.models.base import BaseModel
from backend.fl_engine.models.cnn import CNN
from backend.fl_engine.models.mlp import MLP
from backend.fl_engine.models.resnet import SmallResNet
from backend.fl_engine.models.fedxneuro import FedXNeuroModel


class ModelRegistry:
    """Registry for neural network architectures."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseModel]) -> None:
        cls._registry[name.lower()] = model_cls

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseModel:
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Model '{name}' not found in registry. Available models: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())


# Pre-register supported models
ModelRegistry.register("cnn", CNN)
ModelRegistry.register("mlp", MLP)
ModelRegistry.register("resnet", SmallResNet)
ModelRegistry.register("fedxneuro", FedXNeuroModel)
ModelRegistry.register("fed_xneuro", FedXNeuroModel)
