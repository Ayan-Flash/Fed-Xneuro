"""
Model registry service.
"""

from typing import List
from backend.fl_engine.models import ModelRegistry
from backend.app.schemas.algorithm import ModelInfo


class ModelService:
    DESCRIPTIONS = {
        "cnn": "Standard 2-layer Convolutional Neural Network with 2 Dense layers.",
        "mlp": "Multi-Layer Perceptron (3 Dense layers with ReLU).",
        "resnet": "Small ResNet with 2D residual blocks.",
        "fedxneuro": "3D ResNet18 + Multimodal Fusion + Missing-Visit Imputer + Longitudinal Transformer.",
    }

    DISPLAY_NAMES = {
        "cnn": "CNN (2D)",
        "mlp": "MLP",
        "resnet": "Small ResNet",
        "fedxneuro": "Fed-XNeuro Longitudinal Transformer",
    }

    @classmethod
    def list_models(cls) -> List[ModelInfo]:
        unique_keys = sorted(list(set(k for k in ModelRegistry.list_available() if "_" not in k)))
        res = []
        for k in unique_keys:
            res.append(
                ModelInfo(
                    name=k,
                    display_name=cls.DISPLAY_NAMES.get(k, k.upper()),
                    description=cls.DESCRIPTIONS.get(k, "Neural network architecture"),
                    architecture_type="multimodal" if k == "fedxneuro" else "vision",
                )
            )
        return res
