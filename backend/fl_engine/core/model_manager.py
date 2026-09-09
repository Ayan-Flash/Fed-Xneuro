import copy
from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel
from backend.fl_engine.models.registry import ModelRegistry
from backend.fl_engine.models.connector import ModelConnector
from backend.fl_engine.utils.serialization import save_state_dict, load_state_dict


class ModelManager:
    """
    Central manager for model creation, copying, parameter extraction,
    and checkpoint saving/loading, with support for external model endpoints.
    """

    @staticmethod
    def create_model(model_name: str, **kwargs: Any) -> BaseModel:
        """Instantiates a model architecture from the ModelRegistry."""
        return ModelRegistry.get(model_name, **kwargs)

    @staticmethod
    def clone_model(model: nn.Module) -> nn.Module:
        """Deepcopies a PyTorch model."""
        return copy.deepcopy(model)

    @staticmethod
    def get_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts a detached CPU copy of the model parameters."""
        if hasattr(model, "get_parameters"):
            return model.get_parameters()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def set_parameters(model: nn.Module, parameters: Dict[str, torch.Tensor]) -> None:
        """Loads parameters into a model."""
        if hasattr(model, "set_parameters"):
            model.set_parameters(parameters)
        else:
            model.load_state_dict(parameters, strict=True)

    @staticmethod
    def save_model(model: nn.Module, file_path: str) -> str:
        """Saves model weights to a file path."""
        state = ModelManager.get_parameters(model)
        return save_state_dict(state, file_path)

    @staticmethod
    def load_model(
        model: nn.Module,
        file_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> nn.Module:
        """
        Loads model weights from a local or external file path via ModelConnector.
        """
        return ModelConnector.load_external_weights(model, file_path, device=device, strict=strict)

    @staticmethod
    def register_external_architecture(
        file_path: str,
        class_name: str,
        register_as: str
    ) -> Type[nn.Module]:
        """
        Connects an external Python model class file to the ModelRegistry.
        """
        return ModelConnector.load_external_model_class(file_path, class_name, register_as=register_as)
