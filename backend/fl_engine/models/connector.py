import os
import sys
import importlib.util
from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel
from backend.fl_engine.models.registry import ModelRegistry


class ModelConnector:
    """
    Bridge and Endpoint Connector for integrating models trained in external
    directories or standalone research environments into the PS32 FL Simulator.
    """

    @staticmethod
    def load_external_weights(
        model: nn.Module,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> nn.Module:
        """
        Loads weights from an external checkpoint (.pt, .pth, or .bin).
        Handles diverse checkpoint structures (raw state_dict, nested dict, DataParallel prefixes).

        Args:
            model: PyTorch model instance to load parameters into.
            checkpoint_path: Path to checkpoint file on disk.
            device: Target device (defaults to CPU).
            strict: Enforce exact parameter key match.

        Returns:
            The model with loaded weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        target_device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

        # Extract state_dict if checkpoint is wrapped in dictionary
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            elif "parameters" in checkpoint:
                state_dict = checkpoint["parameters"]
            else:
                state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            raise TypeError(f"Unrecognized checkpoint format at: {checkpoint_path}")

        # Strip 'module.' prefix if saved using nn.DataParallel
        cleaned_state = {}
        for k, v in state_dict.items():
            clean_key = k[7:] if k.startswith("module.") else k
            cleaned_state[clean_key] = v

        model.to(target_device)
        model.load_state_dict(cleaned_state, strict=strict)
        return model

    @staticmethod
    def load_external_model_class(
        file_path: str,
        class_name: str,
        register_as: Optional[str] = None
    ) -> Type[nn.Module]:
        """
        Dynamically imports a model class defined in an external Python file
        (e.g., from your external training directory) and optionally registers it.

        Args:
            file_path: Absolute or relative path to the Python file containing the class.
            class_name: Name of the model class (e.g. 'MCI3DTransformer').
            register_as: Name to register under in ModelRegistry (e.g. 'mci_transformer').

        Returns:
            The imported model class.
        """
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Python model file not found: {abs_path}")

        module_name = f"external_model_{os.path.splitext(os.path.basename(abs_path))[0]}"
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load specification for {abs_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {abs_path}")

        model_cls = getattr(module, class_name)

        if register_as:
            ModelRegistry.register(register_as.lower(), model_cls)

        return model_cls

    @staticmethod
    def export_weights(
        model: nn.Module,
        output_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Exports model state_dict along with optional metadata to any external directory.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        payload = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "metadata": extra_metadata or {}
        }
        torch.save(payload, output_path)
        return output_path
