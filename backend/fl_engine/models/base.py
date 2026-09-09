from abc import ABC, abstractmethod
from typing import Dict
import copy
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models in PS32 FL Simulator."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        pass

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns a detached, CPU deepcopy of the model's state_dict."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Loads state_dict into the model."""
        self.load_state_dict(parameters, strict=True)

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Returns the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
