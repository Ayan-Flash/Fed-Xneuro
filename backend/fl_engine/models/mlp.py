import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron for MNIST classification.

    Architecture:
        Input: 784 (flattened 28x28)
        Linear(784 -> 256) + ReLU + Dropout(0.2)
        Linear(256 -> 128) + ReLU
        Linear(128 -> 10)
    """

    def __init__(self, input_dim: int = 784, hidden_dim1: int = 256, hidden_dim2: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
