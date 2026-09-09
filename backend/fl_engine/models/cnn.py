import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel


class CNN(BaseModel):
    """
    Lightweight Convolutional Neural Network suitable for MNIST classification.
    
    Architecture:
        Input: 1 x 28 x 28
        Conv2D(1 -> 32, 3x3, pad 1) + ReLU + MaxPool(2x2) -> 32 x 14 x 14
        Conv2D(32 -> 64, 3x3, pad 1) + ReLU + MaxPool(2x2) -> 64 x 7 x 7
        Flatten -> 3136
        Linear(3136 -> 128) + ReLU
        Linear(128 -> 10)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
