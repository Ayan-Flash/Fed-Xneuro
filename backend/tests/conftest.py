import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class TinyLinearModel(nn.Module):
    """Minimal linear model for fast, deterministic unit testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=True)
        # Fix weights for deterministic tests
        nn.init.constant_(self.fc.weight, 1.0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)

    def get_parameters(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def set_parameters(self, params):
        self.load_state_dict(params, strict=True)


@pytest.fixture
def tiny_model():
    return TinyLinearModel()


@pytest.fixture
def synthetic_dataset():
    # 40 samples, 4 features, 2 classes (0 or 1)
    torch.manual_seed(42)
    x = torch.randn(40, 4)
    y = torch.randint(0, 2, (40,))
    return TensorDataset(x, y)


@pytest.fixture
def synthetic_targets():
    torch.manual_seed(42)
    return torch.randint(0, 5, (100,)).numpy()
