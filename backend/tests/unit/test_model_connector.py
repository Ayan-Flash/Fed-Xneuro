import os
import tempfile
import torch
import torch.nn as nn
from backend.fl_engine.models.connector import ModelConnector
from backend.fl_engine.models.registry import ModelRegistry
from backend.fl_engine.core.model_manager import ModelManager


class SampleExternalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
        nn.init.constant_(self.fc.weight, 3.14)
        nn.init.constant_(self.fc.bias, 1.618)

    def forward(self, x):
        return self.fc(x)


def test_model_connector_save_and_load_external_weights():
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, "custom_trained_weights.pt")
        net = SampleExternalNet()

        # Export
        ModelConnector.export_weights(net, ckpt_path, extra_metadata={"epoch": 50, "loss": 0.01})
        assert os.path.exists(ckpt_path)

        # Create target model with different weights
        target_net = SampleExternalNet()
        nn.init.constant_(target_net.fc.weight, 0.0)

        # Load weights via ModelConnector
        loaded = ModelConnector.load_external_weights(target_net, ckpt_path)
        assert torch.allclose(loaded.fc.weight, torch.tensor(3.14))
        assert torch.allclose(loaded.fc.bias, torch.tensor(1.618))


def test_model_connector_dynamic_class_import():
    with tempfile.TemporaryDirectory() as temp_dir:
        py_file = os.path.join(temp_dir, "my_custom_architecture.py")
        code = """
import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel

class CustomNeuroNet(BaseModel):
    def __init__(self, in_features=10, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)
"""
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Dynamically import and register into ModelRegistry
        cls = ModelConnector.load_external_model_class(
            file_path=py_file,
            class_name="CustomNeuroNet",
            register_as="custom_neuro_net"
        )
        assert cls.__name__ == "CustomNeuroNet"

        # Instantiate from ModelRegistry
        inst = ModelRegistry.get("custom_neuro_net", in_features=8, num_classes=3)
        assert inst.linear.in_features == 8
        assert inst.linear.out_features == 3
