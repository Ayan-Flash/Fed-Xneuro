import os
import shutil
import tempfile
import torch
import pytest
from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine
from backend.fl_engine.datasets.base import BaseDatasetManager
from backend.fl_engine.datasets.registry import DatasetRegistry
from torch.utils.data import TensorDataset


class MockDatasetManager(BaseDatasetManager):
    """Mock dataset for fast simulation unit testing without downloading external data."""
    def __init__(self, data_dir="data/mock"):
        super().__init__(data_dir=data_dir)
        torch.manual_seed(42)
        # 1x28x28 grayscale images matching CNN architecture
        train_x = torch.randn(20, 1, 28, 28)
        train_y = torch.randint(0, 10, (20,))
        test_x = torch.randn(10, 1, 28, 28)
        test_y = torch.randint(0, 10, (10,))
        self.train_dataset = TensorDataset(train_x, train_y)
        self.test_dataset = TensorDataset(test_x, test_y)

    def load_data(self):
        return self.train_dataset, self.test_dataset

    def get_targets(self):
        return torch.randint(0, 10, (20,)).numpy()

    def get_metadata(self):
        return {
            "name": "MockDataset",
            "num_classes": 10,
            "train_samples": 20,
            "test_samples": 10,
            "input_shape": (1, 28, 28)
        }


# Register mock dataset
DatasetRegistry.register("mock_dataset", MockDatasetManager)


def test_simulation_engine_end_to_end():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(
            dataset="mock_dataset",
            model="cnn",
            algorithm="fedavg",
            num_clients=2,
            client_fraction=1.0,
            num_rounds=2,
            local_epochs=1,
            batch_size=4,
            learning_rate=0.01,
            partition_type="iid",
            seed=42,
            device="cpu",
            results_dir=os.path.join(temp_dir, "results"),
            experiments_dir=os.path.join(temp_dir, "experiments"),
            models_dir=os.path.join(temp_dir, "models"),
            save_checkpoints=True,
        )

        engine = SimulationEngine(config)
        state, summary = engine.run()

        assert state.status == "completed"
        assert state.current_round == 2
        assert summary["total_rounds"] == 2
        assert "final_accuracy" in summary
        assert "final_loss" in summary
        assert "checkpoints_path" in summary
        assert os.path.exists(os.path.join(summary["checkpoints_path"], "final.pt"))
        assert os.path.exists(os.path.join(config.results_dir, "metrics", engine.run_id, "metrics.json"))
        assert os.path.exists(os.path.join(config.results_dir, "metrics", engine.run_id, "metrics.csv"))
