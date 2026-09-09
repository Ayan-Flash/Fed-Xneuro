from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import os
import torch


@dataclass
class SimulationConfig:
    """Configuration parameters for a Federated Learning simulation run."""

    # Core parameters
    dataset: str = "mnist"
    model: str = "cnn"
    algorithm: str = "fedavg"

    # Client & Federation topology
    num_clients: int = 5
    client_fraction: float = 1.0
    num_rounds: int = 10
    local_epochs: int = 1

    # Optimization
    batch_size: int = 32
    learning_rate: float = 0.01

    # Partitioning
    partition_type: str = "iid"
    partition_alpha: float = 0.5

    # System & Reproducibility
    seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', 'cuda'

    # Paths
    data_dir: str = "data/raw/mnist"
    results_dir: str = "results"
    experiments_dir: str = "experiments"
    models_dir: str = "models"
    save_checkpoints: bool = True

    # Extra algorithm-specific arguments
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        """Resolves the configured device setting to a torch.device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Instantiates configuration from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """Loads configuration from a YAML file."""
        import yaml
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
