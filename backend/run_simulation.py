#!/usr/bin/env python
"""
PS32 Federated Learning Simulator - CLI Entry Point.
Executes federated simulations with custom or default configurations.
"""

import os
import sys
import argparse

# Ensure project root is on PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PS32 Federated Learning Simulation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML experiment configuration file"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "cifar10", "fashion_mnist"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model", type=str, default="cnn",
        choices=["cnn", "mlp", "resnet"],
        help="Model architecture"
    )
    parser.add_argument(
        "--algorithm", type=str, default="fedavg",
        choices=["fedavg", "fedprox", "fedavgm", "scaffold"],
        help="Federated aggregation algorithm"
    )
    parser.add_argument(
        "--clients", type=int, default=5, dest="num_clients",
        help="Total number of simulated clients"
    )
    parser.add_argument(
        "--client-fraction", type=float, default=1.0, dest="client_fraction",
        help="Fraction of clients sampled per round"
    )
    parser.add_argument(
        "--rounds", type=int, default=10, dest="num_rounds",
        help="Number of federated training rounds"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1, dest="local_epochs",
        help="Number of local training epochs per client"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest="batch_size",
        help="Local training batch size"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=0.01, dest="learning_rate",
        help="Local training learning rate"
    )
    parser.add_argument(
        "--partition", type=str, default="iid", dest="partition_type",
        choices=["iid", "non_iid", "label_non_iid", "dirichlet", "unbalanced"],
        help="Data partitioning strategy across clients"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, dest="partition_alpha",
        help="Dirichlet concentration parameter (smaller alpha = higher non-IID skew)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto' selects CUDA if available)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(project_root, "data", "raw", "mnist"),
        help="Root path for raw dataset storage"
    )
    parser.add_argument(
        "--results-dir", type=str, default=os.path.join(project_root, "results"),
        help="Path for saving results, metrics, and plots"
    )
    parser.add_argument(
        "--experiments-dir", type=str, default=os.path.join(project_root, "experiments"),
        help="Path for saving run manifests and configs"
    )
    parser.add_argument(
        "--models-dir", type=str, default=os.path.join(project_root, "models"),
        help="Path for saving global and client model checkpoints"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, dest="checkpoint_path",
        help="Path to an existing model checkpoint (.pt / .pth) to resume or initialize training from"
    )
    parser.add_argument(
        "--custom-model-path", type=str, default=None, dest="custom_model_path",
        help="Path to external Python file containing a custom PyTorch model class"
    )
    parser.add_argument(
        "--custom-model-class", type=str, default=None, dest="custom_model_class",
        help="Class name of the custom model inside --custom-model-path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.config:
        config = SimulationConfig.from_yaml(args.config)
        # Apply command-line overrides if non-default
        for key, val in vars(args).items():
            if val is not None and key != "config" and hasattr(config, key):
                setattr(config, key, val)
    else:
        config = SimulationConfig(
            dataset=args.dataset,
            model=args.model,
            algorithm=args.algorithm,
            num_clients=args.num_clients,
            client_fraction=args.client_fraction,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            partition_type=args.partition_type,
            partition_alpha=args.partition_alpha,
            seed=args.seed,
            device=args.device,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            experiments_dir=args.experiments_dir,
            models_dir=args.models_dir,
            checkpoint_path=args.checkpoint_path,
            custom_model_path=args.custom_model_path,
            custom_model_class=args.custom_model_class,
        )

    engine = SimulationEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
