#!/usr/bin/env python
"""
TRAIN / Training / Fed-XNeuro Training Runner.

Executes federated or centralized training of the Fed-XNeuro Multimodal Architecture:
- 3D-CNN + Spatial-Temporal Attention (MRI)
- BiLSTM / Transformer (Cognitive Trajectories)
- Longitudinal EHR Tabular Net
- Cross-Modal Transformer Fusion + Survival/Classification Head
"""

import os
import sys
import argparse
from typing import Dict, Any

# Ensure workspace root is on PYTHONPATH
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.fl_engine.models.fedxneuro import FedXNeuroModel
from backend.fl_engine.simulation.engine import SimulationEngine
from backend.fl_engine.simulation.config import SimulationConfig
from TRAIN.datasets.adni_loader import ADNIDatasetManager
from TRAIN.datasets.oasis_loader import OASISDatasetManager


def run_training(
    dataset_name: str = "adni",
    rounds: int = 5,
    num_clients: int = 3,
    fraction_fit: float = 1.0,
    local_epochs: int = 1,
    batch_size: int = 4,
    algorithm: str = "fedavg",
    mu: float = 0.01,
    dp_enabled: bool = False,
    compression: str = "none",
    device: str = "cpu",
) -> Dict[str, Any]:
    print("=" * 75)
    print("  FED-XNEURO MULTIMODAL FEDERATED TRAINING")
    print("=" * 75)
    print(f"[*] Dataset:         {dataset_name.upper()}")
    print(f"[*] Total Rounds:    {rounds}")
    print(f"[*] Total Clients:   {num_clients}")
    print(f"[*] Algorithm:       {algorithm.upper()}")
    print(f"[*] DP-SGD:          {'Enabled' if dp_enabled else 'Disabled'}")
    print(f"[*] Compression:     {compression.upper()}")
    print(f"[*] Device:          {device}")
    print("-" * 75)

    # Resolve Dataset Manager
    train_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset_name.lower() == "oasis":
        oasis_dir = os.path.join(train_root, "data", "raw", "oasis")
        dataset_mgr = OASISDatasetManager(data_dir=oasis_dir)
        dataset_key = "oasis"
    else:
        adni_dir = os.path.join(train_root, "data", "raw", "adni")
        dataset_mgr = ADNIDatasetManager(data_dir=adni_dir)
        dataset_key = "adni"

    # Configure Simulation Engine
    sim_config = SimulationConfig(
        num_clients=num_clients,
        client_fraction=fraction_fit,
        num_rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        model="fedxneuro",
        dataset=dataset_key,
        algorithm=algorithm.lower(),
        mu=mu,
        enable_dp=dp_enabled,
        device=device,
        seed=42,
    )

    engine = SimulationEngine(config=sim_config)
    state, summary = engine.run()

    print("\n" + "=" * 75)
    print("  TRAINING COMPLETE - SUMMARY RESULTS")
    print("=" * 75)
    final_acc = summary.get("final_accuracy", 0.0)
    final_loss = summary.get("final_loss", 0.0)
    total_time = summary.get("total_duration_sec", summary.get("total_time_seconds", 0.0))

    acc_display = final_acc if final_acc > 1.0 else final_acc * 100.0
    print(f"  Final Evaluation Accuracy: {acc_display:.2f}%")
    print(f"  Final Evaluation Loss:     {final_loss:.4f}")
    print(f"  Total Duration:            {total_time:.2f}s")
    print("=" * 75)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRAIN: Run Fed-XNeuro Federated Multimodal Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="adni", choices=["adni", "oasis"], help="Dataset to train on")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of simulated hospital/clinical clients")
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold", "fedadam"], help="FL aggregation strategy")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal regularization coefficient")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy (DP-SGD)")
    parser.add_argument("--compression", type=str, default="none", choices=["none", "topk", "randk"], help="Gradient compression technique")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        dataset_name=args.dataset,
        rounds=args.rounds,
        num_clients=args.clients,
        algorithm=args.algorithm,
        mu=args.mu,
        dp_enabled=args.dp,
        compression=args.compression,
        device=args.device,
    )


if __name__ == "__main__":
    main()
