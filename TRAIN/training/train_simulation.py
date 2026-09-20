#!/usr/bin/env python
"""
TRAIN / Training / General FL Simulation Runner.

Configurable runner for federated learning across computer vision (MNIST, CIFAR10, Brain MRI 2D)
and multimodal neuroimaging (ADNI, OASIS) datasets.
"""

import os
import sys
import argparse
from typing import Dict, Any

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.fl_engine.simulation.engine import SimulationEngine
from backend.fl_engine.simulation.config import SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRAIN: Federated Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="fedxneuro_multimodal", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="adni_multimodal", help="Dataset name")
    parser.add_argument("--algorithm", type=str, default="fedavg", help="FL algorithm (fedavg, fedprox, scaffold, fedadam)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of clients per round")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=4, help="Local batch size")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx mu parameter")
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--compression", type=str, default="none", help="Gradient compression (none, topk, randk)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    config = SimulationConfig(
        num_clients=args.clients,
        client_fraction=args.fraction,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        model=args.model,
        dataset=args.dataset,
        algorithm=args.algorithm,
        mu=args.mu,
        enable_dp=args.dp,
        device=args.device,
        seed=args.seed,
    )

    print("=" * 70)
    print(f"  LAUNCHING FL SIMULATION: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 70)
    engine = SimulationEngine(config=config)
    state, summary = engine.run()

    final_acc = summary.get('final_accuracy', 0.0)
    acc_display = final_acc if final_acc > 1.0 else final_acc * 100.0
    total_time = summary.get('total_duration_sec', summary.get('total_time_seconds', 0.0))

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Final Accuracy: {acc_display:.2f}%")
    print(f"  Final Loss:     {summary.get('final_loss', 0.0):.4f}")
    print(f"  Duration:       {total_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
