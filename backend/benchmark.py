#!/usr/bin/env python
"""
PS32 Federated Learning Simulator - Multi-Algorithm Benchmark Suite.
Runs comparative evaluations across FedAvg, FedProx, FedAvgM, and SCAFFOLD
under identical non-IID conditions and generates comparative publication-quality plots.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine


def parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PS32 Multi-Algorithm Federated Learning Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "fashion_mnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp", "resnet"])
    parser.add_argument("--clients", type=int, default=5, dest="num_clients")
    parser.add_argument("--rounds", type=int, default=3, dest="num_rounds")
    parser.add_argument("--local-epochs", type=int, default=1, dest="local_epochs")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=0.01, dest="learning_rate")
    parser.add_argument("--partition", type=str, default="dirichlet", dest="partition_type")
    parser.add_argument("--alpha", type=float, default=0.1, dest="partition_alpha")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal mu")
    parser.add_argument("--momentum", type=float, default=0.9, help="FedAvgM server momentum")
    parser.add_argument(
        "--algorithms", nargs="+", default=["fedavg", "fedprox", "fedavgm", "scaffold"],
        help="List of algorithms to benchmark"
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(project_root, "results", "plots", "benchmarks"),
        help="Directory to save benchmark plots and comparison metrics"
    )
    return parser.parse_args()


def plot_benchmark_curves(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
    timestamp: str,
) -> Dict[str, str]:
    """Generates comparison plots for accuracy and loss curves."""
    os.makedirs(output_dir, exist_ok=True)
    colors = {"fedavg": "#2563eb", "fedprox": "#dc2626", "fedavgm": "#16a34a", "scaffold": "#9333ea"}
    markers = {"fedavg": "o", "fedprox": "s", "fedavgm": "^", "scaffold": "D"}

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 5), dpi=150)
    for algo_name, data in results.items():
        rounds = data.get("rounds", [])
        accuracies = data.get("accuracies", [])
        c = colors.get(algo_name.lower(), "#64748b")
        m = markers.get(algo_name.lower(), "o")
        plt.plot(rounds, accuracies, label=algo_name.upper(), color=c, marker=m, linewidth=2)

    plt.title("Federated Learning Benchmark - Global Accuracy Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#f8fafc")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f"benchmark_accuracy_{timestamp}.png")
    plt.savefig(acc_path)
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(9, 5), dpi=150)
    for algo_name, data in results.items():
        rounds = data.get("rounds", [])
        losses = data.get("losses", [])
        c = colors.get(algo_name.lower(), "#64748b")
        m = markers.get(algo_name.lower(), "o")
        plt.plot(rounds, losses, label=algo_name.upper(), color=c, marker=m, linewidth=2)

    plt.title("Federated Learning Benchmark - Global Loss Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Test Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="#f8fafc")
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"benchmark_loss_{timestamp}.png")
    plt.savefig(loss_path)
    plt.close()

    return {"accuracy_plot": acc_path, "loss_plot": loss_path}


def run_benchmarks() -> None:
    args = parse_benchmark_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60, flush=True)
    print("PS32 MULTI-ALGORITHM BENCHMARK RUNNER", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset       : {args.dataset.upper()}", flush=True)
    print(f"Model         : {args.model.upper()}", flush=True)
    print(f"Algorithms    : {', '.join(a.upper() for a in args.algorithms)}", flush=True)
    print(f"Partition     : {args.partition_type.upper()} (alpha={args.partition_alpha})", flush=True)
    print(f"Clients       : {args.num_clients}", flush=True)
    print(f"Rounds        : {args.num_rounds}", flush=True)
    print("=" * 60, flush=True)

    benchmark_data: Dict[str, Dict[str, Any]] = {}
    summary_table: List[Dict[str, Any]] = []

    for algo in args.algorithms:
        print(f"\n>>> Running Benchmark: {algo.upper()} ...", flush=True)
        config = SimulationConfig(
            dataset=args.dataset,
            model=args.model,
            algorithm=algo,
            num_clients=args.num_clients,
            client_fraction=1.0,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            partition_type=args.partition_type,
            partition_alpha=args.partition_alpha,
            seed=args.seed,
            device=args.device,
            mu=args.mu,
            server_momentum=args.momentum,
            server_lr=1.0,
            save_checkpoints=False,
        )

        engine = SimulationEngine(config)
        state, summary = engine.run()

        # Extract round histories
        rounds_list = []
        losses_list = []
        accs_list = []
        for rh in engine.metrics_manager.history:
            rounds_list.append(rh["round"])
            losses_list.append(rh["global_loss"])
            accs_list.append(rh["global_accuracy"])

        benchmark_data[algo] = {
            "rounds": rounds_list,
            "losses": losses_list,
            "accuracies": accs_list,
            "final_accuracy": summary["final_accuracy"],
            "final_loss": summary["final_loss"],
            "best_accuracy": summary["best_accuracy"],
            "total_time": summary["total_duration_sec"],
            "total_comm_bytes": summary["total_communication_bytes"],
        }

        summary_table.append({
            "algorithm": algo.upper(),
            "final_accuracy": f"{summary['final_accuracy']:.2f}%",
            "best_accuracy": f"{summary['best_accuracy']:.2f}%",
            "final_loss": f"{summary['final_loss']:.4f}",
            "time_sec": f"{summary['total_duration_sec']:.2f}s",
            "comm_kb": f"{summary['total_communication_bytes'] / 1024:.1f} KB",
        })

    # Generate comparison plots
    plot_paths = plot_benchmark_curves(benchmark_data, args.output_dir, timestamp)

    # Save benchmark summary JSON
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "dataset": args.dataset,
                "model": args.model,
                "partition": args.partition_type,
                "alpha": args.partition_alpha,
                "rounds": args.num_rounds,
                "clients": args.num_clients,
            },
            "summary_table": summary_table,
            "detailed_results": benchmark_data,
            "plots": plot_paths,
        }, f, indent=2)

    # Print markdown summary table
    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK COMPARATIVE RESULTS", flush=True)
    print("=" * 60, flush=True)
    header = f"| {'Algorithm':<12} | {'Final Acc':<10} | {'Best Acc':<10} | {'Final Loss':<10} | {'Time (s)':<10} | {'Comm (KB)':<10} |"
    print(header, flush=True)
    print("|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|", flush=True)
    for row in summary_table:
        print(f"| {row['algorithm']:<12} | {row['final_accuracy']:<10} | {row['best_accuracy']:<10} | {row['final_loss']:<10} | {row['time_sec']:<10} | {row['comm_kb']:<10} |", flush=True)
    print("=" * 60, flush=True)
    print(f"Plots saved to: {args.output_dir}", flush=True)
    print(f"Summary JSON  : {summary_path}", flush=True)


if __name__ == "__main__":
    run_benchmarks()
