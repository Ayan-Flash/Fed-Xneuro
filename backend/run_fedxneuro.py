#!/usr/bin/env python
"""
Fed-XNeuro: Explainable Multimodal Federated Learning for Privacy-Preserving
MCI to AD Progression Prediction.

CLI Entry Point executing end-to-end simulation across simulated hospital clients.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List

# Ensure project root is on PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from backend.fl_engine.models.fedxneuro import FedXNeuroModel
from backend.fl_engine.datasets.multimodal import ADNIStyleMultimodalDataset
from backend.fl_engine.algorithms.proposed.fedxneuro import FedXNeuro, FedXNeuroClient, FedXNeuroTrainer
from backend.fl_engine.algorithms.proposed.utils import compute_clinical_metrics
from backend.fl_engine.evaluation.dashboard import ClinicianDashboard
from backend.fl_engine.utils.seed import set_seed
from torch.utils.data import Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fed-XNeuro: Multimodal Federated Learning for MCI -> AD Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clients", type=int, default=3, help="Number of hospital clients (e.g. Hospital A, B, C)")
    parser.add_argument("--rounds", type=int, default=3, help="Communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patients", type=int, default=150, help="Total patient cohort size")
    parser.add_argument("--dp", action="store_true", default=True, help="Enable Differential Privacy Guard")
    parser.add_argument("--no-dp", action="store_false", dest="dp", help="Disable Differential Privacy")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="DP gradient/update clipping bound C")
    parser.add_argument("--noise-multiplier", type=float, default=0.5, help="DP Gaussian noise multiplier sigma")
    parser.add_argument("--target-delta", type=float, default=1e-5, help="DP delta parameter")
    parser.add_argument("--explain", action="store_true", default=True, help="Generate clinician explanation reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--results-dir", type=str, default=os.path.join(project_root, "results"), help="Results directory")
    return parser.parse_args()


def run_fedxneuro_simulation(args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 70)
    print("  FED-XNEURO: EXPLAINABLE MULTIMODAL FEDERATED LEARNING SIMULATION")
    print("  Longitudinal Prediction of Progression from MCI to AD")
    print("=" * 70)

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"[*] Execution device: {device}")
    print(f"[*] Participating Hospital Clients: {args.clients}")
    print(f"[*] Communication Rounds: {args.rounds} (Local Epochs: {args.local_epochs})")
    print(f"[*] Differential Privacy: {'ENABLED' if args.dp else 'DISABLED'} (Clip: {args.clip_norm}, Noise: {args.noise_multiplier}, Delta: {args.target_delta})")

    # 1. Dataset generation with patient-level splitting (zero longitudinal leakage)
    print(f"\n[1/5] Loading longitudinal cohort ({args.patients} patients, 6 visits each)...")
    dataset = ADNIStyleMultimodalDataset(num_patients=args.patients, seed=args.seed)

    # 80/20 train/test patient split
    n_total = len(dataset)
    rng = torch.Generator().manual_seed(args.seed)
    shuffled_idx = torch.randperm(n_total, generator=rng).tolist()
    n_train = int(n_total * 0.8)
    train_idx = shuffled_idx[:n_train]
    test_idx = shuffled_idx[n_train:]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    print(f"      Patient Split: {len(train_dataset)} training patients, {len(test_dataset)} testing patients")

    # Partition training patients across hospital clients
    client_datasets: List[Subset] = []
    chunk_size = len(train_idx) // args.clients
    for i in range(args.clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < args.clients - 1 else len(train_idx)
        client_patients = [train_idx[k] for k in range(start, end)]
        client_datasets.append(Subset(dataset, client_patients))
        print(f"      Hospital {chr(65+i)} (Client {i}): {len(client_patients)} private patients")

    # 2. Model initialization
    print("\n[2/5] Initializing Global Fed-XNeuro Longitudinal Transformer Model...")
    global_model = FedXNeuroModel(
        mri_channels=1,
        mri_base_channels=16,
        mri_embedding_dim=64,
        cog_dim=5,
        ehr_dim=7,
        fusion_dim=64,
        transformer_heads=4,
        imputer_layers=2,
        temporal_layers=2,
    ).to(device)
    print(f"      Model parameter count: {global_model.get_num_parameters():,}")

    # 3. Clients & Algorithm initialization
    print("\n[3/5] Instantiating Federated Clients and Fed-XNeuro Optimizer...")
    algorithm = FedXNeuro(
        use_dp=args.dp,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        target_delta=args.target_delta,
    )

    clients: List[FedXNeuroClient] = []
    for i in range(args.clients):
        c = FedXNeuroClient(
            client_id=f"Hospital_{chr(65+i)}",
            dataset=client_datasets[i],
            model=global_model,
            device=device,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_dp=args.dp,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier,
            target_delta=args.target_delta,
        )
        clients.append(c)

    evaluator = FedXNeuroTrainer(device=device)

    # 4. Federated Communication Rounds
    print("\n[4/5] Beginning Federated Training Rounds...")
    round_history: List[Dict[str, Any]] = []

    for r in range(1, args.rounds + 1):
        r_start = time.time()
        print(f"\n--- Round {r}/{args.rounds} ---")

        # Broadcast global parameters to clients
        global_params = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        for client in clients:
            client.set_model_parameters(global_params)

        # Local training & DP updates
        client_updates = []
        for client in clients:
            t_metrics = client.train()
            dp_params = client.get_model_parameters()
            from backend.fl_engine.algorithms.base import ClientUpdate
            client_updates.append(
                ClientUpdate(
                    client_id=client.client_id,
                    parameters=dp_params,
                    num_samples=client.num_samples,
                    metrics=t_metrics,
                )
            )
            print(f"  [{client.client_id}] Train Loss: {t_metrics['train_loss']:.4f}, Train Acc: {t_metrics['train_accuracy']:.1f}%")

        # FedAvg Aggregation
        aggregated_params = algorithm.aggregate(client_updates, global_params)
        global_model.load_state_dict({k: v.to(device) for k, v in aggregated_params.items()})

        # Global Evaluation on hold-out test cohort
        test_metrics = evaluator.evaluate(global_model, test_dataset)
        clin_metrics = compute_clinical_metrics(test_metrics["predictions"], test_metrics["targets"])

        round_time = time.time() - r_start
        epsilon, delta = algorithm.privacy_accountant.compute_privacy_spent(num_rounds=r)

        print(f"  [Global Evaluation] Loss: {test_metrics['loss']:.4f} | Acc: {clin_metrics['accuracy']:.1f}% | "
              f"Sens: {clin_metrics['sensitivity']:.1f}% | Spec: {clin_metrics['specificity']:.1f}% | "
              f"ROC-AUC: {clin_metrics.get('roc_auc') or 0.0:.3f}")
        if args.dp:
            print(f"  [Differential Privacy] Spent: epsilon = {epsilon:.3f}, delta = {delta:.1e}")

        round_history.append({
            "round": r,
            "test_loss": test_metrics["loss"],
            "metrics": clin_metrics,
            "round_duration": round(round_time, 2),
            "epsilon": epsilon,
            "delta": delta,
        })

    # 5. Local Explainability & Clinician Dashboard
    dashboard_ascii = ""
    if args.explain and clients:
        print("\n[5/5] Generating Local Clinician Explanation Report...")
        demo_client = clients[0]
        report = demo_client.explain_patient(index=0)
        dashboard_ascii = ClinicianDashboard.render_ascii(report)
        print("\n" + dashboard_ascii)

        # Save HTML and JSON reports
        os.makedirs(args.results_dir, exist_ok=True)
        json_path = os.path.join(args.results_dir, "clinician_report_sample.json")
        html_path = os.path.join(args.results_dir, "clinician_dashboard.html")
        ClinicianDashboard.save_json_report(report, json_path)
        ClinicianDashboard.save_html_report(report, html_path)
        print(f"      Saved JSON report: {json_path}")
        print(f"      Saved HTML report: {html_path}")

    print("\n" + "=" * 70)
    print("  FED-XNEURO SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return {"rounds": round_history, "last_metrics": round_history[-1] if round_history else {}}


if __name__ == "__main__":
    args = parse_args()
    run_fedxneuro_simulation(args)
