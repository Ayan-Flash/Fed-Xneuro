#!/usr/bin/env python
"""
Fed-XNeuro: Section 10 Comparative Experimental Study & Ablation Benchmark.

Executes the 9-model comparative progression specified in
docs/Fed-XNeuro_Architecture_and_Algorithm.md:
1. Baseline (Majority class baseline)
2. Clinical-only (EHR features alone)
3. MRI-only (3D structural MRI scans alone)
4. Clinical + Cognitive (EHR + Cognitive batteries)
5. Centralized Multimodal (Centralized fusion model)
6. Multimodal Transformer (Centralized longitudinal transformer)
7. Federated Transformer (Multi-hospital federated transformer without DP)
8. Federated + DP (Federated transformer with Differential Privacy Guard)
9. Fed-XNeuro (Full proposed framework with DP, imputation, and XAI)

Evaluates all models using clinical metrics:
Accuracy, Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC, Brier Score.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is on PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.fl_engine.datasets.multimodal import ADNIStyleMultimodalDataset
from backend.fl_engine.models.fedxneuro import FedXNeuroModel, ResNet3DEncoder
from backend.fl_engine.algorithms.proposed.fedxneuro import FedXNeuro, FedXNeuroClient, FedXNeuroTrainer
from backend.fl_engine.algorithms.proposed.utils import compute_clinical_metrics
from backend.fl_engine.algorithms.base import ClientUpdate
from backend.fl_engine.utils.seed import set_seed


# -------------------------------------------------------------------------
# Standalone Ablation Sub-Models
# -------------------------------------------------------------------------

class ClinicalOnlyModel(nn.Module):
    """Predicts conversion risk strictly from EHR variables."""
    def __init__(self, ehr_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ehr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, ehr: torch.Tensor) -> torch.Tensor:
        # ehr: [B, T, D] -> take mean across visits or baseline
        x = ehr.mean(dim=1)
        return self.net(x)


class MRIOnlyModel(nn.Module):
    """Predicts conversion risk strictly from 3D MRI scans."""
    def __init__(self, mri_channels: int = 1, base_ch: int = 8, embed_dim: int = 32) -> None:
        super().__init__()
        self.encoder = ResNet3DEncoder(in_channels=mri_channels, base_channels=base_ch, embedding_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        # mri: [B, T, C, D, H, W] -> evaluate baseline scan t=0
        base_scan = mri[:, 0]
        feat = self.encoder(base_scan)
        return self.head(feat)


class ClinicalCognitiveModel(nn.Module):
    """Predicts conversion risk from EHR and Cognitive assessment panels."""
    def __init__(self, cog_dim: int = 5, ehr_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cog_dim + ehr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cog: torch.Tensor, ehr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cog.mean(dim=1), ehr.mean(dim=1)], dim=-1)
        return self.net(x)


# -------------------------------------------------------------------------
# Benchmark Runner
# -------------------------------------------------------------------------

def run_ablation_benchmark(
    patients: int = 60,
    rounds: int = 2,
    epochs: int = 1,
    seed: int = 42,
    output_dir: str = os.path.join(project_root, "results", "plots", "ablation"),
    device_str: str = "auto",
) -> Dict[str, Any]:
    print("=" * 78)
    print("  FED-XNEURO: SECTION 10 COMPARATIVE EXPERIMENTAL STUDY & ABLATION")
    print("=" * 78)

    set_seed(seed)
    device = torch.device("cuda" if (device_str == "cuda" or (device_str == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"[*] Execution device: {device}")
    print(f"[*] Total Patient Cohort: {patients}")
    print(f"[*] Communication Rounds: {rounds} (Local Epochs: {epochs})")

    # Load dataset with patient-level splitting
    dataset = ADNIStyleMultimodalDataset(
        num_patients=patients,
        num_visits=4,
        mri_shape=(4, 8, 8),
        seed=seed,
    )

    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    train_subset = Subset(dataset, list(range(n_train)))
    test_subset = Subset(dataset, list(range(n_train, n_total)))

    results: Dict[str, Dict[str, Any]] = {}

    # Extract test labels for baseline comparison
    test_loader = DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    test_batch = next(iter(test_loader))
    test_targets = test_batch["label"].squeeze(-1).numpy().tolist()

    # ---------------------------------------------------------------------
    # 1. Baseline (Majority Class)
    # ---------------------------------------------------------------------
    print("\n[1/9] Evaluating Model 1: Baseline (Class-Prior / Majority)...")
    train_labels = [dataset.labels[i] for i in range(n_train)]
    prior_prob = float(np.mean(train_labels))
    preds_baseline = [prior_prob] * len(test_targets)
    m1_metrics = compute_clinical_metrics(preds_baseline, test_targets)
    results["1_Baseline"] = m1_metrics

    # ---------------------------------------------------------------------
    # 2. Clinical-only
    # ---------------------------------------------------------------------
    print("[2/9] Training & Evaluating Model 2: Clinical-Only (EHR)...")
    m2 = ClinicalOnlyModel(ehr_dim=7).to(device)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    for _ in range(epochs * 2):
        for b in train_loader:
            opt2.zero_grad()
            logits = m2(b["ehr"].to(device))
            loss = crit(logits, b["label"].to(device))
            loss.backward()
            opt2.step()

    m2.eval()
    with torch.no_grad():
        preds2 = torch.sigmoid(m2(test_batch["ehr"].to(device))).squeeze().cpu().numpy().tolist()
    results["2_Clinical_Only"] = compute_clinical_metrics(preds2, test_targets)

    # ---------------------------------------------------------------------
    # 3. MRI-only
    # ---------------------------------------------------------------------
    print("[3/9] Training & Evaluating Model 3: MRI-Only (3D ResNet)...")
    m3 = MRIOnlyModel(mri_channels=1, base_ch=4, embed_dim=16).to(device)
    opt3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
    for _ in range(epochs):
        for b in train_loader:
            opt3.zero_grad()
            logits = m3(b["mri"].to(device))
            loss = crit(logits, b["label"].to(device))
            loss.backward()
            opt3.step()

    m3.eval()
    with torch.no_grad():
        preds3 = torch.sigmoid(m3(test_batch["mri"].to(device))).squeeze().cpu().numpy().tolist()
    results["3_MRI_Only"] = compute_clinical_metrics(preds3, test_targets)

    # ---------------------------------------------------------------------
    # 4. Clinical + Cognitive
    # ---------------------------------------------------------------------
    print("[4/9] Training & Evaluating Model 4: Clinical + Cognitive...")
    m4 = ClinicalCognitiveModel(cog_dim=5, ehr_dim=7).to(device)
    opt4 = torch.optim.Adam(m4.parameters(), lr=1e-2)
    for _ in range(epochs * 2):
        for b in train_loader:
            opt4.zero_grad()
            logits = m4(b["cognitive"].to(device), b["ehr"].to(device))
            loss = crit(logits, b["label"].to(device))
            loss.backward()
            opt4.step()

    m4.eval()
    with torch.no_grad():
        preds4 = torch.sigmoid(m4(test_batch["cognitive"].to(device), test_batch["ehr"].to(device))).squeeze().cpu().numpy().tolist()
    results["4_Clinical_Cognitive"] = compute_clinical_metrics(preds4, test_targets)

    # ---------------------------------------------------------------------
    # 5. Centralized Multimodal
    # ---------------------------------------------------------------------
    print("[5/9] Training & Evaluating Model 5: Centralized Multimodal...")
    m5 = FedXNeuroModel(
        mri_channels=1, mri_base_channels=4, mri_embedding_dim=16,
        cog_dim=5, ehr_dim=7, fusion_dim=16, transformer_heads=2,
        imputer_layers=1, temporal_layers=1,
    ).to(device)
    trainer = FedXNeuroTrainer(device=device)
    trainer.train(m5, train_subset, local_epochs=epochs, batch_size=8, learning_rate=1e-3)
    eval5 = trainer.evaluate(m5, test_subset, batch_size=8)
    results["5_Centralized_Multimodal"] = compute_clinical_metrics(eval5["predictions"], eval5["targets"])

    # ---------------------------------------------------------------------
    # 6. Multimodal Transformer
    # ---------------------------------------------------------------------
    print("[6/9] Training & Evaluating Model 6: Multimodal Transformer...")
    m6 = FedXNeuroModel(
        mri_channels=1, mri_base_channels=8, mri_embedding_dim=32,
        cog_dim=5, ehr_dim=7, fusion_dim=32, transformer_heads=2,
        imputer_layers=1, temporal_layers=2,
    ).to(device)
    trainer.train(m6, train_subset, local_epochs=epochs * 2, batch_size=8, learning_rate=1e-3)
    eval6 = trainer.evaluate(m6, test_subset, batch_size=8)
    results["6_Multimodal_Transformer"] = compute_clinical_metrics(eval6["predictions"], eval6["targets"])

    # ---------------------------------------------------------------------
    # 7. Federated Transformer (FedAvg, No DP)
    # ---------------------------------------------------------------------
    print("[7/9] Training & Evaluating Model 7: Federated Transformer (FedAvg)...")
    m7 = FedXNeuroModel(
        mri_channels=1, mri_base_channels=4, mri_embedding_dim=16,
        cog_dim=5, ehr_dim=7, fusion_dim=16, transformer_heads=2,
        imputer_layers=1, temporal_layers=1,
    ).to(device)
    algo7 = FedXNeuro(use_dp=False)

    # 2 clients
    c1_data = Subset(dataset, list(range(0, n_train // 2)))
    c2_data = Subset(dataset, list(range(n_train // 2, n_train)))
    c7_1 = FedXNeuroClient("Hosp_A", c1_data, m7, device, local_epochs=epochs, batch_size=4, use_dp=False)
    c7_2 = FedXNeuroClient("Hosp_B", c2_data, m7, device, local_epochs=epochs, batch_size=4, use_dp=False)

    for r in range(rounds):
        g_params = {k: v.detach().cpu().clone() for k, v in m7.state_dict().items()}
        c7_1.set_model_parameters(g_params)
        c7_2.set_model_parameters(g_params)
        m_a = c7_1.train()
        m_b = c7_2.train()
        up = [
            ClientUpdate("Hosp_A", c7_1.get_model_parameters(), len(c1_data), m_a),
            ClientUpdate("Hosp_B", c7_2.get_model_parameters(), len(c2_data), m_b),
        ]
        agg = algo7.aggregate(up, g_params)
        m7.load_state_dict(agg)

    eval7 = trainer.evaluate(m7, test_subset, batch_size=4)
    results["7_Federated_Transformer"] = compute_clinical_metrics(eval7["predictions"], eval7["targets"])

    # ---------------------------------------------------------------------
    # 8. Federated + DP
    # ---------------------------------------------------------------------
    print("[8/9] Training & Evaluating Model 8: Federated + Differential Privacy...")
    m8 = FedXNeuroModel(
        mri_channels=1, mri_base_channels=4, mri_embedding_dim=16,
        cog_dim=5, ehr_dim=7, fusion_dim=16, transformer_heads=2,
        imputer_layers=1, temporal_layers=1,
    ).to(device)
    algo8 = FedXNeuro(use_dp=True, clip_norm=1.0, noise_multiplier=0.5)

    c8_1 = FedXNeuroClient("Hosp_A", c1_data, m8, device, local_epochs=epochs, batch_size=4, use_dp=True, clip_norm=1.0, noise_multiplier=0.5)
    c8_2 = FedXNeuroClient("Hosp_B", c2_data, m8, device, local_epochs=epochs, batch_size=4, use_dp=True, clip_norm=1.0, noise_multiplier=0.5)

    for r in range(rounds):
        g_params = {k: v.detach().cpu().clone() for k, v in m8.state_dict().items()}
        c8_1.set_model_parameters(g_params)
        c8_2.set_model_parameters(g_params)
        m_a = c8_1.train()
        m_b = c8_2.train()
        up = [
            ClientUpdate("Hosp_A", c8_1.get_model_parameters(), len(c1_data), m_a),
            ClientUpdate("Hosp_B", c8_2.get_model_parameters(), len(c2_data), m_b),
        ]
        agg = algo8.aggregate(up, g_params)
        m8.load_state_dict(agg)

    eval8 = trainer.evaluate(m8, test_subset, batch_size=4)
    results["8_Federated_DP"] = compute_clinical_metrics(eval8["predictions"], eval8["targets"])

    # ---------------------------------------------------------------------
    # 9. Fed-XNeuro (Full)
    # ---------------------------------------------------------------------
    print("[9/9] Training & Evaluating Model 9: Fed-XNeuro (Full Framework)...")
    m9 = FedXNeuroModel(
        mri_channels=1, mri_base_channels=8, mri_embedding_dim=32,
        cog_dim=5, ehr_dim=7, fusion_dim=32, transformer_heads=4,
        imputer_layers=2, temporal_layers=2,
    ).to(device)
    algo9 = FedXNeuro(use_dp=True, clip_norm=1.0, noise_multiplier=0.5)

    c9_1 = FedXNeuroClient("Hosp_A", c1_data, m9, device, local_epochs=epochs, batch_size=4, use_dp=True, clip_norm=1.0, noise_multiplier=0.5)
    c9_2 = FedXNeuroClient("Hosp_B", c2_data, m9, device, local_epochs=epochs, batch_size=4, use_dp=True, clip_norm=1.0, noise_multiplier=0.5)

    for r in range(rounds):
        g_params = {k: v.detach().cpu().clone() for k, v in m9.state_dict().items()}
        c9_1.set_model_parameters(g_params)
        c9_2.set_model_parameters(g_params)
        m_a = c9_1.train()
        m_b = c9_2.train()
        up = [
            ClientUpdate("Hosp_A", c9_1.get_model_parameters(), len(c1_data), m_a),
            ClientUpdate("Hosp_B", c9_2.get_model_parameters(), len(c2_data), m_b),
        ]
        agg = algo9.aggregate(up, g_params)
        m9.load_state_dict(agg)

    eval9 = trainer.evaluate(m9, test_subset, batch_size=4)
    results["9_FedXNeuro_Full"] = compute_clinical_metrics(eval9["predictions"], eval9["targets"])

    # ---------------------------------------------------------------------
    # Results Presentation & Export
    # ---------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("                      SECTION 10 ABLATION STUDY COMPARISON TABLE")
    print("=" * 88)
    print(f"{'Model Architecture':<28} | {'Accuracy':<9} | {'ROC-AUC':<9} | {'PR-AUC':<8} | {'Sens':<7} | {'Spec':<7} | {'Brier':<7}")
    print("-" * 88)

    markdown_rows = [
        "| Model Architecture | Accuracy | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, m in results.items():
        disp_name = model_name.replace("_", " ")
        acc = f"{m.get('accuracy', 0.0):.1f}%"
        roc = f"{m.get('roc_auc', 0.0):.4f}"
        pr = f"{m.get('pr_auc', 0.0):.4f}"
        sens = f"{m.get('sensitivity', 0.0):.3f}"
        spec = f"{m.get('specificity', 0.0):.3f}"
        brier = f"{m.get('brier_score', 0.0):.4f}"

        print(f"{disp_name:<28} | {acc:<9} | {roc:<9} | {pr:<8} | {sens:<7} | {spec:<7} | {brier:<7}")
        markdown_rows.append(f"| {disp_name} | {acc} | {roc} | {pr} | {sens} | {spec} | {brier} |")

    print("=" * 88 + "\n")

    # Save Markdown & JSON summaries
    os.makedirs(output_dir, exist_ok=True)
    summary_md_path = os.path.join(output_dir, "ablation_summary.md")
    summary_json_path = os.path.join(output_dir, "ablation_summary.json")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Section 10 Ablation Study & Comparative Results\n\n")
        f.write("\n".join(markdown_rows) + "\n")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[*] Saved Markdown summary: {summary_md_path}")
    print(f"[*] Saved JSON summary:     {summary_json_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fed-XNeuro Section 10 Ablation Benchmark")
    parser.add_argument("--patients", type=int, default=60, help="Total patient cohort size")
    parser.add_argument("--rounds", type=int, default=2, help="Federated communication rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "results", "plots", "ablation"))
    args = parser.parse_args()

    run_ablation_benchmark(
        patients=args.patients,
        rounds=args.rounds,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
