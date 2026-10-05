#!/usr/bin/env python
"""
Fed-XNeuro: Comprehensive Model Testing & Validation Script
Loads the Kaggle-trained checkpoint 'fedxneuro_best (1).pt', validates on
unseen holdout cohorts (OASIS and ADNI), computes publication-grade metrics,
generates explainability (XAI) scorecards, and exports diagnostic plots.
"""

import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import math
import json
import shutil
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss,
    balanced_accuracy_score, accuracy_score
)

# =========================================================================
# 1. Architecture Modules (matching kaggle_train_fedxneuro.py)
# =========================================================================

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, embedding_dim=128, layers=(1, 1, 1, 1)):
        super().__init__()
        self.in_planes = base_channels
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv3d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, stride=2)

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers_list = [Conv3DBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers_list.append(Conv3DBlock(self.in_planes, planes))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        is_seq = x.dim() == 6
        if is_seq:
            b, t, c, d, h, w = x.shape
            x = x.view(b * t, c, d, h, w)

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(self.global_pool(out), 1)
        out = self.norm(self.fc(out))

        if is_seq:
            out = out.view(b, t, self.embedding_dim)
        return out


class MultimodalFusion(nn.Module):
    def __init__(self, mri_dim=128, cog_dim=5, ehr_dim=7, fusion_dim=128, dropout=0.2):
        super().__init__()
        total_in = mri_dim + cog_dim + ehr_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, mri_feats, cog_scores, ehr_feats):
        combined = torch.cat([mri_feats, cog_scores, ehr_feats], dim=-1)
        return self.fusion(combined)


class ContinuousTemporalEncoding(nn.Module):
    def __init__(self, dim, max_period=100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)
        self.dim = dim

    def forward(self, time_gaps):
        args = time_gaps.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=emb.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class MissingVisitImputation(nn.Module):
    def __init__(self, dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.missing_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.missing_token, std=0.02)
        self.temporal_enc = ContinuousTemporalEncoding(dim)
        self.mask_indicator = nn.Embedding(2, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.imputer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, visit_mask, time_gaps):
        b, t, d = x.shape
        missing = self.missing_token.expand(b, t, d)
        m = visit_mask.unsqueeze(-1).to(x.dtype)
        h = (
            x * m + missing * (1.0 - m)
            + self.temporal_enc(time_gaps)
            + self.mask_indicator(visit_mask.long())
        )
        return self.norm(self.imputer(h))


class LongitudinalTransformer(nn.Module):
    def __init__(self, dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Sequential(
            nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.transformer(x)
        attn_weights = F.softmax(self.attn_pool(h), dim=1)
        pooled = torch.sum(h * attn_weights, dim=1)
        return self.norm(pooled), attn_weights


class RiskPredictionHead(nn.Module):
    def __init__(self, in_dim=128, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits, torch.sigmoid(logits)

    @staticmethod
    def stratify_risk(prob: float) -> str:
        if prob > 0.70:
            return "HIGH"
        elif prob > 0.30:
            return "MODERATE"
        else:
            return "LOW"


class FedXNeuroModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.mri_encoder = ResNet3DEncoder(
            in_channels=cfg.get("mri_channels", 1),
            base_channels=cfg.get("mri_base_channels", 16),
            embedding_dim=cfg.get("mri_embedding_dim", 128),
        )
        self.fusion = MultimodalFusion(
            mri_dim=cfg.get("mri_embedding_dim", 128),
            cog_dim=cfg.get("cog_dim", 5),
            ehr_dim=cfg.get("ehr_dim", 7),
            fusion_dim=cfg.get("fusion_dim", 128),
            dropout=cfg.get("dropout", 0.2),
        )
        self.imputation = MissingVisitImputation(
            dim=cfg.get("fusion_dim", 128),
            num_heads=cfg.get("transformer_heads", 4),
            num_layers=cfg.get("imputer_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
        self.temporal = LongitudinalTransformer(
            dim=cfg.get("fusion_dim", 128),
            num_heads=cfg.get("transformer_heads", 4),
            num_layers=cfg.get("temporal_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
        self.risk_head = RiskPredictionHead(
            in_dim=cfg.get("fusion_dim", 128),
            dropout=cfg.get("dropout", 0.2),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mri = batch["mri"]
        cog = batch["cognitive"]
        ehr = batch["ehr"]
        mask = batch["visit_mask"]
        gaps = batch["time_gaps"]

        mri_feats = self.mri_encoder(mri)
        fused = self.fusion(mri_feats, cog, ehr)
        reconstructed = self.imputation(fused, mask, gaps)
        disease_state, visit_attn = self.temporal(reconstructed)
        logits, probs = self.risk_head(disease_state)

        return {
            "logits": logits,
            "probabilities": probs,
            "disease_state": disease_state,
            "visit_attention": visit_attn,
            "mri_features": mri_feats,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# 2. XAI / Explainability Functions
# =========================================================================

def compute_integrated_gradients_3d(
    model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    n_steps: int = 15,
) -> np.ndarray:
    model.eval()
    mri_input = sample_batch["mri"].clone().detach().requires_grad_(False)
    baseline = torch.zeros_like(mri_input)
    accumulated_grads = torch.zeros_like(mri_input)

    for step in range(n_steps):
        alpha = float(step) / max(1, n_steps)
        interpolated = baseline + alpha * (mri_input - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        interp_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }
        interp_batch["mri"] = interpolated

        output = model(interp_batch)
        prob = output["probabilities"].sum()
        prob.backward(retain_graph=False)
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach()

    ig_attributions = (mri_input - baseline) * (accumulated_grads / n_steps)
    return ig_attributions.cpu().numpy()


def compute_clinical_attributions(
    model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    cog_input = sample_batch["cognitive"].clone().detach().requires_grad_(True)
    ehr_input = sample_batch["ehr"].clone().detach().requires_grad_(True)

    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in sample_batch.items()}
    batch_copy["cognitive"] = cog_input
    batch_copy["ehr"] = ehr_input

    output = model(batch_copy)
    prob = output["probabilities"].sum()
    prob.backward()

    cog_names = ["MMSE", "CDR-SB", "ADAS11", "ADAS13", "FAQ"]
    ehr_names = ["Age", "Gender", "EDUC", "nWBV", "eTIV", "ASF", "BloodPressure"]

    attributions = {}
    if cog_input.grad is not None:
        cog_grad = torch.abs(cog_input.grad).mean(dim=(0, 1)).cpu().numpy()
        for i, name in enumerate(cog_names):
            attributions[f"Cognitive: {name}"] = float(cog_grad[i]) if i < len(cog_grad) else 0.0

    if ehr_input.grad is not None:
        ehr_grad = torch.abs(ehr_input.grad).mean(dim=(0, 1)).cpu().numpy()
        for i, name in enumerate(ehr_names):
            attributions[f"Biomarker: {name}"] = float(ehr_grad[i]) if i < len(ehr_grad) else 0.0

    return attributions


# =========================================================================
# 3. Main Evaluation Runner
# =========================================================================

def evaluate_cohort(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    cohort_name: str,
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    all_probs = []
    all_labels = []
    patient_results = []

    with torch.no_grad():
        for s in samples:
            batch = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in s.items()
            }
            output = model(batch)
            prob = float(output["probabilities"].cpu().item())
            true_label = int(s["label"].item() if isinstance(s["label"], torch.Tensor) else s["label"])
            pid = str(s["patient_id"])
            attn = output["visit_attention"].cpu().numpy().flatten().tolist()

            pred_label = 1 if prob >= 0.5 else 0
            all_probs.append(prob)
            all_labels.append(true_label)

            patient_results.append({
                "patient_id": pid,
                "true_label": true_label,
                "predicted_prob": prob,
                "predicted_label": pred_label,
                "risk_tier": RiskPredictionHead.stratify_risk(prob),
                "correct": (pred_label == true_label),
                "visit_attention": attn,
            })

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(labels, preds)
    b_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    sensitivity = recall_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    brier = brier_score_loss(labels, probs)

    try:
        if len(np.unique(labels)) > 1:
            auc_roc = float(roc_auc_score(labels, probs))
            pr_auc = float(average_precision_score(labels, probs))
        else:
            auc_roc = 1.0 if (preds == labels).all() else 0.5
            pr_auc = 1.0 if (preds == labels).all() else 0.5
    except Exception:
        auc_roc = 0.5
        pr_auc = 0.5

    return {
        "cohort_name": cohort_name,
        "num_patients": len(samples),
        "num_converters": int(labels.sum()),
        "num_stable": int(len(labels) - labels.sum()),
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "brier_score": float(brier),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "all_labels": all_labels,
        "all_probs": all_probs,
        "patient_results": patient_results,
    }


def main():
    print("=" * 75)
    print("  🧠 FED-XNEURO MODEL EVALUATION ON HOLDOUT COHORTS")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Locate checkpoint
    ckpt_name = "fedxneuro_best (1).pt"
    ckpt_path = os.path.join(models_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        fallback_path = os.path.join(models_dir, "fedxneuro_best.pt")
        if os.path.exists(fallback_path):
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"  📂 Loaded Checkpoint: {ckpt_path}")
    print(f"     File Size: {os.path.getsize(ckpt_path) / (1024*1024):.2f} MB")

    # 2. Synchronize to standard location models/fedxneuro_best.pt
    canonical_ckpt = os.path.join(models_dir, "fedxneuro_best.pt")
    global_ckpt = os.path.join(models_dir, "global", "fedxneuro_best.pt")
    os.makedirs(os.path.join(models_dir, "global"), exist_ok=True)
    if ckpt_path != canonical_ckpt:
        shutil.copy2(ckpt_path, canonical_ckpt)
        print(f"  💾 Synchronized to canonical: {canonical_ckpt}")
    shutil.copy2(ckpt_path, global_ckpt)
    print(f"  💾 Synchronized to global:    {global_ckpt}")

    # 3. Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ⚡ Computation Device: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    history = checkpoint.get("history", {})
    best_train_auc = checkpoint.get("best_auc", None)

    print("\n" + "─" * 75)
    print("  📋 CHECKPOINT METADATA & TRAINING DETAILS")
    print("─" * 75)
    print(f"  Federated Algorithm:     {cfg.get('algorithm', 'fedavg').upper()}")
    print(f"  Participating Hospitals: {cfg.get('num_clients', 3)}")
    print(f"  Communication Rounds:    {cfg.get('num_rounds', 15)}")
    print(f"  Local Epochs/Round:      {cfg.get('local_epochs', 10)}")
    print(f"  Non-IID Dirichlet α:     {cfg.get('dirichlet_alpha', 0.5)}")
    print(f"  Differential Privacy:    {cfg.get('enable_dp', True)} (Clip={cfg.get('dp_clip_norm', 1.0)}, σ={cfg.get('dp_noise_multiplier', 0.2)})")
    if best_train_auc is not None:
        print(f"  Kaggle Train Best AUC:   {float(best_train_auc):.4f}")
    if "test_auc" in history and history["test_auc"]:
        final_test_auc = history["test_auc"][-1]
        print(f"  Final Round Test AUC:    {float(final_test_auc):.4f}")

    # 4. Instantiate Model and Load State Dict
    model = FedXNeuroModel(cfg).to(device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  🧠 FedXNeuroModel:       {model.count_parameters():,} trainable parameters loaded.")

    # 5. Load Datasets
    oasis_test_path = os.path.join(base_dir, "TRAIN", "data", "processed", "oasis", "test_cohort.pt")
    adni_test_path = os.path.join(base_dir, "TRAIN", "data", "processed", "adni", "test_cohort.pt")

    oasis_data = torch.load(oasis_test_path, map_location="cpu", weights_only=False) if os.path.exists(oasis_test_path) else []
    adni_data = torch.load(adni_test_path, map_location="cpu", weights_only=False) if os.path.exists(adni_test_path) else []

    print("\n" + "─" * 75)
    print("  🧪 EVALUATION COHORTS DETECTED")
    print("─" * 75)
    print(f"  OASIS Holdout Cohort: {len(oasis_data)} patients")
    print(f"  ADNI Holdout Cohort:  {len(adni_data)} patients (Cross-Domain Generalization)")
    combined_data = oasis_data + adni_data
    print(f"  Combined Holdout:     {len(combined_data)} patients")

    # 6. Run Evaluation
    results = {}
    if oasis_data:
        results["OASIS"] = evaluate_cohort(model, oasis_data, "OASIS Holdout", device)
    if adni_data:
        results["ADNI"] = evaluate_cohort(model, adni_data, "ADNI Cross-Domain", device)
    if combined_data:
        results["COMBINED"] = evaluate_cohort(model, combined_data, "Combined Cohorts", device)

    # 7. Print Publication Metrics Table
    print("\n" + "=" * 75)
    print("  📊 PUBLICATION-GRADE EVALUATION METRICS TABLE")
    print("=" * 75)
    header = f"{'Metric':<26} | {'OASIS Holdout':<14} | {'ADNI Generalization':<20} | {'Combined':<12}"
    print(header)
    print("─" * 75)

    metric_keys = [
        ("Cohort Size (N)", lambda r: f"{r['num_patients']} pts"),
        ("AD Converters / Stable", lambda r: f"{r['num_converters']} / {r['num_stable']}"),
        ("ROC-AUC", lambda r: f"{r['auc_roc']:.4f}"),
        ("PR-AUC (Avg Precision)", lambda r: f"{r['pr_auc']:.4f}"),
        ("Accuracy", lambda r: f"{r['accuracy']*100:.1f}%"),
        ("Balanced Accuracy", lambda r: f"{r['balanced_accuracy']*100:.1f}%"),
        ("F1-Score", lambda r: f"{r['f1_score']:.4f}"),
        ("Sensitivity (Recall)", lambda r: f"{r['sensitivity']:.4f}"),
        ("Specificity", lambda r: f"{r['specificity']:.4f}"),
        ("Precision (PPV)", lambda r: f"{r['precision']:.4f}"),
        ("Brier Calibration Score", lambda r: f"{r['brier_score']:.4f}"),
    ]

    for label, fn in metric_keys:
        o_val = fn(results["OASIS"]) if "OASIS" in results else "N/A"
        a_val = fn(results["ADNI"]) if "ADNI" in results else "N/A"
        c_val = fn(results["COMBINED"]) if "COMBINED" in results else "N/A"
        print(f"{label:<26} | {o_val:<14} | {a_val:<20} | {c_val:<12}")

    print("=" * 75)

    # 8. Print Confusion Matrix Breakdown
    for name, r in results.items():
        cm = r["confusion_matrix"]
        print(f"\n  Confusion Matrix [{r['cohort_name']}]:")
        print(f"    True Negatives (TN): {cm['TN']:2d}  |  False Positives (FP): {cm['FP']:2d}")
        print(f"    False Negatives (FN): {cm['FN']:2d}  |  True Positives (TP):  {cm['TP']:2d}")

    # 9. Patient-Level Prediction Breakdown
    print("\n" + "─" * 75)
    print("  📋 PATIENT-BY-PATIENT INFERENCE BREAKDOWN (OASIS Holdout)")
    print("─" * 75)
    print(f"  {'Patient ID':<16} | {'True Status':<12} | {'Risk Prob':<10} | {'Tier':<8} | {'Prediction':<10}")
    print("  " + "─" * 68)
    for p in results["OASIS"]["patient_results"]:
        status = "Converter" if p["true_label"] == 1 else "Stable"
        verdict = "✅ Match" if p["correct"] else "❌ Mismatch"
        print(f"  {p['patient_id']:<16} | {status:<12} | {p['predicted_prob']*100:>6.2f}%   | {p['risk_tier']:<8} | {verdict:<10}")

    # 10. Generate High-Res Diagnostic Plots
    print("\n" + "─" * 75)
    print("  🎨 GENERATING PUBLICATION PLOTS & VISUALIZATIONS")
    print("─" * 75)

    # Plot 1: ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = {"OASIS": "#2563eb", "ADNI": "#10b981", "COMBINED": "#8b5cf6"}

    for key, color in palette.items():
        if key in results:
            r = results[key]
            lbls = np.array(r["all_labels"])
            prbs = np.array(r["all_probs"])
            if len(np.unique(lbls)) > 1:
                fpr, tpr, _ = roc_curve(lbls, prbs)
                ax1.plot(fpr, tpr, color=color, lw=2.2, label=f"{r['cohort_name']} (AUC = {r['auc_roc']:.3f})")
                prec, rec, _ = precision_recall_curve(lbls, prbs)
                ax2.plot(rec, prec, color=color, lw=2.2, label=f"{r['cohort_name']} (AP = {r['pr_auc']:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC=0.50)")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=9.5)

    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (PPV)", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9.5)

    roc_pr_path = os.path.join(plots_dir, "evaluation_metrics_roc_pr.png")
    fig.tight_layout()
    fig.savefig(roc_pr_path, dpi=160)
    plt.close(fig)
    print(f"  📈 ROC & PR Curves saved to: {roc_pr_path}")

    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cohort_keys = [("OASIS", axes[0]), ("ADNI", axes[1])]
    for key, ax in cohort_keys:
        if key in results:
            r = results[key]
            cm = np.array([[r["confusion_matrix"]["TN"], r["confusion_matrix"]["FP"]],
                           [r["confusion_matrix"]["FN"], r["confusion_matrix"]["TP"]]])
            cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Stable (0)", "Converter (1)"])
            ax.set_yticklabels(["Stable (0)", "Converter (1)"])
            ax.set_xlabel("Predicted Diagnosis", fontsize=10.5, labelpad=8)
            ax.set_ylabel("True Diagnosis", fontsize=10.5)
            ax.set_title(f"{r['cohort_name']}\nAccuracy: {r['accuracy']*100:.1f}%", fontsize=11, fontweight="bold", pad=12)

    cm_path = os.path.join(plots_dir, "confusion_matrices_holdout.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)
    print(f"  📊 Confusion Matrices saved to: {cm_path}")

    # Plot 3: Risk Score Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comb_results = results["COMBINED"]["patient_results"]
    conv_probs = [p["predicted_prob"] for p in comb_results if p["true_label"] == 1]
    stable_probs = [p["predicted_prob"] for p in comb_results if p["true_label"] == 0]

    bins = np.linspace(0, 1, 15)
    ax.hist(stable_probs, bins=bins, alpha=0.7, color="#3b82f6", label=f"Stable MCI (N={len(stable_probs)})", edgecolor="black")
    ax.hist(conv_probs, bins=bins, alpha=0.7, color="#ef4444", label=f"AD Converters (N={len(conv_probs)})", edgecolor="black")
    ax.axvline(0.30, color="#f59e0b", linestyle="--", lw=1.8, label="Low/Mod Boundary (0.30)")
    ax.axvline(0.70, color="#dc2626", linestyle="--", lw=1.8, label="Mod/High Boundary (0.70)")
    ax.set_xlabel("Predicted Progression Probability P(AD)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_title("Fed-XNeuro Risk Probability Distribution (All Holdout Cohorts)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.25)

    risk_dist_path = os.path.join(plots_dir, "risk_score_distribution.png")
    fig.tight_layout()
    fig.savefig(risk_dist_path, dpi=160)
    plt.close(fig)
    print(f"  📊 Risk Distribution plot saved to: {risk_dist_path}")

    # 11. Generate Clinician XAI Scorecards for sample patients
    sample_converter = next((s for s in oasis_data if s["label"].item() > 0.5), oasis_data[0])
    sample_stable = next((s for s in oasis_data if s["label"].item() < 0.5), oasis_data[1])

    for sample, prefix in [(sample_converter, "converter"), (sample_stable, "stable")]:
        pid = sample["patient_id"]
        sample_batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }

        # Inference
        with torch.no_grad():
            out = model(sample_batch)
            pred_prob = float(out["probabilities"].item())
            attn_weights = out["visit_attention"].cpu().numpy().flatten()
            tier = RiskPredictionHead.stratify_risk(pred_prob)

        # XAI: Integrated Gradients
        ig_3d = compute_integrated_gradients_3d(model, sample_batch, n_steps=12)
        # XAI: Feature Attributions
        clin_attr = compute_clinical_attributions(model, sample_batch)

        # Build 3-panel scorecard figure
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3])

        # Panel 1: Longitudinal Visit Attention
        ax1 = fig.add_subplot(gs[0])
        visits = [f"V{v+1}" for v in range(len(attn_weights))]
        v_colors = ["#2563eb" if v < len(attn_weights)-1 else "#dc2626" for v in range(len(attn_weights))]
        ax1.bar(visits, attn_weights, color=v_colors, edgecolor="black", alpha=0.85)
        ax1.set_ylim([0, max(attn_weights) * 1.35 if max(attn_weights) > 0 else 1.0])
        ax1.set_xlabel("Longitudinal Clinical Visits", fontsize=10.5)
        ax1.set_ylabel("Attention Weight α_t", fontsize=10.5)
        ax1.set_title(f"Visit Attention Trajectory\nP(Risk)={pred_prob*100:.1f}% ({tier})", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.25)

        # Panel 2: 3D MRI Saliency Slice
        ax2 = fig.add_subplot(gs[1])
        mri_np = sample["mri"].cpu().numpy()
        # Mid axial slice of last observed visit
        last_mri = mri_np[-1, 0] # (D, H, W)
        mid_d = last_mri.shape[0] // 2
        mri_slice = last_mri[mid_d]
        ig_slice = np.abs(ig_3d[0, -1, 0, mid_d])

        ax2.imshow(mri_slice, cmap="gray", alpha=0.6)
        cax = ax2.imshow(ig_slice, cmap="hot", alpha=0.55, aspect="auto")
        plt.colorbar(cax, ax=ax2, shrink=0.7, label="|Attribution|")
        ax2.set_title("3D MRI Saliency (Axial Slice)\nHippocampus / Ventricles", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # Panel 3: Clinical Risk Factor Attributions
        ax3 = fig.add_subplot(gs[2])
        sorted_attr = sorted(clin_attr.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        feat_names = [a[0].replace("Cognitive: ", "Cog: ").replace("Biomarker: ", "Bio: ") for a in sorted_attr]
        feat_vals = [a[1] for a in sorted_attr]

        y_pos = np.arange(len(feat_names))
        bar_colors = ["#dc2626" if "Cog" in n else "#2563eb" for n in feat_names]
        ax3.barh(y_pos, feat_vals[::-1], color=bar_colors[::-1], edgecolor="black", alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(feat_names[::-1], fontsize=9)
        ax3.set_xlabel("Saliency Attribution Magnitude (|∂P/∂x|)", fontsize=10)
        ax3.set_title("Top Multimodal Risk Drivers", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.25)

        fig.suptitle(
            f"Clinician Explainability Scorecard — Patient {pid} (True: {'AD Converter' if sample['label'].item()>0.5 else 'Stable MCI'}, Pred: {tier})",
            fontsize=13, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        sc_path = os.path.join(plots_dir, f"clinician_scorecard_{prefix}_{pid}.png")
        fig.savefig(sc_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  🔬 Clinician XAI Scorecard saved to: {sc_path}")

    # 12. Save JSON Summary Report
    json_report_path = os.path.join(metrics_dir, "test_evaluation_report.json")
    json_data = {
        "checkpoint": os.path.basename(ckpt_path),
        "model_parameters": model.count_parameters(),
        "federated_algorithm": cfg.get("algorithm", "fedavg"),
        "training_rounds": cfg.get("num_rounds", 15),
        "differential_privacy": cfg.get("enable_dp", True),
        "metrics_summary": {
            k: {
                mk: mv for mk, mv in v.items()
                if mk not in ["all_labels", "all_probs", "patient_results"]
            }
            for k, v in results.items()
        },
        "oasis_patient_predictions": results["OASIS"]["patient_results"],
    }
    with open(json_report_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  📄 Full Evaluation JSON report saved to: {json_report_path}")

    print("\n" + "=" * 75)
    print("  ✅ FED-XNEURO MODEL TESTING COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
