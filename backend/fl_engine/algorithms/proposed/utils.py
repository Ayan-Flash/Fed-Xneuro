"""
Utility and evaluation routines for proposed Fed-XNeuro framework.
Implements clinical metrics: ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier Score,
using pure Python/PyTorch/NumPy without requiring incompatible external binary packages.
"""

from typing import Dict, List, Any, Optional
import math
import numpy as np


def compute_clinical_metrics(
    probabilities: List[float],
    targets: List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical prediction metrics for MCI-to-AD progression:
    - Accuracy
    - Sensitivity / Recall
    - Specificity
    - F1 Score
    - ROC-AUC
    - PR-AUC
    - Brier Score
    """
    if not probabilities or not targets or len(probabilities) != len(targets):
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1_score": 0.0,
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": 0.0,
        }

    probs = np.array(probabilities, dtype=np.float64)
    trues = np.array(targets, dtype=np.int64)
    preds = (probs >= threshold).astype(np.int64)

    total = len(trues)
    tp = int(np.sum((preds == 1) & (trues == 1)))
    tn = int(np.sum((preds == 0) & (trues == 0)))
    fp = int(np.sum((preds == 1) & (trues == 0)))
    fn = int(np.sum((preds == 0) & (trues == 1)))

    accuracy = float(tp + tn) / max(1, total) * 100.0
    sensitivity = float(tp) / max(1, tp + fn) * 100.0
    specificity = float(tn) / max(1, tn + fp) * 100.0

    precision = float(tp) / max(1, tp + fp)
    recall = float(tp) / max(1, tp + fn)
    f1 = 2.0 * (precision * recall) / max(1e-8, (precision + recall))

    # Brier score: MSE of probabilities relative to actual binary outcomes
    brier = float(np.mean((probs - trues) ** 2))

    # ROC-AUC via rank-sum (Mann-Whitney U statistic)
    n_pos = int(np.sum(trues == 1))
    n_neg = int(np.sum(trues == 0))

    roc_auc = None
    pr_auc = None

    if n_pos > 0 and n_neg > 0:
        # Sort by predicted probability
        order = np.argsort(probs)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(probs) + 1)
        rank_sum_pos = np.sum(rank[trues == 1])
        u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        roc_auc = float(u_stat / (n_pos * n_neg))

        # Trapezoidal PR-AUC
        sort_desc = np.argsort(-probs)
        sorted_trues = trues[sort_desc]
        cum_tp = np.cumsum(sorted_trues == 1)
        cum_fp = np.cumsum(sorted_trues == 0)
        p_curve = cum_tp / np.maximum(1, cum_tp + cum_fp)
        r_curve = cum_tp / float(n_pos)

        # Numerical integration
        r_curve = np.concatenate(([0.0], r_curve))
        p_curve = np.concatenate(([p_curve[0] if len(p_curve) > 0 else 1.0], p_curve))
        pr_auc = float(np.sum((r_curve[1:] - r_curve[:-1]) * p_curve[1:]))

    return {
        "accuracy": round(accuracy, 2),
        "sensitivity": round(sensitivity, 2),
        "specificity": round(specificity, 2),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4),
    }
