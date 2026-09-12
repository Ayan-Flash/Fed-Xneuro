"""
Integration test for Fed-XNeuro end-to-end pipeline:
Simulated multi-hospital federated training round, DP update generation,
FedAvg aggregation, global evaluation, and clinician report generation.
"""

import pytest
import torch
from torch.utils.data import Subset

from backend.fl_engine.models.fedxneuro import FedXNeuroModel
from backend.fl_engine.datasets.multimodal import ADNIStyleMultimodalDataset
from backend.fl_engine.algorithms.proposed.fedxneuro import FedXNeuro, FedXNeuroClient, FedXNeuroTrainer
from backend.fl_engine.algorithms.base import ClientUpdate
from backend.fl_engine.algorithms.proposed.utils import compute_clinical_metrics


def test_fedxneuro_end_to_end_pipeline():
    """Simulates 2 hospitals, 1 federated training round with DP, aggregation, and XAI."""
    device = torch.device("cpu")
    num_patients = 20
    dataset = ADNIStyleMultimodalDataset(
        num_patients=num_patients,
        num_visits=3,
        mri_shape=(4, 8, 8),
        seed=123,
    )

    # 16 train, 4 test
    train_subset = Subset(dataset, list(range(16)))
    test_subset = Subset(dataset, list(range(16, 20)))

    # Hospital A (8 patients), Hospital B (8 patients)
    hosp_a_data = Subset(dataset, list(range(0, 8)))
    hosp_b_data = Subset(dataset, list(range(8, 16)))

    global_model = FedXNeuroModel(
        mri_channels=1,
        mri_base_channels=8,
        mri_embedding_dim=16,
        cog_dim=5,
        ehr_dim=7,
        fusion_dim=16,
        transformer_heads=2,
        imputer_layers=1,
        temporal_layers=1,
    ).to(device)

    algorithm = FedXNeuro(use_dp=True, clip_norm=1.0, noise_multiplier=0.5)

    client_a = FedXNeuroClient(
        client_id="Hospital_A",
        dataset=hosp_a_data,
        model=global_model,
        device=device,
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_dp=True,
    )
    client_b = FedXNeuroClient(
        client_id="Hospital_B",
        dataset=hosp_b_data,
        model=global_model,
        device=device,
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        use_dp=True,
    )

    # Broadcast global weights to clients
    global_params = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    client_a.set_model_parameters(global_params)
    client_b.set_model_parameters(global_params)

    # Local training and DP parameter update extraction
    metrics_a = client_a.train()
    dp_params_a = client_a.get_model_parameters()
    assert "train_loss" in metrics_a

    metrics_b = client_b.train()
    dp_params_b = client_b.get_model_parameters()
    assert "train_loss" in metrics_b

    updates = [
        ClientUpdate(client_id="Hospital_A", parameters=dp_params_a, num_samples=len(hosp_a_data), metrics=metrics_a),
        ClientUpdate(client_id="Hospital_B", parameters=dp_params_b, num_samples=len(hosp_b_data), metrics=metrics_b),
    ]

    # Server aggregates DP updates via sample-weighted FedAvg
    aggregated_params = algorithm.aggregate(updates, global_params)
    global_model.load_state_dict(aggregated_params)

    # Evaluate global model on hold-out test cohort
    evaluator = FedXNeuroTrainer(device=device)
    eval_res = evaluator.evaluate(global_model, test_subset, batch_size=4)
    clin_metrics = compute_clinical_metrics(eval_res["predictions"], eval_res["targets"])
    assert "accuracy" in clin_metrics
    assert "brier_score" in clin_metrics

    # On-device clinician explanation generation
    explanation = client_a.explain_patient(index=0)
    assert explanation["patient_id"] == hosp_a_data[0]["patient_id"]
    assert "risk_probability" in explanation
    assert "clinical_importance" in explanation
    assert "hippocampus_importance_pct" in explanation["mri_attribution"]
