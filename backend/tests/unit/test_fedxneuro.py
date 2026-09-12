"""
Unit tests for Fed-XNeuro algorithm and architecture components.
"""

import pytest
import torch
import numpy as np

from backend.fl_engine.models.fedxneuro import (
    ResNet3DEncoder,
    MultimodalFusion,
    MissingVisitImputationModule,
    LongitudinalTransformer,
    RiskPredictionHead,
    FedXNeuroModel,
)
from backend.fl_engine.algorithms.proposed.privacy import DifferentialPrivacyGuard
from backend.fl_engine.algorithms.proposed.explainability import (
    IntegratedGradientsMRI,
    ClinicalSHAPAttributor,
    LocalExplainabilityEngine,
)
from backend.fl_engine.algorithms.proposed.fedxneuro import FedXNeuro, FedXNeuroClient
from backend.fl_engine.datasets.multimodal import ADNIStyleMultimodalDataset, MultimodalDatasetManager
from backend.fl_engine.evaluation.dashboard import ClinicianDashboard


def test_resnet3d_encoder_shapes():
    """Verify 3D ResNet encoder extracts neuroimaging embeddings for single and sequence volumes."""
    encoder = ResNet3DEncoder(in_channels=1, base_channels=8, embedding_dim=32)

    # Single 3D volume [B, 1, D, H, W]
    x_single = torch.randn(2, 1, 8, 16, 16)
    out_single = encoder(x_single)
    assert out_single.shape == (2, 32)

    # Longitudinal sequence [B, T, 1, D, H, W]
    x_seq = torch.randn(2, 4, 1, 8, 16, 16)
    out_seq = encoder(x_seq)
    assert out_seq.shape == (2, 4, 32)


def test_multimodal_fusion():
    """Verify multimodal fusion concatenates and projects MRI, cognitive, and EHR features."""
    fusion = MultimodalFusion(mri_dim=32, cog_dim=5, ehr_dim=7, fusion_dim=64)
    mri = torch.randn(2, 4, 32)
    cog = torch.randn(2, 4, 5)
    ehr = torch.randn(2, 4, 7)

    fused = fusion(mri, cog, ehr)
    assert fused.shape == (2, 4, 64)


def test_missing_visit_imputation():
    """Verify missing-visit module handles masked visits and continuous time gaps."""
    imputer = MissingVisitImputationModule(dim=64, num_heads=2, num_layers=1)
    fused = torch.randn(2, 4, 64)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    time_gaps = torch.tensor([[0.0, 6.0, 12.0, 24.0], [0.0, 6.5, 12.2, 23.8]])

    reconstructed = imputer(fused, mask, time_gaps)
    assert reconstructed.shape == (2, 4, 64)
    # Ensure gradient flows through reconstructed visits
    loss = reconstructed.sum()
    loss.backward()
    assert imputer.missing_token.grad is not None


def test_longitudinal_transformer_and_risk_head():
    """Verify temporal transformer models disease trajectory and risk head predicts probabilities."""
    transformer = LongitudinalTransformer(dim=64, num_heads=2, num_layers=1)
    head = RiskPredictionHead(in_dim=64)

    x = torch.randn(2, 4, 64)
    state, attn = transformer(x)
    assert state.shape == (2, 64)
    assert attn.shape == (2, 4, 1)

    logits, probs = head(state)
    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_fedxneuro_model_end_to_end():
    """Verify unified FedXNeuroModel full forward pass and risk prediction interface."""
    model = FedXNeuroModel(
        mri_channels=1,
        mri_base_channels=8,
        mri_embedding_dim=32,
        cog_dim=5,
        ehr_dim=7,
        fusion_dim=32,
        transformer_heads=2,
        imputer_layers=1,
        temporal_layers=1,
    )

    mri = torch.randn(2, 3, 1, 8, 16, 16)
    cog = torch.randn(2, 3, 5)
    ehr = torch.randn(2, 3, 7)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    time_gaps = torch.tensor([[0.0, 6.0, 12.0], [0.0, 6.2, 11.9]])

    out = model(mri, cog, ehr, mask, time_gaps)
    assert "logits" in out
    assert "probabilities" in out
    assert out["probabilities"].shape == (2, 1)

    # Test clinical convenience method
    res = model.predict_risk(mri[:1], cog[:1], ehr[:1], mask[:1], time_gaps[:1])
    assert "probability" in res
    assert "risk_score_percent" in res
    assert res["category"] in ["LOW", "MODERATE", "HIGH"]


def test_differential_privacy_guard():
    """Verify L2 norm clipping, Gaussian noise injection, and privacy budget accounting."""
    dp = DifferentialPrivacyGuard(clip_norm=1.0, noise_multiplier=0.5, target_delta=1e-5)

    local_p = {"weight": torch.ones(10) * 5.0}
    global_p = {"weight": torch.zeros(10)}

    privatized, stats = dp.privatize_update(local_p, global_p)
    assert "dp_raw_norm" in stats
    assert stats["dp_raw_norm"] > 1.0
    # Parameter update was bounded and perturbed
    diff = privatized["weight"] - global_p["weight"]
    assert not torch.allclose(diff, local_p["weight"])

    eps, delta = dp.compute_privacy_spent(num_rounds=10)
    assert eps > 0.0
    assert delta == 1e-5


def test_local_explainability_engine():
    """Verify Integrated Gradients for MRI and SHAP for clinical features."""
    model = FedXNeuroModel(
        mri_channels=1,
        mri_base_channels=8,
        mri_embedding_dim=16,
        cog_dim=5,
        ehr_dim=7,
        fusion_dim=16,
        transformer_heads=2,
        imputer_layers=1,
        temporal_layers=1,
    )
    explainer = LocalExplainabilityEngine(ig_steps=5, shap_steps=5)

    patient_data = {
        "mri": torch.randn(1, 3, 1, 6, 12, 12),
        "cognitive": torch.randn(1, 3, 5),
        "ehr": torch.randn(1, 3, 7),
        "visit_mask": torch.ones(1, 3),
        "time_gaps": torch.tensor([[0.0, 6.0, 12.0]]),
    }

    report = explainer.explain_patient(model, patient_data, patient_id="PAT_TEST")
    assert report["patient_id"] == "PAT_TEST"
    assert "risk_probability" in report
    assert "clinical_importance" in report
    assert len(report["clinical_importance"]) > 0
    assert "hippocampus_importance_pct" in report["mri_attribution"]


def test_patient_level_dataset_partitioning():
    """Verify that dataset splits strictly by patient to avoid longitudinal leakage."""
    dataset = ADNIStyleMultimodalDataset(num_patients=20, num_visits=4, mri_shape=(6, 12, 12))
    assert len(dataset) == 20

    item = dataset[0]
    assert item["mri"].shape == (4, 1, 6, 12, 12)
    assert item["cognitive"].shape == (4, 5)
    assert item["ehr"].shape == (4, 7)
    assert item["visit_mask"].shape == (4,)
    assert item["time_gaps"].shape == (4,)

    manager = MultimodalDatasetManager(num_patients=30, num_visits=4, mri_shape=(6, 12, 12))
    train_set, test_set = manager.load_data()
    assert len(train_set) == 24
    assert len(test_set) == 6

    # Verify zero patient overlap
    train_pids = {manager.dataset[i]["patient_id"] for i in train_set.indices}
    test_pids = {manager.dataset[i]["patient_id"] for i in test_set.indices}
    assert len(train_pids.intersection(test_pids)) == 0


def test_clinician_dashboard_rendering(tmp_path):
    """Verify ASCII, JSON, and HTML dashboard rendering."""
    sample_report = {
        "patient_id": "PAT_0042",
        "risk_probability": 0.784,
        "risk_category": "HIGH",
        "clinical_importance": [
            {"feature": "MMSE", "relative_pct": 34.5},
            {"feature": "CDR-SB", "relative_pct": 28.2},
            {"feature": "Age", "relative_pct": 14.1},
        ],
        "mri_attribution": {
            "hippocampus_importance_pct": 21.8,
            "peak_attribution_voxel": [3, 6, 6],
        },
        "longitudinal_visit_weights": [0.15, 0.20, 0.30, 0.35],
    }

    ascii_out = ClinicianDashboard.render_ascii(sample_report)
    assert "FED-XNEURO DASHBOARD" in ascii_out
    assert "78.4%" in ascii_out
    assert "HIGH" in ascii_out
    assert "MMSE" in ascii_out

    json_file = str(tmp_path / "report.json")
    html_file = str(tmp_path / "report.html")
    ClinicianDashboard.save_json_report(sample_report, json_file)
    ClinicianDashboard.save_html_report(sample_report, html_file)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
