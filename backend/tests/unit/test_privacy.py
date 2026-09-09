import math
import pytest
import torch
import torch.nn as nn
from backend.fl_engine.algorithms.base import ClientUpdate
from backend.fl_engine.privacy.dp_mechanism import (
    DifferentialPrivacyMechanism,
    compute_param_l2_norm,
    clip_parameter_update,
    add_gaussian_noise,
)
from backend.fl_engine.privacy.rdp_accountant import RDPAccountant
from backend.fl_engine.privacy.inversion_defense import InversionAttackSimulator


def test_compute_param_l2_norm():
    """Verify correct L2 norm calculation across parameter tensors."""
    params = {
        "w1": torch.tensor([3.0, 4.0]),  # norm = 5.0
        "w2": torch.tensor([0.0, 0.0]),
    }
    norm = compute_param_l2_norm(params)
    assert math.isclose(norm, 5.0, rel_tol=1e-5)


def test_clip_parameter_update_within_threshold():
    """Verify updates with norm <= C are preserved unchanged."""
    params = {"w": torch.tensor([1.0, 1.0])}  # norm = sqrt(2) ~ 1.414
    clipped, orig_norm = clip_parameter_update(params, max_norm=2.0)
    assert math.isclose(orig_norm, math.sqrt(2.0), rel_tol=1e-5)
    assert torch.allclose(clipped["w"], params["w"])


def test_clip_parameter_update_exceeds_threshold():
    """Verify updates with norm > C are scaled down to exactly norm C."""
    params = {"w": torch.tensor([6.0, 8.0])}  # norm = 10.0
    clipped, orig_norm = clip_parameter_update(params, max_norm=5.0)
    assert math.isclose(orig_norm, 10.0, rel_tol=1e-5)

    clipped_norm = compute_param_l2_norm(clipped)
    assert math.isclose(clipped_norm, 5.0, rel_tol=1e-5)
    assert torch.allclose(clipped["w"], torch.tensor([3.0, 4.0]))


def test_add_gaussian_noise():
    """Verify Gaussian noise perturbation preserves tensor shape and changes values."""
    torch.manual_seed(42)
    params = {"w": torch.zeros(100, 100)}
    noised = add_gaussian_noise(params, noise_std=0.5)

    assert noised["w"].shape == params["w"].shape
    assert not torch.allclose(noised["w"], params["w"])
    # Sample mean should be close to 0
    assert abs(noised["w"].mean().item()) < 0.1


def test_rdp_accountant_monotonicity():
    """Verify epsilon increases monotonically with rounds and decreases with higher noise."""
    acc_low_noise = RDPAccountant(target_delta=1e-5)
    acc_high_noise = RDPAccountant(target_delta=1e-5)

    # 5 rounds with q=0.5
    for _ in range(5):
        acc_low_noise.step(sampling_ratio=0.5, noise_multiplier=0.5)
        acc_high_noise.step(sampling_ratio=0.5, noise_multiplier=1.5)

    eps_low = acc_low_noise.get_epsilon()
    eps_high = acc_high_noise.get_epsilon()

    assert eps_low > eps_high, "Lower noise must consume higher epsilon privacy budget"
    assert eps_low > 0.0
    assert eps_high > 0.0


def test_differential_privacy_mechanism_orchestration():
    """Verify end-to-end client update clipping and aggregated perturbation in DP mechanism."""
    dp = DifferentialPrivacyMechanism(clip_norm=1.0, noise_multiplier=0.2, target_delta=1e-5)
    global_p = {"fc.weight": torch.zeros(2, 2)}

    # Client update with large norm: delta = [[2, 2], [2, 2]] (norm = 4.0)
    client_p = {"fc.weight": torch.full((2, 2), 2.0)}
    updates = [ClientUpdate(client_id="c1", parameters=client_p, num_samples=10, metrics={})]

    clipped_updates, orig_norms = dp.clip_client_updates(updates, global_p)
    assert orig_norms[0] == 4.0
    clipped_p = clipped_updates[0].parameters["fc.weight"]
    # Clipped delta norm must equal 1.0
    assert math.isclose(clipped_p.norm().item(), 1.0, rel_tol=1e-4)

    # Perturb aggregated parameters
    agg_p = {"fc.weight": torch.ones(2, 2)}
    noised_agg = dp.perturb_aggregated_parameters(agg_p, num_participating_clients=1)
    assert noised_agg["fc.weight"].shape == agg_p["fc.weight"].shape


def test_inversion_defense_simulator():
    """Verify InversionAttackSimulator runs and computes reconstruction MSE."""
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    sample_x = torch.randn(1, 8)
    sample_y = torch.tensor([1])

    res = InversionAttackSimulator.simulate_attack(
        model=model,
        true_input=sample_x,
        true_label=sample_y,
        enable_dp=True,
        noise_std=0.2,
        iterations=10,
    )

    assert "initial_mse" in res
    assert "final_mse" in res
    assert "defense_effective" in res
    assert res["enable_dp"] is True
