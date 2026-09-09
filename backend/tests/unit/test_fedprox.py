import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from backend.fl_engine.algorithms.baselines.fedprox import FedProx
from backend.fl_engine.algorithms.base import ClientUpdate
from backend.fl_engine.core.trainer import Trainer


def test_fedprox_initialization():
    """Verify FedProx baseline initializes with specified proximal hyperparameter mu."""
    algo = FedProx(mu=0.05)
    assert algo.name == "FedProx"
    assert algo.mu == 0.05


def test_fedprox_proximal_term_regularization():
    """
    Verify that FedProx proximal penalty (mu / 2) * ||w - w_ref||^2 pulls weights
    closer to the reference parameters compared to unregularized training (mu=0).
    """
    torch.manual_seed(42)
    inputs = torch.randn(40, 4)
    targets = torch.randint(0, 2, (40,))
    dataset = TensorDataset(inputs, targets)

    # Initialize two identical linear models
    model_unreg = nn.Linear(4, 2, bias=False)
    model_prox = copy.deepcopy(model_unreg)

    ref_weights = {k: v.detach().clone() for k, v in model_unreg.state_dict().items()}

    trainer = Trainer(device=torch.device("cpu"))

    # Train unregularized (mu = 0.0)
    trainer.train(
        model=model_unreg,
        dataset=dataset,
        local_epochs=5,
        batch_size=8,
        learning_rate=0.05,
        mu=0.0,
    )

    # Train with strong proximal penalty (mu = 2.0)
    trainer.train(
        model=model_prox,
        dataset=dataset,
        local_epochs=5,
        batch_size=8,
        learning_rate=0.05,
        proximal_reference=ref_weights,
        mu=2.0,
    )

    # Calculate L2 distance from original reference
    dist_unreg = (model_unreg.weight - ref_weights["weight"]).norm().item()
    dist_prox = (model_prox.weight - ref_weights["weight"]).norm().item()

    # The proximal term must penalize drift, ensuring dist_prox < dist_unreg
    assert dist_prox < dist_unreg, f"Expected dist_prox ({dist_prox}) < dist_unreg ({dist_unreg})"


def test_fedprox_aggregation():
    """Verify FedProx aggregates client updates using weighted FedAvg."""
    algo = FedProx(mu=0.01)
    p1 = {"w": torch.tensor([1.0, 2.0])}
    p2 = {"w": torch.tensor([3.0, 4.0])}

    updates = [
        ClientUpdate(client_id="c1", parameters=p1, num_samples=100, metrics={}),
        ClientUpdate(client_id="c2", parameters=p2, num_samples=300, metrics={}),
    ]

    aggregated = algo.aggregate(updates)
    # Expected weighted average: 0.25 * [1, 2] + 0.75 * [3, 4] = [2.5, 3.5]
    expected = torch.tensor([2.5, 3.5])
    assert torch.allclose(aggregated["w"], expected)


def test_fedprox_empty_updates_raises():
    """Verify FedProx raises ValueError when update list is empty."""
    algo = FedProx()
    with pytest.raises(ValueError):
        algo.aggregate([])
