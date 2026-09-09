import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from backend.fl_engine.algorithms.baselines.scaffold import SCAFFOLD
from backend.fl_engine.algorithms.base import ClientUpdate
from backend.fl_engine.core.client import FederatedClient


def test_scaffold_initialization():
    """Verify SCAFFOLD initializes with server learning rate and empty control variate dict."""
    algo = SCAFFOLD(server_lr=1.5)
    assert algo.name == "SCAFFOLD"
    assert algo.server_lr == 1.5
    assert algo.server_control == {}


def test_scaffold_control_variate_aggregation():
    """Verify SCAFFOLD updates server control variate c = c + (1 / |S|) * sum(delta_c_i)."""
    algo = SCAFFOLD()
    p1 = {"fc.weight": torch.tensor([1.0, 2.0])}
    p2 = {"fc.weight": torch.tensor([3.0, 4.0])}
    dc1 = {"fc.weight": torch.tensor([0.2, 0.4])}
    dc2 = {"fc.weight": torch.tensor([0.4, 0.8])}

    updates = [
        ClientUpdate(client_id="c1", parameters=p1, num_samples=10, metrics={}, control_variate_delta=dc1),
        ClientUpdate(client_id="c2", parameters=p2, num_samples=10, metrics={}, control_variate_delta=dc2),
    ]

    algo.aggregate(updates)
    # Average delta: (0.2 + 0.4)/2 = 0.3, (0.4 + 0.8)/2 = 0.6
    expected_c = torch.tensor([0.3, 0.6])
    assert torch.allclose(algo.server_control["fc.weight"], expected_c)


def test_scaffold_client_drift_tracking():
    """Verify FederatedClient tracks control variate deltas during local training under SCAFFOLD."""
    model = nn.Linear(4, 2, bias=False)
    inputs = torch.randn(20, 4)
    targets = torch.randint(0, 2, (20,))
    dataset = TensorDataset(inputs, targets)

    client = FederatedClient(
        client_id="client_0",
        dataset=dataset,
        model=model,
        device=torch.device("cpu"),
        local_epochs=1,
        batch_size=10,
        learning_rate=0.01,
    )

    server_control = {"weight": torch.zeros_like(model.weight)}
    metrics = client.train(server_control=server_control)
    delta_c = client.get_control_variate_delta()

    assert "weight" in delta_c
    assert delta_c["weight"].shape == model.weight.shape
    assert metrics["num_samples"] == 20


def test_scaffold_empty_updates_raises():
    """Verify SCAFFOLD raises ValueError when update list is empty."""
    algo = SCAFFOLD()
    with pytest.raises(ValueError):
        algo.aggregate([])
