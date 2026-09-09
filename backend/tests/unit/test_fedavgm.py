import pytest
import torch
from backend.fl_engine.algorithms.baselines.fedavgm import FedAvgM
from backend.fl_engine.algorithms.base import ClientUpdate


def test_fedavgm_initialization():
    """Verify FedAvgM initializes with momentum and learning rate."""
    algo = FedAvgM(server_momentum=0.85, server_lr=1.2)
    assert algo.name == "FedAvgM"
    assert algo.server_momentum == 0.85
    assert algo.server_lr == 1.2
    assert algo.momentum_buffer == {}


def test_fedavgm_zero_momentum_fallback():
    """Verify that beta=0 matches standard FedAvg averaging."""
    algo = FedAvgM(server_momentum=0.0)
    p1 = {"w": torch.tensor([2.0, 4.0])}
    p2 = {"w": torch.tensor([6.0, 8.0])}
    global_p = {"w": torch.tensor([0.0, 0.0])}

    updates = [
        ClientUpdate(client_id="c1", parameters=p1, num_samples=1, metrics={}),
        ClientUpdate(client_id="c2", parameters=p2, num_samples=1, metrics={}),
    ]
    res = algo.aggregate(updates, global_parameters=global_p)
    expected = torch.tensor([4.0, 6.0])
    assert torch.allclose(res["w"], expected)


def test_fedavgm_momentum_buffer_update():
    """
    Verify server momentum updates velocity buffer across rounds:
    v_{t+1} = beta * v_t + (w_t - avg_{t+1})
    w_{t+1} = w_t - server_lr * v_{t+1}
    """
    algo = FedAvgM(server_momentum=0.9, server_lr=1.0)
    w0 = {"w": torch.tensor([10.0, 10.0])}

    # Round 1: avg = [8.0, 8.0] -> delta = [2.0, 2.0]
    # v1 = [2.0, 2.0], w1 = 10.0 - 2.0 = [8.0, 8.0]
    u1 = [ClientUpdate(client_id="c1", parameters={"w": torch.tensor([8.0, 8.0])}, num_samples=1, metrics={})]
    w1 = algo.aggregate(u1, global_parameters=w0)
    assert torch.allclose(w1["w"], torch.tensor([8.0, 8.0]))
    assert torch.allclose(algo.momentum_buffer["w"], torch.tensor([2.0, 2.0]))

    # Round 2: avg = [7.0, 7.0] -> delta = w1 - avg = [8.0 - 7.0] = [1.0, 1.0]
    # v2 = 0.9 * [2.0, 2.0] + [1.0, 1.0] = [2.8, 2.8]
    # w2 = 8.0 - 2.8 = [5.2, 5.2]
    u2 = [ClientUpdate(client_id="c1", parameters={"w": torch.tensor([7.0, 7.0])}, num_samples=1, metrics={})]
    w2 = algo.aggregate(u2, global_parameters=w1)
    assert torch.allclose(w2["w"], torch.tensor([5.2, 5.2]))
    assert torch.allclose(algo.momentum_buffer["w"], torch.tensor([2.8, 2.8]))


def test_fedavgm_empty_updates_raises():
    """Verify FedAvgM raises ValueError when update list is empty."""
    algo = FedAvgM()
    with pytest.raises(ValueError):
        algo.aggregate([])
