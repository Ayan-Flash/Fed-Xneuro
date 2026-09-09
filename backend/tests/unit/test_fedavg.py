import pytest
import torch
from backend.fl_engine.core.aggregator import FedAvgAggregator
from backend.fl_engine.algorithms.baselines.fedavg import FedAvg
from backend.fl_engine.algorithms.base import ClientUpdate


def test_fedavg_weighted_averaging_exact():
    """
    Test exact mathematical averaging specified in requirements:
    Client A: weights = 1.0, samples = 10
    Client B: weights = 3.0, samples = 30
    Expected: global = (10*1.0 + 30*3.0) / 40 = 100 / 40 = 2.5
    """
    client_a_weights = {"weight": torch.tensor([1.0, 1.0])}
    client_b_weights = {"weight": torch.tensor([3.0, 3.0])}

    aggregated = FedAvgAggregator.aggregate(
        client_parameters=[client_a_weights, client_b_weights],
        client_sample_counts=[10, 30]
    )

    expected = torch.tensor([2.5, 2.5])
    assert torch.allclose(aggregated["weight"], expected), f"Expected {expected}, got {aggregated['weight']}"


def test_fedavg_algorithm_wrapper():
    """Test FedAvg algorithm class wrapper with ClientUpdate objects."""
    fedavg = FedAvg()

    update_a = ClientUpdate(
        client_id="0",
        parameters={"layer.weight": torch.tensor([2.0])},
        num_samples=20,
        metrics={},
    )
    update_b = ClientUpdate(
        client_id="1",
        parameters={"layer.weight": torch.tensor([6.0])},
        num_samples=20,
        metrics={},
    )

    result = fedavg.aggregate([update_a, update_b])
    expected = torch.tensor([4.0])
    assert torch.allclose(result["layer.weight"], expected)


def test_fedavg_integer_tensors():
    """Test that integer parameters (e.g. tracking counters) are handled safely."""
    client_a = {"counter": torch.tensor(10, dtype=torch.int64)}
    client_b = {"counter": torch.tensor(30, dtype=torch.int64)}

    aggregated = FedAvgAggregator.aggregate(
        client_parameters=[client_a, client_b],
        client_sample_counts=[10, 30]
    )

    assert aggregated["counter"].dtype == torch.int64
    # (10*10 + 30*30) / 40 = 1000 / 40 = 25
    assert aggregated["counter"].item() == 25


def test_fedavg_empty_input_raises():
    """Test that aggregating empty list raises ValueError."""
    with pytest.raises(ValueError):
        FedAvgAggregator.aggregate([], [])


def test_fedavg_mismatched_lengths_raises():
    """Test that mismatched parameters and counts raise ValueError."""
    with pytest.raises(ValueError):
        FedAvgAggregator.aggregate([{"w": torch.tensor([1.0])}], [10, 20])
