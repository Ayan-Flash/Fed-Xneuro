import pytest
from backend.fl_engine.algorithms.baselines.fedprox import FedProx


def test_fedprox_initialization():
    """Verify FedProx baseline initializes with specified proximal hyperparameter mu."""
    algo = FedProx(mu=0.05)
    assert algo.name == "FedProx"
    assert algo.mu == 0.05


@pytest.mark.skip(reason="Full local proximal loss regularization loop scheduled for Phase 2")
def test_fedprox_proximal_term_regularization():
    """Placeholder for Phase 2 FedProx loss regularization tests."""
    pass
