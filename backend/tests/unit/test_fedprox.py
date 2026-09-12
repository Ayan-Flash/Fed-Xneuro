import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from backend.fl_engine.algorithms.baselines.fedprox import FedProx
from backend.fl_engine.core.trainer import Trainer


def test_fedprox_initialization():
    """Verify FedProx baseline initializes with specified proximal hyperparameter mu."""
    algo = FedProx(mu=0.05)
    assert algo.name == "FedProx"
    assert algo.mu == 0.05


def test_fedprox_proximal_term_regularization():
    """Verify FedProx proximal term keeps local parameters closer to global model."""
    torch.manual_seed(42)
    # Simple linear model
    x = torch.randn(20, 5)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)

    global_model = nn.Linear(5, 2)
    global_params = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    # Model 1: Trained without proximal term (mu = 0)
    model_standard = nn.Linear(5, 2)
    model_standard.load_state_dict(global_params)
    trainer = Trainer()
    trainer.train(
        model=model_standard,
        dataset=dataset,
        local_epochs=5,
        batch_size=4,
        learning_rate=0.1,
        proximal_mu=0.0,
        global_parameters=global_params,
    )

    # Model 2: Trained with strong proximal term (mu = 10.0)
    model_prox = nn.Linear(5, 2)
    model_prox.load_state_dict(global_params)
    trainer.train(
        model=model_prox,
        dataset=dataset,
        local_epochs=5,
        batch_size=4,
        learning_rate=0.1,
        proximal_mu=10.0,
        global_parameters=global_params,
    )

    # Compute L2 distance from global weights
    dist_standard = sum((p - global_params[n]).norm(2).item() ** 2 for n, p in model_standard.named_parameters())
    dist_prox = sum((p - global_params[n]).norm(2).item() ** 2 for n, p in model_prox.named_parameters())

    # Regularized model must drift significantly less from global model
    assert dist_prox < dist_standard
