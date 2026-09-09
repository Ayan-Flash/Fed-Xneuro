import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from backend.fl_engine.core.client import FederatedClient
from backend.fl_engine.core.trainer import Trainer


def test_client_training_updates_parameters(tiny_model, synthetic_dataset):
    device = torch.device("cpu")
    initial_params = tiny_model.get_parameters()

    client = FederatedClient(
        client_id="test_client_0",
        dataset=synthetic_dataset,
        model=tiny_model,
        device=device,
        local_epochs=2,
        batch_size=8,
        learning_rate=0.05,
    )

    metrics = client.train()

    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert metrics["num_samples"] == len(synthetic_dataset)

    updated_params = client.get_model_parameters()

    # Verify weights have changed from initial
    weight_diff = torch.norm(updated_params["fc.weight"] - initial_params["fc.weight"]).item()
    assert weight_diff > 1e-4, "Client model weights should update after local training"

    # Verify original tiny_model instance was NOT modified (loose coupling)
    orig_diff = torch.norm(tiny_model.fc.weight - initial_params["fc.weight"]).item()
    assert orig_diff == 0.0, "Original model instance must not be modified directly by client"


def test_client_evaluation(tiny_model, synthetic_dataset):
    device = torch.device("cpu")
    client = FederatedClient(
        client_id="test_client_1",
        dataset=synthetic_dataset,
        model=tiny_model,
        device=device,
    )

    eval_res = client.evaluate()
    assert "eval_loss" in eval_res
    assert "eval_accuracy" in eval_res
    assert 0.0 <= eval_res["eval_accuracy"] <= 100.0
    assert eval_res["num_samples"] == len(synthetic_dataset)


def test_client_parameter_get_and_set(tiny_model, synthetic_dataset):
    device = torch.device("cpu")
    client = FederatedClient(
        client_id="test_client_2",
        dataset=synthetic_dataset,
        model=tiny_model,
        device=device,
    )

    custom_params = {
        "fc.weight": torch.full_like(tiny_model.fc.weight, 7.5),
        "fc.bias": torch.full_like(tiny_model.fc.bias, 2.5),
    }
    client.set_model_parameters(custom_params)
    retrieved = client.get_model_parameters()

    assert torch.allclose(retrieved["fc.weight"], custom_params["fc.weight"])
    assert torch.allclose(retrieved["fc.bias"], custom_params["fc.bias"])
