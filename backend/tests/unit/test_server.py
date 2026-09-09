import torch
from torch.utils.data import TensorDataset
from backend.fl_engine.core.server import FederatedServer
from backend.fl_engine.core.client import FederatedClient
from backend.fl_engine.core.client_selector import RandomClientSelector
from backend.fl_engine.algorithms.baselines.fedavg import FedAvg


def test_server_client_selection(tiny_model, synthetic_dataset):
    device = torch.device("cpu")
    server = FederatedServer(
        global_model=tiny_model,
        client_selector=RandomClientSelector(client_fraction=0.5, seed=42),
    )

    for i in range(10):
        client = FederatedClient(
            client_id=str(i),
            dataset=synthetic_dataset,
            model=tiny_model,
            device=device,
        )
        server.register_client(client)

    selected = server.select_clients()
    # 10 clients * 0.5 = 5 clients selected
    assert len(selected) == 5
    assert len(set(selected)) == 5


def test_server_round_execution(tiny_model, synthetic_dataset):
    device = torch.device("cpu")
    server = FederatedServer(
        global_model=tiny_model,
        algorithm=FedAvg(),
        test_dataset=synthetic_dataset,
        device=device,
    )

    for i in range(3):
        client = FederatedClient(
            client_id=str(i),
            dataset=synthetic_dataset,
            model=tiny_model,
            device=device,
            local_epochs=1,
            batch_size=10,
            learning_rate=0.01,
        )
        server.register_client(client)

    initial_weight = tiny_model.fc.weight.clone()
    round_result = server.run_round(round_num=1)

    assert round_result["round"] == 1
    assert round_result["selected_clients"] == 3
    assert "global_loss" in round_result
    assert "global_accuracy" in round_result
    assert round_result["communication_bytes"] > 0

    # Global model weights should have updated
    updated_weight = server.global_model.fc.weight
    diff = torch.norm(updated_weight - initial_weight).item()
    assert diff > 1e-4, "Global model parameters should update after aggregation"
