import time
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backend.fl_engine.core.client import FederatedClient
from backend.fl_engine.core.client_selector import RandomClientSelector
from backend.fl_engine.core.aggregator import FedAvgAggregator
from backend.fl_engine.core.communication import CommunicationTracker
from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.algorithms.baselines.fedavg import FedAvg
from backend.fl_engine.evaluation.accuracy import calculate_accuracy
from backend.fl_engine.evaluation.loss import calculate_loss


class FederatedServer:
    """
    Federated Learning Orchestration Server.
    Manages the global model, client coordination, model distribution,
    aggregation delegation, and global evaluation.
    """

    def __init__(
        self,
        global_model: nn.Module,
        algorithm: Optional[BaseFederatedAlgorithm] = None,
        client_selector: Optional[RandomClientSelector] = None,
        test_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        dp_mechanism: Optional[Any] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.global_model = global_model.to(self.device)
        self.algorithm = algorithm or FedAvg()
        self.client_selector = client_selector or RandomClientSelector(client_fraction=1.0)
        self.test_dataset = test_dataset
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.dp_mechanism = dp_mechanism

        self.clients: Dict[str, FederatedClient] = {}
        self.communication_tracker = CommunicationTracker(
            model_size_bytes=CommunicationTracker.calculate_model_size_bytes(self.global_model)
        )

    def register_client(self, client: FederatedClient) -> None:
        """Registers a client instance with the server."""
        self.clients[client.client_id] = client

    def register_clients(self, client_list: List[FederatedClient]) -> None:
        """Registers multiple client instances with the server."""
        for client in client_list:
            self.register_client(client)

    def select_clients(self) -> List[str]:
        """Selects a subset of client IDs for the current round."""
        available_ids = list(self.clients.keys())
        return self.client_selector.select_clients(available_ids)

    def send_global_model(self, selected_client_ids: List[str]) -> None:
        """Distributes current global model parameters to selected clients."""
        global_parameters = {
            k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()
        }
        for cid in selected_client_ids:
            self.clients[cid].set_model_parameters(global_parameters)

    def receive_client_updates(self, selected_client_ids: List[str]) -> List[ClientUpdate]:
        """
        Commands selected clients to perform local training and collects updates.
        Passes proximal reference or control variates if configured by the algorithm.
        """
        # Determine if algorithm requires FedProx proximal regularization
        mu = getattr(self.algorithm, "mu", 0.0)
        proximal_ref = None
        if mu > 0.0:
            proximal_ref = {
                k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()
            }

        # Determine if algorithm requires SCAFFOLD server control variate
        server_control = getattr(self.algorithm, "server_control", None)

        updates: List[ClientUpdate] = []
        for cid in selected_client_ids:
            client = self.clients[cid]
            train_metrics = client.train(
                proximal_reference=proximal_ref,
                mu=mu,
                server_control=server_control,
            )
            params = client.get_model_parameters()
            cv_delta = client.get_control_variate_delta() if server_control is not None else None

            updates.append(
                ClientUpdate(
                    client_id=cid,
                    parameters=params,
                    num_samples=client.num_samples,
                    metrics=train_metrics,
                    control_variate_delta=cv_delta,
                )
            )
        return updates

    def aggregate_updates(self, client_updates: List[ClientUpdate]) -> None:
        """
        Delegates aggregation of client updates to the configured algorithm
        and loads the resulting parameters into the global model.
        """
        current_global_params = {
            k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()
        }
        aggregated_params = self.algorithm.aggregate(
            client_updates=client_updates,
            global_parameters=current_global_params,
        )
        # Move parameters to server device and load into global model
        dev_params = {k: v.to(self.device) for k, v in aggregated_params.items()}
        self.global_model.load_state_dict(dev_params, strict=True)

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluates global model on the global test dataset."""
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        loss = calculate_loss(
            model=self.global_model,
            dataset_or_loader=self.test_dataset,
            criterion=self.criterion,
            device=self.device,
        )
        accuracy = calculate_accuracy(
            model=self.global_model,
            dataset_or_loader=self.test_dataset,
            device=self.device,
        )

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Executes a single federated training round:
        1. Select clients
        2. Distribute global model
        3. Train clients locally
        4. Aggregate updates into global model
        5. Evaluate updated global model
        6. Record round metrics and communication
        """
        round_start = time.time()
        self.algorithm.on_round_start(round_num)

        # 1. Selection
        selected_client_ids = self.select_clients()

        # 2. Distribution
        self.send_global_model(selected_client_ids)

        # 3. Local Training
        train_start = time.time()
        client_updates = self.receive_client_updates(selected_client_ids)
        train_duration = time.time() - train_start

        # 3b. Differential Privacy: Clip client model updates
        if self.dp_mechanism is not None:
            current_global_params = {
                k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()
            }
            client_updates, _ = self.dp_mechanism.clip_client_updates(
                client_updates, current_global_params
            )

        # 4. Aggregation
        agg_start = time.time()
        self.aggregate_updates(client_updates)

        # 4b. Differential Privacy: Add calibrated Gaussian noise to aggregated model
        if self.dp_mechanism is not None:
            noised_params = self.dp_mechanism.perturb_aggregated_parameters(
                self.global_model.state_dict(),
                num_participating_clients=len(selected_client_ids)
            )
            dev_params = {k: v.to(self.device) for k, v in noised_params.items()}
            self.global_model.load_state_dict(dev_params, strict=True)

        agg_duration = time.time() - agg_start

        # 5. Global Evaluation
        eval_start = time.time()
        eval_metrics = self.evaluate_global_model()
        eval_duration = time.time() - eval_start

        # 6. Communication tracking
        comm_stats = self.communication_tracker.track_round(
            num_selected_clients=len(selected_client_ids)
        )

        round_duration = time.time() - round_start

        round_result = {
            "round": round_num,
            "selected_clients": len(selected_client_ids),
            "client_ids": selected_client_ids,
            "global_loss": eval_metrics["loss"],
            "global_accuracy": eval_metrics["accuracy"],
            "training_time": round(train_duration, 4),
            "aggregation_time": round(agg_duration, 4),
            "evaluation_time": round(eval_duration, 4),
            "round_duration": round(round_duration, 4),
            "communication_bytes": comm_stats["round_total_bytes"],
            "cumulative_communication_bytes": comm_stats["cumulative_total_bytes"],
            "client_metrics": {u.client_id: u.metrics for u in client_updates},
        }

        self.algorithm.on_round_end(round_num, round_result)
        return round_result
