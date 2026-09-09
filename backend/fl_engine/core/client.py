from typing import Dict, Any, Optional
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from backend.fl_engine.core.trainer import Trainer


class FederatedClient:
    """
    Simulated Federated Learning Client.
    Holds a local copy of the model, trains strictly on its local partition,
    and returns model parameters without directly modifying the global model.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        model: nn.Module,
        device: torch.device,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        trainer: Optional[Trainer] = None,
    ) -> None:
        self.client_id = str(client_id)
        self.dataset = dataset
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create an independent local copy of the model
        self.model = copy.deepcopy(model).to(self.device)
        self.trainer = trainer or Trainer(device=self.device)
        self.control_variate: Dict[str, torch.Tensor] = {}
        self.last_control_variate_delta: Dict[str, torch.Tensor] = {}
        self.last_metrics: Dict[str, Any] = {}

    def train(
        self,
        proximal_reference: Optional[Dict[str, torch.Tensor]] = None,
        mu: float = 0.0,
        server_control: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Trains the local model on the client's local dataset.
        Supports FedProx proximal regularization and SCAFFOLD control variate correction.
        
        Returns:
            Dict containing training metrics.
        """
        initial_params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # If mu > 0 and no explicit reference given, the initial parameters serve as reference
        if mu > 0.0 and proximal_reference is None:
            proximal_reference = initial_params

        # Prepare SCAFFOLD control variates
        cv_tuple = None
        if server_control is not None:
            if not self.control_variate:
                self.control_variate = {k: torch.zeros_like(v) for k, v in initial_params.items()}
            cv_tuple = (self.control_variate, server_control)

        metrics = self.trainer.train(
            model=self.model,
            dataset=self.dataset,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            proximal_reference=proximal_reference,
            mu=mu,
            control_variates=cv_tuple,
        )

        # Update SCAFFOLD control variate and delta
        if server_control is not None:
            total_batches = max(1, metrics.get("total_batches", 1))
            step_scale = 1.0 / (total_batches * self.learning_rate)
            current_params = self.model.state_dict()
            delta_c = {}
            for k in initial_params.keys():
                ci = self.control_variate[k].to(self.device)
                cs = server_control.get(k, torch.zeros_like(ci)).to(self.device)
                x = initial_params[k].to(self.device)
                y = current_params[k].to(self.device)
                # c_new = ci - cs + (1 / (K * eta)) * (x - y)
                c_new = ci - cs + step_scale * (x - y)
                delta = c_new - ci
                delta_c[k] = delta.detach().cpu().clone()
                self.control_variate[k] = c_new.detach().clone()
            self.last_control_variate_delta = delta_c
        else:
            self.last_control_variate_delta = {}

        self.last_metrics = {
            "client_id": self.client_id,
            "train_loss": metrics["loss"],
            "train_accuracy": metrics["accuracy"],
            "duration": metrics["duration"],
            "num_samples": metrics["samples_trained"],
        }
        return self.last_metrics

    def get_control_variate_delta(self) -> Dict[str, torch.Tensor]:
        """Returns the control variate delta from the most recent training round."""
        return self.last_control_variate_delta

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the local model on the client's local dataset.

        Returns:
            Dict containing local evaluation metrics.
        """
        eval_metrics = self.trainer.evaluate(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size
        )
        return {
            "client_id": self.client_id,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "num_samples": eval_metrics["num_samples"],
        }

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns a detached, CPU copy of the local model's state_dict.
        """
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Loads parameters into the local model.
        """
        # Ensure parameters are placed on the client's device
        device_parameters = {k: v.to(self.device) for k, v in parameters.items()}
        self.model.load_state_dict(device_parameters, strict=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the most recent metrics recorded by this client."""
        return self.last_metrics

    @property
    def num_samples(self) -> int:
        """Returns the number of training samples assigned to this client."""
        return len(self.dataset)
