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
        self.last_metrics: Dict[str, Any] = {}

    def train(self) -> Dict[str, Any]:
        """
        Trains the local model on the client's local dataset.
        
        Returns:
            Dict containing training metrics.
        """
        metrics = self.trainer.train(
            model=self.model,
            dataset=self.dataset,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        self.last_metrics = {
            "client_id": self.client_id,
            "train_loss": metrics["loss"],
            "train_accuracy": metrics["accuracy"],
            "duration": metrics["duration"],
            "num_samples": metrics["samples_trained"],
        }
        return self.last_metrics

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
