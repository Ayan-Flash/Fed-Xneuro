import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Trainer:
    """
    Reusable local trainer responsible for PyTorch model training and local evaluation.
    Decoupled from client orchestration to enable algorithmic customizations (e.g. FedProx).
    """

    def __init__(
        self,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cpu")

    def train(
        self,
        model: nn.Module,
        dataset: Dataset,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Executes local training loop for the given number of epochs.

        Returns:
            Dict containing training metrics (loss, accuracy, duration, samples_trained).
        """
        model.to(self.device)
        model.train()

        # Handle empty dataset edge case
        if len(dataset) == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "duration": 0.0,
                "samples_trained": 0,
            }

        data_loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            drop_last=False
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_size_actual = data.size(0)
                epoch_loss += loss.item() * batch_size_actual
                preds = output.argmax(dim=1, keepdim=True)
                epoch_correct += preds.eq(target.view_as(preds)).sum().item()
                epoch_samples += batch_size_actual

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        duration = time.time() - start_time
        avg_loss = total_loss / max(1, total_samples)
        accuracy = (correct_predictions / max(1, total_samples)) * 100.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "duration": float(duration),
            "samples_trained": len(dataset),
        }

    def evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """Evaluates model performance on local dataset."""
        model.to(self.device)
        model.eval()

        if len(dataset) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "num_samples": 0}

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1, keepdim=True)
                correct += preds.eq(target.view_as(preds)).sum().item()
                total += data.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = (correct / max(1, total)) * 100.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "num_samples": total
        }
