import time
from typing import Dict, Any, Optional, Tuple
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
        proximal_mu: float = 0.0,
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
        proximal_reference: Optional[Dict[str, torch.Tensor]] = None,
        mu: float = 0.0,
        control_variates: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes local training loop for the given number of epochs.
        Supports FedProx proximal loss regularization when proximal_mu > 0.

        Returns:
            Dict containing training metrics (loss, accuracy, duration, samples_trained, total_batches).
        """
        eff_mu = mu if mu > 0.0 else proximal_mu
        eff_ref = proximal_reference if proximal_reference is not None else global_parameters
        model.to(self.device)
        model.train()

        # Handle empty dataset edge case
        if len(dataset) == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "duration": 0.0,
                "samples_trained": 0,
                "total_batches": 0,
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
            momentum=momentum if control_variates is None else 0.0,  # SCAFFOLD standard uses SGD without momentum
            weight_decay=weight_decay
        )

        start_time = time.time()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_batches = 0

        # Unpack control variates if provided for SCAFFOLD
        c_client, c_server = (None, None)
        if control_variates is not None:
            c_client, c_server = control_variates

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = model(data)
                loss = self.criterion(output, target)

                # FedProx proximal regularization: (mu / 2) * sum(||w - w_global||^2)
                if eff_mu > 0.0 and eff_ref is not None:
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, param in model.named_parameters():
                        if name in eff_ref:
                            g_param = eff_ref[name].to(param.device)
                            proximal_term = proximal_term + (param - g_param).norm(2) ** 2
                    loss = loss + (eff_mu / 2.0) * proximal_term

                loss.backward()

                # SCAFFOLD: Adjust parameter gradients with (c_server - c_client)
                if c_client is not None and c_server is not None:
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in c_client and name in c_server:
                            cs = c_server[name].to(self.device)
                            ci = c_client[name].to(self.device)
                            param.grad.data.add_(cs - ci)

                optimizer.step()

                batch_size_actual = data.size(0)
                epoch_loss += loss.item() * batch_size_actual
                preds = output.argmax(dim=1, keepdim=True)
                epoch_correct += preds.eq(target.view_as(preds)).sum().item()
                epoch_samples += batch_size_actual
                total_batches += 1

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
            "total_batches": total_batches,
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
