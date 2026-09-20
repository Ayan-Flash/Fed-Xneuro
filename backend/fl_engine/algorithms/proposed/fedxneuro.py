"""
Fed-XNeuro Federated Learning Algorithm & Execution Architecture.

Implements Modules 7, 10, 11, 12, 13, 14 from Fed-XNeuro specification:
- FedXNeuroTrainer: Multimodal longitudinal local trainer with BCEWithLogitsLoss
- FedXNeuroClient: Medical center node with private patient cohort, local DP guard, and on-device XAI
- FedXNeuro: Core federated algorithm orchestrator with FedAvg aggregation and privacy budget accounting
"""

import copy
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.fl_engine.algorithms.base import BaseFederatedAlgorithm, ClientUpdate
from backend.fl_engine.core.aggregator import FedAvgAggregator
from backend.fl_engine.algorithms.proposed.privacy import DifferentialPrivacyGuard
from backend.fl_engine.algorithms.proposed.explainability import LocalExplainabilityEngine


class FedXNeuroTrainer:
    """
    Multimodal longitudinal PyTorch trainer for Fed-XNeuro.
    Handles longitudinal batches of (MRI, Cognitive, EHR, Visit Mask, Time Gaps, Label).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def train(
        self,
        model: nn.Module,
        dataset: Dataset,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Executes local training on multimodal patient dataset.
        """
        model.to(self.device)
        model.train()

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
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            for batch in data_loader:
                mri = batch["mri"].to(self.device)
                cog = batch["cognitive"].to(self.device)
                ehr = batch["ehr"].to(self.device)
                mask = batch["visit_mask"].to(self.device)
                time_gaps = batch["time_gaps"].to(self.device)
                labels = batch["label"].to(self.device)  # [B, 1]

                optimizer.zero_grad()
                out = model(
                    mri=mri,
                    cognitive=cog,
                    ehr=ehr,
                    visit_mask=mask,
                    time_gaps=time_gaps,
                )
                logits = torch.clamp(out["logits"], -20.0, 20.0)
                loss = self.criterion(logits, labels)
                loss.backward()
                # Clip local gradients before optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                b_size = labels.size(0)
                total_loss += loss.item() * b_size

                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += preds.eq(labels).sum().item()
                total_samples += b_size

        duration = time.time() - start_time
        avg_loss = total_loss / max(1, total_samples)
        accuracy = (total_correct / max(1, total_samples)) * 100.0

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
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Evaluates multimodal model performance."""
        model.to(self.device)
        model.eval()

        if len(dataset) == 0:
            return {"loss": 0.0, "accuracy": 0.0, "num_samples": 0}

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs: List[float] = []
        all_targets: List[float] = []

        with torch.no_grad():
            for batch in data_loader:
                mri = batch["mri"].to(self.device)
                cog = batch["cognitive"].to(self.device)
                ehr = batch["ehr"].to(self.device)
                mask = batch["visit_mask"].to(self.device)
                time_gaps = batch["time_gaps"].to(self.device)
                labels = batch["label"].to(self.device)

                out = model(
                    mri=mri,
                    cognitive=cog,
                    ehr=ehr,
                    visit_mask=mask,
                    time_gaps=time_gaps,
                )
                logits = torch.clamp(out["logits"], -20.0, 20.0)
                loss = self.criterion(logits, labels)

                b_size = labels.size(0)
                total_loss += loss.item() * b_size

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total_correct += preds.eq(labels).sum().item()
                total_samples += b_size

                all_probs.extend(probs.view(-1).cpu().tolist())
                all_targets.extend(labels.view(-1).cpu().tolist())

        avg_loss = total_loss / max(1, total_samples)
        accuracy = (total_correct / max(1, total_samples)) * 100.0

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "num_samples": total_samples,
            "predictions": all_probs,
            "targets": all_targets,
        }


class FedXNeuroClient:
    """
    Simulated Hospital Node in the Fed-XNeuro network.
    Holds a private longitudinal patient cohort, trains locally,
    applies client-side Differential Privacy (L2 clipping + Gaussian noise),
    and generates on-device clinical explainability reports.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        model: nn.Module,
        device: torch.device,
        local_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        use_dp: bool = True,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
    ) -> None:
        self.client_id = str(client_id)
        self.dataset = dataset
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_dp = use_dp

        self.model = copy.deepcopy(model).to(self.device)
        self.trainer = FedXNeuroTrainer(device=self.device)
        self.dp_guard = DifferentialPrivacyGuard(
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
        )
        self.explainer = LocalExplainabilityEngine(ig_steps=10, shap_steps=10)

        self.last_global_parameters: Optional[Dict[str, torch.Tensor]] = None
        self.last_metrics: Dict[str, Any] = {}

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Receives global model parameters from the server."""
        self.last_global_parameters = {k: v.detach().cpu().clone() for k, v in parameters.items()}
        dev_params = {k: v.to(self.device) for k, v in parameters.items()}
        self.model.load_state_dict(dev_params, strict=True)

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes private local training on hospital cohort.
        """
        train_metrics = self.trainer.train(
            model=self.model,
            dataset=self.dataset,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        self.last_metrics = {
            "client_id": self.client_id,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "duration": train_metrics["duration"],
            "num_samples": train_metrics["samples_trained"],
        }
        return self.last_metrics

    def evaluate(self) -> Dict[str, Any]:
        """Evaluates model on local hospital cohort."""
        eval_metrics = self.trainer.evaluate(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size,
        )
        return {
            "client_id": self.client_id,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "num_samples": eval_metrics["num_samples"],
        }

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns model parameters for server aggregation.
        If Differential Privacy is enabled, clips parameter deltas and injects Gaussian noise.
        """
        raw_params = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if not self.use_dp or self.last_global_parameters is None:
            return raw_params

        # Apply Differential Privacy Guard (Module 6 & 10)
        privatized_params, dp_stats = self.dp_guard.privatize_update(
            local_parameters=raw_params,
            global_parameters=self.last_global_parameters,
        )
        self.last_metrics.update(dp_stats)
        return privatized_params

    def explain_patient(self, index: int = 0) -> Dict[str, Any]:
        """
        Generates local explainability report for a patient in this hospital.
        STRICT PRIVACY: Executed locally, never transmitted to server.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Patient index {index} out of range (0-{len(self.dataset)-1}).")

        sample = self.dataset[index]
        patient_data = {
            "mri": sample["mri"].unsqueeze(0).to(self.device),
            "cognitive": sample["cognitive"].unsqueeze(0).to(self.device),
            "ehr": sample["ehr"].unsqueeze(0).to(self.device),
            "visit_mask": sample["visit_mask"].unsqueeze(0).to(self.device),
            "time_gaps": sample["time_gaps"].unsqueeze(0).to(self.device),
        }
        return self.explainer.explain_patient(
            model=self.model,
            patient_data=patient_data,
            patient_id=sample.get("patient_id", f"PAT_{self.client_id}_{index}"),
        )

    @property
    def num_samples(self) -> int:
        return len(self.dataset)


class FedXNeuro(BaseFederatedAlgorithm):
    """
    Fed-XNeuro Federated Optimization Algorithm.
    Aggregates privacy-protected client updates using sample-weighted Federated Averaging,
    tracks differential privacy budget across rounds, and logs multimodal performance.
    """

    def __init__(
        self,
        use_dp: bool = True,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
    ) -> None:
        super().__init__(name="Fed-XNeuro")
        self.aggregator = FedAvgAggregator()
        self.use_dp = use_dp
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta

        self.current_round = 0
        self.privacy_accountant = DifferentialPrivacyGuard(
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
        )

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        global_parameters: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates privacy-protected updates from hospital clients using sample-weighted FedAvg.
        """
        client_parameters = [update.parameters for update in client_updates]
        client_sample_counts = [update.num_samples for update in client_updates]

        return self.aggregator.aggregate(client_parameters, client_sample_counts)

    def on_round_start(self, round_num: int) -> None:
        self.current_round = round_num

    def on_round_end(self, round_num: int, round_metrics: Dict[str, Any]) -> None:
        if self.use_dp:
            epsilon, delta = self.privacy_accountant.compute_privacy_spent(num_rounds=round_num)
            round_metrics["privacy_spent_epsilon"] = epsilon
            round_metrics["privacy_spent_delta"] = delta
