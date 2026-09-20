import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Callable, List
import torch

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.state import SimulationState
from backend.fl_engine.core.client import FederatedClient
from backend.fl_engine.core.server import FederatedServer
from backend.fl_engine.core.client_selector import RandomClientSelector
from backend.fl_engine.core.dataset_manager import DatasetManager
from backend.fl_engine.core.model_manager import ModelManager
from backend.fl_engine.core.metrics_manager import MetricsManager
from backend.fl_engine.algorithms import create_algorithm
from backend.fl_engine.algorithms.proposed.fedxneuro import FedXNeuroClient, FedXNeuroTrainer
from backend.fl_engine.algorithms.proposed.utils import compute_clinical_metrics
from backend.fl_engine.evaluation.dashboard import ClinicianDashboard
from backend.fl_engine.privacy.dp_mechanism import DifferentialPrivacyMechanism
from backend.fl_engine.privacy.rdp_accountant import RDPAccountant
from backend.fl_engine.utils.seed import set_seed
from backend.fl_engine.evaluation.convergence import ConvergenceTracker
from backend.fl_engine.evaluation.fairness import FairnessEvaluator


class SimulationEngine:
    """
    Primary Orchestrator for the Federated Learning Simulation.
    Initializes datasets, models, clients, and server; drives round iterations;
    collects metrics; and saves reproducible experiment artifacts.
    """

    def __init__(
        self,
        config: SimulationConfig,
        on_round_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.state = SimulationState(simulation_id=self.run_id)
        self.metrics_manager = MetricsManager(
            run_id=self.run_id,
            results_dir=config.results_dir,
            experiments_dir=config.experiments_dir,
        )
        self.convergence_tracker = ConvergenceTracker()
        self.on_round_complete = on_round_complete

    def run(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """
        Executes the federated learning simulation.
        
        Returns:
            Tuple of (SimulationState, summary_dict).
        """
        self._print_header()
        set_seed(self.config.seed)
        self.state.start(total_rounds=self.config.num_rounds)

        try:
            # 1. Dataset loading and partitioning
            data_mgr = DatasetManager(
                dataset_name=self.config.dataset,
                num_clients=self.config.num_clients,
                partition_type=self.config.partition_type,
                data_dir=self.config.data_dir,
                seed=self.config.seed,
                alpha=self.config.partition_alpha,
            )
            client_datasets, test_dataset = data_mgr.setup()
            data_stats = data_mgr.get_statistics()

            # 2. Global Model Initialization
            if self.config.custom_model_path and self.config.custom_model_class:
                print(f"Connecting external model architecture: {self.config.custom_model_class} from {self.config.custom_model_path}", flush=True)
                ModelManager.register_external_architecture(
                    file_path=self.config.custom_model_path,
                    class_name=self.config.custom_model_class,
                    register_as=self.config.model,
                )

            global_model = ModelManager.create_model(self.config.model)

            # Load external checkpoint if provided
            if self.config.checkpoint_path:
                print(f"Loading external model weights from: {self.config.checkpoint_path}", flush=True)
                ModelManager.load_model(global_model, self.config.checkpoint_path, device=self.device)

            # 3. Client Creation
            is_fedxneuro = (
                self.config.model.lower() in ["fedxneuro", "fed_xneuro"]
                or self.config.algorithm.lower() in ["fedxneuro", "fed_xneuro", "fedxneuro_personalized", "fed_xneuro_personalized", "fedxneuro_p"]
                or self.config.dataset.lower() in ["multimodal", "adni_mci"]
            )

            clients = []
            for cid in range(self.config.num_clients):
                if is_fedxneuro:
                    client = FedXNeuroClient(
                        client_id=f"Hospital_{chr(65 + cid)}",
                        dataset=client_datasets[cid],
                        model=global_model,
                        device=self.device,
                        local_epochs=self.config.local_epochs,
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                        use_dp=self.config.enable_dp,
                        clip_norm=self.config.dp_clip_norm,
                        noise_multiplier=self.config.dp_noise_multiplier,
                        target_delta=self.config.dp_target_delta,
                    )
                else:
                    client = FederatedClient(
                        client_id=str(cid),
                        dataset=client_datasets[cid],
                        model=global_model,
                        device=self.device,
                        local_epochs=self.config.local_epochs,
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                    )
                clients.append(client)

            # 4. Algorithm & Server Initialization
            algo_kwargs = dict(self.config.algorithm_kwargs)
            algo_key = self.config.algorithm.lower()
            if algo_key == "fedprox" and "mu" not in algo_kwargs:
                algo_kwargs["mu"] = self.config.mu
            elif algo_key == "fedavgm":
                if "server_momentum" not in algo_kwargs:
                    algo_kwargs["server_momentum"] = self.config.server_momentum
                if "server_lr" not in algo_kwargs:
                    algo_kwargs["server_lr"] = self.config.server_lr
            elif algo_key == "scaffold" and "server_lr" not in algo_kwargs:
                algo_kwargs["server_lr"] = self.config.server_lr
            elif algo_key in ["fedxneuro", "fed_xneuro", "fedxneuro_personalized"]:
                algo_kwargs["use_dp"] = self.config.enable_dp
                algo_kwargs["clip_norm"] = self.config.dp_clip_norm
                algo_kwargs["noise_multiplier"] = self.config.dp_noise_multiplier
                algo_kwargs["target_delta"] = self.config.dp_target_delta

            algorithm = create_algorithm(self.config.algorithm, **algo_kwargs)
            client_selector = RandomClientSelector(
                client_fraction=self.config.client_fraction,
                seed=self.config.seed,
            )

            # Initialize Differential Privacy if enabled
            dp_mechanism = None
            rdp_accountant = None
            if self.config.enable_dp and not is_fedxneuro:
                # FedXNeuro clients handle DP internally via their local DP Guard
                dp_mechanism = DifferentialPrivacyMechanism(
                    clip_norm=self.config.dp_clip_norm,
                    noise_multiplier=self.config.dp_noise_multiplier,
                    target_delta=self.config.dp_target_delta,
                )
                rdp_accountant = RDPAccountant(target_delta=self.config.dp_target_delta)

            evaluator = FedXNeuroTrainer(device=self.device) if is_fedxneuro else None

            server = FederatedServer(
                global_model=global_model,
                algorithm=algorithm,
                client_selector=client_selector,
                test_dataset=test_dataset,
                device=self.device,
                dp_mechanism=dp_mechanism,
                evaluator=evaluator,
            )
            server.register_clients(clients)

            # Model checkpoint directory
            checkpoint_dir = os.path.join(self.config.models_dir, "global", self.run_id)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save initial config
            run_exp_dir = os.path.join(self.config.experiments_dir, "runs", self.run_id)
            os.makedirs(run_exp_dir, exist_ok=True)
            with open(os.path.join(run_exp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            # 5. Round Loop
            for round_num in range(1, self.config.num_rounds + 1):
                round_result = server.run_round(round_num)

                # Record differential privacy budget if enabled
                if self.config.enable_dp and rdp_accountant is not None:
                    sampling_ratio = round_result["selected_clients"] / float(max(1, self.config.num_clients))
                    rdp_accountant.step(
                        sampling_ratio=sampling_ratio,
                        noise_multiplier=self.config.dp_noise_multiplier
                    )
                    eps_spent = rdp_accountant.get_epsilon()
                    round_result["privacy_budget_spent"] = round(eps_spent, 4)
                    round_result["privacy_delta"] = self.config.dp_target_delta

                # Compute clinical metrics for multimodal Fed-XNeuro
                if is_fedxneuro and "predictions" in round_result and "targets" in round_result:
                    clin_metrics = compute_clinical_metrics(
                        round_result["predictions"], round_result["targets"]
                    )
                    round_result["clinical_metrics"] = clin_metrics
                    for k, v in clin_metrics.items():
                        round_result[f"clinical_{k}"] = v

                # Also retrieve DP budget if FedXNeuro
                if is_fedxneuro and hasattr(algorithm, "get_privacy_spent"):
                    eps_spent = algorithm.get_privacy_spent()
                    round_result["privacy_budget_spent"] = round(eps_spent, 4)
                    round_result["privacy_delta"] = self.config.dp_target_delta

                self.metrics_manager.log_round(round_result)
                self.state.update_round(
                    round_num=round_num,
                    loss=round_result["global_loss"],
                    accuracy=round_result["global_accuracy"],
                    selected_clients=round_result["client_ids"],
                )
                self.convergence_tracker.update(
                    round_num=round_num,
                    loss=round_result["global_loss"],
                    accuracy=round_result["global_accuracy"],
                )

                self._print_round(round_result)

                # Invoke on_round_complete callback if registered
                if self.on_round_complete is not None:
                    try:
                        self.on_round_complete(round_num, round_result)
                    except Exception as cb_err:
                        print(f"Warning in on_round_complete callback: {cb_err}", flush=True)

                # Save round checkpoint
                if self.config.save_checkpoints:
                    round_ckpt_path = os.path.join(checkpoint_dir, f"round_{round_num:03d}.pt")
                    ModelManager.save_model(server.global_model, round_ckpt_path)

            # 6. Save final checkpoint
            final_ckpt_path = os.path.join(checkpoint_dir, "final.pt")
            ModelManager.save_model(server.global_model, final_ckpt_path)

            # 7. Post-simulation client evaluations for fairness metrics
            client_eval_accuracies = {}
            for client in clients:
                eval_res = client.evaluate()
                client_eval_accuracies[client.client_id] = eval_res["eval_accuracy"]
            fairness_metrics = FairnessEvaluator.evaluate(client_eval_accuracies)

            # 8. Final Summary & Results Export
            convergence_summary = self.convergence_tracker.get_summary()
            summary = {
                "run_id": self.run_id,
                "status": "completed",
                "config": self.config.to_dict(),
                "dataset_statistics": {
                    "total_train_samples": data_stats["total_train_samples"],
                    "total_test_samples": data_stats["total_test_samples"],
                    "samples_per_client": data_stats["samples_per_client"],
                },
                "final_accuracy": round_result["global_accuracy"],
                "final_loss": round_result["global_loss"],
                "best_accuracy": convergence_summary["best_accuracy"],
                "min_loss": convergence_summary["min_loss"],
                "total_rounds": self.config.num_rounds,
                "total_duration_sec": self.state.elapsed_time,
                "total_communication_bytes": round_result["cumulative_communication_bytes"],
                "fairness_metrics": fairness_metrics,
                "checkpoints_path": checkpoint_dir,
            }

            if self.config.enable_dp and rdp_accountant is not None:
                summary["privacy_metrics"] = {
                    "enable_dp": True,
                    "clip_norm": self.config.dp_clip_norm,
                    "noise_multiplier": self.config.dp_noise_multiplier,
                    "target_delta": self.config.dp_target_delta,
                    "final_epsilon": round(rdp_accountant.get_epsilon(), 4),
                }

            self.metrics_manager.set_summary(summary)
            saved_paths = self.metrics_manager.save_results()

            # 9. Explainability Report for Clinicians if Fed-XNeuro
            if is_fedxneuro and clients and hasattr(clients[0], "explain_patient"):
                try:
                    report = clients[0].explain_patient(index=0)
                    json_path = os.path.join(self.config.results_dir, "clinician_report_sample.json")
                    html_path = os.path.join(self.config.results_dir, "clinician_dashboard.html")
                    ClinicianDashboard.save_json_report(report, json_path)
                    ClinicianDashboard.save_html_report(report, html_path)
                    summary["clinician_report"] = report
                    saved_paths["clinician_report_json"] = json_path
                    saved_paths["clinician_dashboard_html"] = html_path
                except Exception as exp_err:
                    print(f"Notice: Clinician report generation skipped: {exp_err}", flush=True)

            self.state.complete()

            self._print_completion(summary, saved_paths)
            return self.state, summary

        except Exception as e:
            self.state.fail(str(e))
            raise e

    def _print_header(self) -> None:
        device_str = "CUDA" if self.device.type == "cuda" else "CPU"
        algo_info = self.config.algorithm.upper()
        if self.config.algorithm.lower() == "fedprox":
            algo_info += f" (mu={self.config.mu})"
        elif self.config.algorithm.lower() == "fedavgm":
            algo_info += f" (momentum={self.config.server_momentum}, lr={self.config.server_lr})"
        elif self.config.algorithm.lower() == "scaffold":
            algo_info += f" (server_lr={self.config.server_lr})"

        print("=" * 50, flush=True)
        print("PS32 Federated Learning Simulator", flush=True)
        print("=" * 50, flush=True)
        print(f"Dataset       : {self.config.dataset.upper()}", flush=True)
        print(f"Model         : {self.config.model.upper()}", flush=True)
        print(f"Algorithm     : {algo_info}", flush=True)
        print(f"Clients       : {self.config.num_clients}", flush=True)
        print(f"Rounds        : {self.config.num_rounds}", flush=True)
        print(f"Local Epochs  : {self.config.local_epochs}", flush=True)
        print(f"Partition     : {self.config.partition_type.upper()}", flush=True)
        print(f"Device        : {device_str}", flush=True)
        print(f"Seed          : {self.config.seed}", flush=True)
        if self.config.enable_dp:
            print(f"Privacy (DP)  : CLIP={self.config.dp_clip_norm} | SIGMA={self.config.dp_noise_multiplier} | DELTA={self.config.dp_target_delta}", flush=True)
        print(flush=True)

    def _print_round(self, r: Dict[str, Any]) -> None:
        print("-" * 50, flush=True)
        print(f"Round {r['round']}/{self.config.num_rounds}", flush=True)
        print("-" * 50, flush=True)
        print(f"Selected Clients : {r['selected_clients']}/{self.config.num_clients}", flush=True)
        print("Training         : completed", flush=True)
        print(f"Aggregation      : {self.config.algorithm.capitalize()}", flush=True)
        print(f"Global Loss      : {r['global_loss']:.4f}", flush=True)
        print(f"Global Accuracy  : {r['global_accuracy']:.2f}%", flush=True)
        print(f"Training Time    : {r['training_time']:.2f}s", flush=True)
        if "privacy_budget_spent" in r:
            print(f"Privacy Spent    : eps = {r['privacy_budget_spent']:.4f} (delta = {r.get('privacy_delta', 1e-5)})", flush=True)
        print(flush=True)

    def _print_completion(self, summary: Dict[str, Any], saved_paths: Dict[str, str]) -> None:
        print("=" * 50, flush=True)
        print("Simulation Complete", flush=True)
        print("=" * 50, flush=True)
        print(f"Final Accuracy : {summary['final_accuracy']:.2f}%", flush=True)
        print(f"Final Loss     : {summary['final_loss']:.4f}", flush=True)
        print(f"Best Accuracy  : {summary['best_accuracy']:.2f}%", flush=True)
        print(f"Total Time     : {summary['total_duration_sec']:.2f}s", flush=True)
        print(flush=True)
        print("Results saved to:", flush=True)
        for k, v in saved_paths.items():
            print(f"  {k}: {v}", flush=True)
        print(f"  checkpoints: {summary['checkpoints_path']}", flush=True)
        print(flush=True)
