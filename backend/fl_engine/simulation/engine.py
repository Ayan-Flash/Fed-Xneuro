import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple
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
from backend.fl_engine.utils.seed import set_seed
from backend.fl_engine.evaluation.convergence import ConvergenceTracker
from backend.fl_engine.evaluation.fairness import FairnessEvaluator


class SimulationEngine:
    """
    Primary Orchestrator for the Federated Learning Simulation.
    Initializes datasets, models, clients, and server; drives round iterations;
    collects metrics; and saves reproducible experiment artifacts.
    """

    def __init__(self, config: SimulationConfig) -> None:
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
            clients = []
            for cid in range(self.config.num_clients):
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
            algorithm = create_algorithm(self.config.algorithm, **self.config.algorithm_kwargs)
            client_selector = RandomClientSelector(
                client_fraction=self.config.client_fraction,
                seed=self.config.seed,
            )
            server = FederatedServer(
                global_model=global_model,
                algorithm=algorithm,
                client_selector=client_selector,
                test_dataset=test_dataset,
                device=self.device,
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

            self.metrics_manager.set_summary(summary)
            saved_paths = self.metrics_manager.save_results()
            self.state.complete()

            self._print_completion(summary, saved_paths)
            return self.state, summary

        except Exception as e:
            self.state.fail(str(e))
            raise e

    def _print_header(self) -> None:
        device_str = "CUDA" if self.device.type == "cuda" else "CPU"
        print("=" * 50, flush=True)
        print("PS32 Federated Learning Simulator", flush=True)
        print("=" * 50, flush=True)
        print(f"Dataset       : {self.config.dataset.upper()}", flush=True)
        print(f"Model         : {self.config.model.upper()}", flush=True)
        print(f"Algorithm     : {self.config.algorithm.upper()}", flush=True)
        print(f"Clients       : {self.config.num_clients}", flush=True)
        print(f"Rounds        : {self.config.num_rounds}", flush=True)
        print(f"Local Epochs  : {self.config.local_epochs}", flush=True)
        print(f"Partition     : {self.config.partition_type.upper()}", flush=True)
        print(f"Device        : {device_str}", flush=True)
        print(f"Seed          : {self.config.seed}", flush=True)
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
