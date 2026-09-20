import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsManager:
    """
    Manages collection, persistence, and visualization of experiment metrics.
    Stores round-level metrics, exports to JSON and CSV, and renders plots.
    """

    def __init__(self, run_id: str, results_dir: str = "results", experiments_dir: str = "experiments") -> None:
        self.run_id = run_id
        self.results_dir = results_dir
        self.experiments_dir = experiments_dir
        self.round_metrics: List[Dict[str, Any]] = []
        self.summary_metrics: Dict[str, Any] = {}

    def log_round(self, round_data: Dict[str, Any]) -> None:
        """Records metrics for a completed federated round."""
        self.round_metrics.append(round_data)

    def get_round_metrics(self) -> List[Dict[str, Any]]:
        """Returns the full list of round metrics."""
        return self.round_metrics

    def set_summary(self, summary: Dict[str, Any]) -> None:
        """Sets summary statistics for the completed simulation."""
        self.summary_metrics = summary

    def save_results(self) -> Dict[str, str]:
        """
        Saves metrics to JSON and CSV in both experiments/runs/<run_id>/
        and results/metrics/<run_id>/.

        Returns:
            Dict mapping file descriptions to saved paths.
        """
        run_exp_dir = os.path.join(self.experiments_dir, "runs", self.run_id)
        res_metrics_dir = os.path.join(self.results_dir, "metrics", self.run_id)
        os.makedirs(run_exp_dir, exist_ok=True)
        os.makedirs(res_metrics_dir, exist_ok=True)

        saved_files: Dict[str, str] = {}

        # 1. Save metrics.json
        exp_json_path = os.path.join(run_exp_dir, "metrics.json")
        res_json_path = os.path.join(res_metrics_dir, "metrics.json")
        for path in [exp_json_path, res_json_path]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.round_metrics, f, indent=2)
        saved_files["metrics_json"] = res_json_path

        # 2. Save metrics.csv
        if self.round_metrics:
            df = pd.DataFrame(self.round_metrics)
            # Flatten or format complex columns if present
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

            exp_csv_path = os.path.join(run_exp_dir, "metrics.csv")
            res_csv_path = os.path.join(res_metrics_dir, "metrics.csv")
            df.to_csv(exp_csv_path, index=False)
            df.to_csv(res_csv_path, index=False)
            saved_files["metrics_csv"] = res_csv_path

        # 3. Save summary.json
        if self.summary_metrics:
            exp_summary_path = os.path.join(run_exp_dir, "summary.json")
            res_summary_path = os.path.join(res_metrics_dir, "summary.json")
            for path in [exp_summary_path, res_summary_path]:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.summary_metrics, f, indent=2)
            saved_files["summary_json"] = res_summary_path

        # 4. Generate plots
        plot_path = self.plot_metrics()
        if plot_path:
            saved_files["plot_curves"] = plot_path

        return saved_files

    def plot_metrics(self) -> Optional[str]:
        """
        Generates and saves accuracy and loss progression plots.
        """
        if not self.round_metrics:
            return None

        rounds = [m.get("round", i + 1) for i, m in enumerate(self.round_metrics)]
        accuracies = [m.get("global_accuracy", 0.0) for m in self.round_metrics]
        losses = [m.get("global_loss", 0.0) for m in self.round_metrics]

        plots_dir = os.path.join(self.results_dir, "plots", self.run_id)
        os.makedirs(plots_dir, exist_ok=True)
        plot_file = os.path.join(plots_dir, "training_curves.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Accuracy plot
        ax1.plot(rounds, accuracies, marker="o", color="#2563EB", linewidth=2)
        ax1.set_title("Global Accuracy vs. Rounds", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Federated Round", fontsize=10)
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Loss plot
        ax2.plot(rounds, losses, marker="s", color="#DC2626", linewidth=2)
        ax2.set_title("Global Loss vs. Rounds", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Federated Round", fontsize=10)
        ax2.set_ylabel("Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_file, dpi=200)
        plt.close(fig)

        return plot_file
