from typing import Dict, List, Any
import numpy as np


class FairnessEvaluator:
    """
    Evaluates performance distribution across clients to measure fairness.
    Computes mean, min, max, variance, standard deviation, and accuracy gap.
    """

    @staticmethod
    def evaluate(client_accuracies: Dict[str, float] | List[float]) -> Dict[str, float]:
        """
        Computes client fairness statistics.

        Args:
            client_accuracies: Dict or List of client accuracy values (%).

        Returns:
            Dict containing mean, min, max, variance, std_dev, and accuracy gap.
        """
        if isinstance(client_accuracies, dict):
            values = list(client_accuracies.values())
        else:
            values = list(client_accuracies)

        if not values:
            return {
                "mean_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "accuracy_variance": 0.0,
                "accuracy_std": 0.0,
                "accuracy_gap": 0.0,
            }

        arr = np.array(values, dtype=np.float64)
        mean_acc = float(np.mean(arr))
        min_acc = float(np.min(arr))
        max_acc = float(np.max(arr))
        var_acc = float(np.var(arr))
        std_acc = float(np.std(arr))
        gap_acc = float(max_acc - min_acc)

        return {
            "mean_accuracy": mean_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "accuracy_variance": var_acc,
            "accuracy_std": std_acc,
            "accuracy_gap": gap_acc,
        }
