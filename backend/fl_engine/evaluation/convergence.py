from typing import List, Dict, Any, Optional


class ConvergenceTracker:
    """
    Tracks convergence behavior over federated rounds (loss reduction,
    accuracy progression, target milestones).
    """

    def __init__(self, target_accuracies: Optional[List[float]] = None) -> None:
        self.target_accuracies = target_accuracies or [50.0, 70.0, 80.0, 90.0, 95.0]
        self.rounds_to_target: Dict[float, int] = {}
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def update(self, round_num: int, loss: float, accuracy: float) -> Dict[str, Any]:
        """
        Updates the tracker with current round metrics and checks for milestone hits.
        """
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        # Check milestones
        for target in self.target_accuracies:
            if target not in self.rounds_to_target and accuracy >= target:
                self.rounds_to_target[target] = round_num

        loss_delta = 0.0
        if len(self.loss_history) > 1:
            loss_delta = self.loss_history[-1] - self.loss_history[-2]

        acc_delta = 0.0
        if len(self.accuracy_history) > 1:
            acc_delta = self.accuracy_history[-1] - self.accuracy_history[-2]

        return {
            "round": round_num,
            "loss_delta": float(loss_delta),
            "accuracy_delta": float(acc_delta),
            "milestones_reached": {f"{k}%": v for k, v in self.rounds_to_target.items()},
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns overall convergence summary."""
        best_accuracy = max(self.accuracy_history) if self.accuracy_history else 0.0
        min_loss = min(self.loss_history) if self.loss_history else 0.0

        return {
            "best_accuracy": float(best_accuracy),
            "min_loss": float(min_loss),
            "final_accuracy": float(self.accuracy_history[-1]) if self.accuracy_history else 0.0,
            "final_loss": float(self.loss_history[-1]) if self.loss_history else 0.0,
            "milestones": {f"{k}%": v for k, v in self.rounds_to_target.items()},
            "total_rounds": len(self.loss_history),
        }
