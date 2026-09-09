from dataclasses import dataclass, asdict
from typing import Dict, List, Any


@dataclass
class RoundResult:
    """Encapsulates the state and metrics of a single federated training round."""

    round_number: int
    selected_clients: List[str]
    global_loss: float
    global_accuracy: float
    training_time: float
    aggregation_time: float
    evaluation_time: float
    round_duration: float
    communication_bytes: int
    cumulative_communication_bytes: int
    client_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes round result to dictionary."""
        return asdict(self)
