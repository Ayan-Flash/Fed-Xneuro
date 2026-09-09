from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import time


@dataclass
class SimulationState:
    """
    Tracks the lifecycle and runtime progress of a simulation run.
    Designed to easily interface with future FastAPI endpoints and WebSockets.
    """

    simulation_id: str
    status: str = "created"  # 'created', 'running', 'completed', 'failed', 'stopped'
    current_round: int = 0
    total_rounds: int = 10
    selected_clients: List[str] = field(default_factory=list)
    completed_clients: List[str] = field(default_factory=list)
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None

    def start(self, total_rounds: int) -> None:
        self.status = "running"
        self.total_rounds = total_rounds
        self.current_round = 0
        self.start_time = time.time()

    def update_round(self, round_num: int, loss: float, accuracy: float, selected_clients: List[str]) -> None:
        self.current_round = round_num
        self.global_loss = loss
        self.global_accuracy = accuracy
        self.selected_clients = selected_clients

    def complete(self) -> None:
        self.status = "completed"
        self.end_time = time.time()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.error_message = error
        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["elapsed_time"] = self.elapsed_time
        return d
