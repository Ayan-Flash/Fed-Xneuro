"""
Integration test verifying SimulationEngine with multimodal Fed-XNeuro.
"""

import os
import pytest
import torch

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine


def test_simulation_engine_runs_fedxneuro():
    """Verify that SimulationEngine seamlessly executes multimodal Fed-XNeuro."""
    round_callbacks_received = []

    def on_round(r_num, r_data):
        round_callbacks_received.append((r_num, r_data.get("global_loss")))

    config = SimulationConfig(
        dataset="multimodal",
        model="fedxneuro",
        algorithm="fedxneuro",
        num_clients=2,
        num_rounds=2,
        local_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=101,
        device="cpu",
    )

    engine = SimulationEngine(config, on_round_complete=on_round)
    state, summary = engine.run()

    assert state.status == "completed"
    assert summary["total_rounds"] == 2
    assert "final_accuracy" in summary
    assert "final_loss" in summary
    assert len(round_callbacks_received) == 2
    assert round_callbacks_received[0][0] == 1
    assert round_callbacks_received[1][0] == 2
    assert "clinician_report" in summary
