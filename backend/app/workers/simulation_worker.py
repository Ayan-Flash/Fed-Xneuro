"""
Background worker executing simulation jobs asynchronously.
"""

import asyncio
from typing import Dict, Any
from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine
from backend.app.core.logging import logger
from backend.app.websocket.manager import ws_manager


class SimulationWorker:
    """Worker handling background simulation lifecycle."""

    @staticmethod
    def run_simulation_job(config_dict: Dict[str, Any], sim_id: int) -> Dict[str, Any]:
        """
        Executes a simulation job synchronously or inside a background thread.
        """
        logger.info(f"Starting simulation job {sim_id}")
        config = SimulationConfig.from_dict(config_dict)
        engine = SimulationEngine(config)
        state, summary = engine.run()
        logger.info(f"Completed simulation job {sim_id}")
        return summary
