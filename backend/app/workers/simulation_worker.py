"""
Background worker executing simulation jobs asynchronously.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine
from backend.app.core.logging import logger
from backend.app.websocket.manager import ws_manager
from backend.app.db.session import SessionLocal
from backend.app.db.models.simulation import Simulation
from backend.app.db.models.metric import Metric
from backend.app.db.models.result import Result


class SimulationWorker:
    """Worker handling background simulation lifecycle."""

    @staticmethod
    def _send_ws(sim_id: str, message: Dict[str, Any]) -> None:
        """Helper to send WebSocket messages safely from synchronous context."""
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                asyncio.create_task(ws_manager.broadcast(sim_id, message))
            else:
                loop.run_until_complete(ws_manager.broadcast(sim_id, message))
        except Exception as e:
            logger.warning(f"Failed to broadcast WebSocket message for sim {sim_id}: {e}")

    @classmethod
    def start_simulation_task(cls, simulation_id: int) -> Dict[str, Any]:
        """
        Executes a simulation job asynchronously, updating the database
        and broadcasting round updates to WebSocket subscribers.
        """
        db = SessionLocal()
        try:
            sim = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            if not sim:
                logger.error(f"Simulation {simulation_id} not found")
                return {}

            sim.status = "running"
            db.commit()
            db.refresh(sim)

            cls._send_ws(
                str(sim.id),
                {"event": "simulation_started", "simulation_id": sim.id, "run_id": sim.run_id},
            )

            # Build SimulationConfig from Simulation DB model
            config = SimulationConfig(
                dataset=sim.dataset,
                model=sim.model,
                algorithm=sim.algorithm,
                num_clients=sim.num_clients,
                client_fraction=sim.client_fraction,
                num_rounds=sim.num_rounds,
                local_epochs=sim.local_epochs,
                batch_size=sim.batch_size,
                learning_rate=sim.learning_rate,
                partition_type=sim.partition_type,
                partition_alpha=sim.partition_alpha,
                seed=sim.seed,
                device=sim.device,
            )

            def on_round_complete(round_num: int, round_data: Dict[str, Any]) -> None:
                round_db = SessionLocal()
                try:
                    s = round_db.query(Simulation).filter(Simulation.id == simulation_id).first()
                    if s:
                        s.current_round = round_num
                        round_db.commit()

                    metric = Metric(
                        simulation_id=simulation_id,
                        round_num=round_num,
                        loss=float(round_data.get("global_loss", 0.0)),
                        accuracy=float(round_data.get("global_accuracy", 0.0)),
                        communication_bytes=float(round_data.get("cumulative_communication_bytes", 0.0)),
                        client_metrics_json=json.dumps(round_data.get("client_metrics", {})),
                    )
                    round_db.add(metric)
                    round_db.commit()

                    cls._send_ws(
                        str(simulation_id),
                        {
                            "event": "round_complete",
                            "simulation_id": simulation_id,
                            "round": round_num,
                            "loss": round_data.get("global_loss", 0.0),
                            "accuracy": round_data.get("global_accuracy", 0.0),
                            "communication_bytes": round_data.get("cumulative_communication_bytes", 0.0),
                            "privacy_budget_spent": round_data.get("privacy_budget_spent"),
                            "clinical_metrics": round_data.get("clinical_metrics"),
                        },
                    )
                except Exception as ex:
                    logger.error(f"Error persisting round {round_num} for sim {simulation_id}: {ex}")
                finally:
                    round_db.close()

            engine = SimulationEngine(config, on_round_complete=on_round_complete)
            state, summary = engine.run()

            # Record completion in DB
            sim.status = "completed"
            sim.current_round = sim.num_rounds
            sim.final_loss = summary.get("final_loss")
            sim.final_accuracy = summary.get("final_accuracy")
            sim.results_json = json.dumps(summary)
            db.commit()

            # Create or update Result record
            result = db.query(Result).filter(Result.simulation_id == simulation_id).first()
            if not result:
                result = Result(
                    simulation_id=simulation_id,
                    final_loss=summary.get("final_loss"),
                    final_accuracy=summary.get("final_accuracy"),
                    summary_json=json.dumps(summary),
                )
                db.add(result)
            else:
                result.final_loss = summary.get("final_loss")
                result.final_accuracy = summary.get("final_accuracy")
                result.summary_json = json.dumps(summary)
            db.commit()

            cls._send_ws(
                str(simulation_id),
                {
                    "event": "simulation_completed",
                    "simulation_id": simulation_id,
                    "final_accuracy": summary.get("final_accuracy"),
                    "final_loss": summary.get("final_loss"),
                    "summary": summary,
                },
            )
            return summary

        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            sim = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            if sim:
                sim.status = "failed"
                db.commit()

            cls._send_ws(
                str(simulation_id),
                {"event": "simulation_failed", "simulation_id": simulation_id, "error": str(e)},
            )
            return {"error": str(e)}
        finally:
            db.close()

    @classmethod
    def run_simulation_job(cls, config_dict: Dict[str, Any], sim_id: int) -> Dict[str, Any]:
        """Legacy compatibility wrapper."""
        return cls.start_simulation_task(sim_id)
