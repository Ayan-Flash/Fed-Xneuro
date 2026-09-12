"""
Integration tests for SimulationService and database models.
"""

import pytest
from backend.app.db.session import SessionLocal
from backend.app.db.database import init_db
from backend.app.db.models.metric import Metric
from backend.app.schemas.simulation import SimulationCreate
from backend.app.services.simulation_service import SimulationService
from backend.app.services.metrics_service import MetricsService


@pytest.fixture(scope="module", autouse=True)
def init_database():
    init_db()


def test_simulation_service_and_metrics_persistence():
    """Verify simulation lifecycle, round metrics insertion, and status transition in DB."""
    db = SessionLocal()
    try:
        sim_in = SimulationCreate(
            name="Service Test Sim",
            description="Testing DB persistence",
            dataset="mnist",
            model="cnn",
            algorithm="fedavg",
            num_clients=2,
            num_rounds=3,
        )
        sim = SimulationService.create_simulation(db, sim_in)
        assert sim.id is not None
        assert sim.status == "created"
        # Ensure fresh state for this sim
        db.query(Metric).filter(Metric.simulation_id == sim.id).delete()
        db.commit()

        # Update status to running
        sim = SimulationService.update_simulation_status(db, sim.id, status="running", current_round=1)
        assert sim.status == "running"
        assert sim.current_round == 1

        # Add round metrics
        MetricsService.add_round_metric(
            db, simulation_id=sim.id, round_num=1, loss=0.52, accuracy=84.5, communication_bytes=1024.0
        )
        MetricsService.add_round_metric(
            db, simulation_id=sim.id, round_num=2, loss=0.38, accuracy=89.2, communication_bytes=2048.0
        )

        metrics = MetricsService.get_metrics_for_simulation(db, sim.id)
        assert len(metrics) == 2
        assert metrics[1].accuracy == 89.2

        # Mark completed
        sim = SimulationService.update_simulation_status(
            db, sim.id, status="completed", current_round=3, final_loss=0.35, final_accuracy=90.1
        )
        assert sim.status == "completed"
        assert sim.final_accuracy == 90.1

        # Clean up
        SimulationService.delete_simulation(db, sim.id)
    finally:
        db.close()
