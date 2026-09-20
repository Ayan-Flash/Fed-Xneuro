"""
Integration tests for FastAPI simulation execution endpoint and dashboard UI.
"""

import pytest
import httpx
from backend.app.main import app
from backend.app.db.database import init_db
from backend.app.db.session import SessionLocal
from backend.app.db.models.simulation import Simulation
from backend.app.db.models.metric import Metric
from backend.app.db.models.result import Result
from backend.app.workers.simulation_worker import SimulationWorker

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


async def test_dashboard_route_serves_html(async_client: httpx.AsyncClient):
    """Verify /dashboard serves the clinician & researcher portal HTML."""
    res = await async_client.get("/dashboard")
    assert res.status_code == 200
    assert "FED-XNEURO" in res.text
    assert "Clinician Diagnostic View" in res.text


async def test_simulation_start_api_and_worker_execution(async_client: httpx.AsyncClient):
    """Verify simulation creation, start endpoint, worker execution, DB metrics, and results."""
    # 1. Create Simulation for API start endpoint test
    payload = {
        "name": "API Start Endpoint Test",
        "dataset": "mnist",
        "model": "cnn",
        "algorithm": "fedavg",
        "num_clients": 2,
        "num_rounds": 1,
    }
    r_create = await async_client.post("/api/v1/simulations", json=payload)
    assert r_create.status_code == 201
    sim_id = r_create.json()["id"]

    # 2. Call Start Endpoint
    r_start = await async_client.post(f"/api/v1/simulations/{sim_id}/start")
    assert r_start.status_code == 200
    assert r_start.json()["status"] == "starting"

    # 3. Create fresh simulation and directly execute worker to verify DB persistence
    payload_worker = {
        "name": "Worker Execution Test",
        "dataset": "mnist",
        "model": "cnn",
        "algorithm": "fedavg",
        "num_clients": 2,
        "num_rounds": 2,
        "local_epochs": 1,
        "batch_size": 16,
    }
    r_worker = await async_client.post("/api/v1/simulations", json=payload_worker)
    worker_sim_id = r_worker.json()["id"]

    summary = SimulationWorker.start_simulation_task(worker_sim_id)
    assert "final_accuracy" in summary

    # 4. Verify DB State for worker_sim_id
    db = SessionLocal()
    try:
        sim = db.query(Simulation).filter(Simulation.id == worker_sim_id).first()
        assert sim.status == "completed"
        assert sim.current_round == 2
        assert sim.final_accuracy is not None

        metrics = db.query(Metric).filter(Metric.simulation_id == worker_sim_id).all()
        assert len(metrics) == 2

        result = db.query(Result).filter(Result.simulation_id == worker_sim_id).first()
        assert result is not None
        assert result.final_accuracy is not None
    finally:
        db.close()
