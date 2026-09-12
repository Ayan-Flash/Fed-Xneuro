"""
Integration tests for FastAPI REST endpoints using httpx.AsyncClient.
"""

import pytest
import httpx
from backend.app.main import app
from backend.app.db.database import init_db

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


async def test_health_check_endpoints(async_client: httpx.AsyncClient):
    """Verify root and health check endpoints."""
    r_root = await async_client.get("/")
    assert r_root.status_code == 200
    assert "Fed-XNeuro" in r_root.json()["project"]

    r_health = await async_client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


async def test_algorithms_endpoint(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/algorithms returns registry list including fedxneuro."""
    r = await async_client.get("/api/v1/algorithms")
    assert r.status_code == 200
    algos = r.json()
    names = [a["name"] for a in algos]
    assert "fedavg" in names
    assert "fedxneuro" in names


async def test_models_endpoint(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/models returns registered models including fedxneuro."""
    r = await async_client.get("/api/v1/models")
    assert r.status_code == 200
    models = r.json()
    names = [m["name"] for m in models]
    assert "cnn" in names
    assert "fedxneuro" in names


async def test_datasets_endpoint(async_client: httpx.AsyncClient):
    """Verify GET /api/v1/datasets returns registered datasets including multimodal."""
    r = await async_client.get("/api/v1/datasets")
    assert r.status_code == 200
    datasets = r.json()
    names = [d["name"] for d in datasets]
    assert "mnist" in names
    assert "multimodal" in names


async def test_simulation_crud_lifecycle(async_client: httpx.AsyncClient):
    """Verify creation, retrieval, status checking, and deletion of a simulation."""
    sim_payload = {
        "name": "Integration Test Simulation",
        "description": "API verification run",
        "dataset": "multimodal",
        "model": "fedxneuro",
        "algorithm": "fedxneuro",
        "num_clients": 3,
        "num_rounds": 2,
    }
    r_create = await async_client.post("/api/v1/simulations", json=sim_payload)
    assert r_create.status_code == 201
    sim = r_create.json()
    sim_id = sim["id"]
    assert sim["algorithm"] == "fedxneuro"

    r_get = await async_client.get(f"/api/v1/simulations/{sim_id}")
    assert r_get.status_code == 200
    assert r_get.json()["id"] == sim_id

    r_status = await async_client.get(f"/api/v1/simulations/{sim_id}/status")
    assert r_status.status_code == 200
    assert r_status.json()["simulation_id"] == sim_id

    r_dashboard = await async_client.get(f"/api/v1/results/{sim_id}/clinician-dashboard")
    assert r_dashboard.status_code == 200
    assert "FED-XNEURO DASHBOARD" in r_dashboard.json()["ascii"]

    r_del = await async_client.delete(f"/api/v1/simulations/{sim_id}")
    assert r_del.status_code == 204
