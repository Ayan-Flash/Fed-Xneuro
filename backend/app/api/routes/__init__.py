"""
API Routes package.
"""

from fastapi import APIRouter
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.users import router as users_router
from backend.app.api.routes.simulations import router as simulations_router
from backend.app.api.routes.algorithms import router as algorithms_router
from backend.app.api.routes.models import router as models_router
from backend.app.api.routes.datasets import router as datasets_router
from backend.app.api.routes.metrics import router as metrics_router
from backend.app.api.routes.results import router as results_router
from backend.app.api.routes.clients import router as clients_router
from backend.app.api.routes.experiments import router as experiments_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(simulations_router)
api_router.include_router(algorithms_router)
api_router.include_router(models_router)
api_router.include_router(datasets_router)
api_router.include_router(metrics_router)
api_router.include_router(results_router)
api_router.include_router(clients_router)
api_router.include_router(experiments_router)

__all__ = ["api_router"]
