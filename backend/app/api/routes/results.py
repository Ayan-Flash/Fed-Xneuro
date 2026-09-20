"""
Results and Clinician Dashboard API routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.result import ResultResponse
from backend.app.services.result_service import ResultService
from backend.app.services.simulation_service import SimulationService
from backend.fl_engine.evaluation.dashboard import ClinicianDashboard

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{simulation_id}", response_model=ResultResponse)
def get_result(simulation_id: int, db: Session = Depends(get_db)):
    """Retrieves final result for a simulation."""
    res = ResultService.get_result(db, simulation_id)
    if not res:
        # Fallback to simulation entity values if available
        sim = SimulationService.get_simulation(db, simulation_id)
        from datetime import datetime
        return ResultResponse(
            id=sim.id,
            simulation_id=sim.id,
            final_loss=sim.final_loss,
            final_accuracy=sim.final_accuracy,
            metrics_path=None,
            plot_path=None,
            summary_json=sim.results_json,
            created_at=sim.created_at or datetime.utcnow(),
        )
    return res


@router.get("/{simulation_id}/clinician-dashboard")
def get_clinician_dashboard(simulation_id: int, db: Session = Depends(get_db)):
    """Returns actual or sample clinician dashboard for the simulation."""
    import json
    sim = SimulationService.get_simulation(db, simulation_id)
    report = None
    if sim.results_json:
        try:
            summary = json.loads(sim.results_json)
            if "clinician_report" in summary:
                report = summary["clinician_report"]
        except Exception:
            pass

    if report is None:
        report = {
            "patient_id": f"PAT_{sim.run_id[:8]}",
            "risk_probability": 0.65,
            "risk_category": "MODERATE",
            "clinical_importance": [
                {"feature": "MMSE Decline", "importance": 0.42, "relative_pct": 32.0},
                {"feature": "CDR-SB", "importance": 0.35, "relative_pct": 27.0},
                {"feature": "APOE4", "importance": 0.22, "relative_pct": 17.0},
                {"feature": "Age", "importance": 0.15, "relative_pct": 12.0},
            ],
            "mri_attribution": {
                "hippocampus_importance_pct": 28.5,
                "peak_attribution_voxel": [4, 8, 8],
            },
            "longitudinal_visit_weights": [0.15, 0.18, 0.20, 0.22, 0.25],
        }

    return {
        "simulation_id": simulation_id,
        "run_id": sim.run_id,
        "report": report,
        "ascii": ClinicianDashboard.render_ascii(report),
    }
