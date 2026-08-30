"""
FastAPI wrapper around the Narayani relocation intelligence engine.

Endpoints
---------
GET  /health              liveness + layer counts
GET  /api/habitations     scored habitation GeoJSON
GET  /api/safe-sites      safe-site GeoJSON with C_max
GET  /api/infrastructure  hospitals, schools, grid, waterworks
GET  /api/context         district boundary + river
GET  /api/allocations     phased relocation plan (triggers solver)
GET  /api/dashboard       aggregated payload for the command centre
POST /api/simulate        what-if re-run with perturbed parameters
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .schemas import SimulateRequest
from .simulator import RelocationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("narayani.api")

engine: RelocationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initialising relocation engine and fitting viability model")
    engine = RelocationEngine()
    app.state.engine = engine
    k = engine.baseline["kpis"]
    logger.info(
        "Baseline ready: %s habitations, %s at-risk, C_max=%s",
        k["habitation_count"],
        k["at_risk_habitations"],
        k["total_capacity"],
    )
    yield
    engine = None


app = FastAPI(
    title="Narayani Relocation Intelligence",
    description=(
        "Multi-hazard risk → habitation viability → carrying capacity → "
        "phased allocation → what-if planning."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _eng() -> RelocationEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine is still initialising")
    return engine


@app.get("/health")
def health() -> dict[str, Any]:
    eng = _eng()
    return {
        "status": "ok",
        "version": __version__,
        "habitations": len(eng.base_habitations.get("features", [])),
        "safe_sites": len(eng.base_sites.get("features", [])),
    }


@app.get("/api/habitations")
def get_habitations() -> dict[str, Any]:
    """GeoJSON of all habitations with risk, viability and urgency scores."""
    try:
        return _eng().baseline["habitations"]
    except Exception as exc:  # noqa: BLE001
        logger.exception("habitations endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/safe-sites")
def get_safe_sites() -> dict[str, Any]:
    """GeoJSON of candidate relocation sites with carrying-capacity breakdown."""
    try:
        return _eng().baseline["safe_sites"]
    except Exception as exc:  # noqa: BLE001
        logger.exception("safe-sites endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/infrastructure")
def get_infrastructure() -> dict[str, Any]:
    return _eng().infrastructure


@app.get("/api/context")
def get_context() -> dict[str, Any]:
    return _eng().context


@app.get("/api/allocations")
def get_allocations() -> dict[str, Any]:
    """Trigger (cached baseline) and return the optimised, phased relocation plan."""
    try:
        payload = _eng().baseline
        return {
            "allocation": payload["allocation"],
            "kpis": payload["kpis"],
            "scenario": payload["scenario"],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("allocations endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dashboard")
def get_dashboard() -> dict[str, Any]:
    """Single round-trip payload for the command-centre UI."""
    try:
        return _eng().baseline
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/simulate")
def simulate(body: SimulateRequest) -> dict[str, Any]:
    """
    Re-run the full graph under a perturbed scenario.

    The baseline is not mutated; the response is a complete alternative plan.
    """
    try:
        result = _eng().run_scenario(
            flood_frequency_multiplier=body.flood_frequency_multiplier,
            population_growth=body.population_growth,
            land_availability_factor=body.land_availability_factor,
            infrastructure_boost=body.infrastructure_boost,
            site_id_boost=body.site_id_boost,
            viability_threshold=body.viability_threshold,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("simulation failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc
