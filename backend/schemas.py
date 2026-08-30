"""Pydantic contracts for the REST surface."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Parameter perturbation for the what-if engine (Module 9)."""

    flood_frequency_multiplier: float = Field(
        1.0,
        ge=0.4,
        le=3.0,
        description="Scales flood exposure and historical flood counts.",
    )
    population_growth: float = Field(
        1.0,
        ge=0.7,
        le=1.8,
        description="Multiplies habitation populations (e.g. 1.12 = +12%).",
    )
    land_availability_factor: float = Field(
        1.0,
        ge=0.4,
        le=1.4,
        description="Scales usable land (and therefore land-based C_max).",
    )
    infrastructure_boost: float = Field(
        0.0,
        ge=0.0,
        le=0.8,
        description="Fractional boost to water, power, schools and hospital beds.",
    )
    site_id_boost: Optional[str] = Field(
        None,
        description="If set, infrastructure_boost is applied only to this site.",
    )
    viability_threshold: Optional[float] = Field(
        None,
        ge=0.2,
        le=0.8,
        description="Override the stay-vs-relocate viability cut-off.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    habitations: int
    safe_sites: int


class ErrorResponse(BaseModel):
    detail: str
    error: str


class KPIBlock(BaseModel):
    total_population: int
    at_risk_habitations: int
    relocate_population: int
    stay_population: int
    total_capacity: int
    residual_capacity: int
    capacity_gap: int
    mean_risk: float
    assigned_population: int
    unassigned_population: int
    mean_distance_km: float
    solver_status: str
    scenario: dict[str, Any] = Field(default_factory=dict)
