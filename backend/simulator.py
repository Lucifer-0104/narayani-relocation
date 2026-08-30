"""
What-if scenario simulator (Module 9).

Perturbs flood frequency, population, land availability and infrastructure
investment, then re-runs the full scoring → capacity → allocation graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .carrying_capacity import annotate_sites, filter_safe_sites
from .config import DATA_DIR, VIABILITY_THRESHOLD
from .data_generator import ensure_data
from .optimization import optimize_allocation, summarize_kpis
from .pipeline import apply_scenario_to_habitations, apply_scenario_to_sites, run_pipeline


def load_geojson(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class RelocationEngine:
    """In-memory engine holding baseline layers and producing scenario runs."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        ensure_data(self.data_dir)
        self.base_habitations = load_geojson(self.data_dir / "mock_habitations.geojson")
        self.base_sites = load_geojson(self.data_dir / "mock_safe_sites.geojson")
        self.infrastructure = load_geojson(self.data_dir / "mock_infrastructure.geojson")
        self.context = load_geojson(self.data_dir / "mock_context.geojson")
        self.baseline = self.run_scenario()

    def run_scenario(
        self,
        flood_frequency_multiplier: float = 1.0,
        population_growth: float = 1.0,
        land_availability_factor: float = 1.0,
        infrastructure_boost: float = 0.0,
        site_id_boost: Optional[str] = None,
        viability_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        threshold = float(viability_threshold) if viability_threshold is not None else VIABILITY_THRESHOLD
        scenario = {
            "flood_frequency_multiplier": flood_frequency_multiplier,
            "population_growth": population_growth,
            "land_availability_factor": land_availability_factor,
            "infrastructure_boost": infrastructure_boost,
            "site_id_boost": site_id_boost,
            "viability_threshold": threshold,
        }

        hab_raw = apply_scenario_to_habitations(
            self.base_habitations,
            flood_frequency_multiplier=flood_frequency_multiplier,
            population_growth=population_growth,
        )
        site_raw = apply_scenario_to_sites(
            self.base_sites,
            land_availability_factor=land_availability_factor,
            infrastructure_boost=infrastructure_boost,
            site_id_boost=site_id_boost,
        )
        habitations = run_pipeline(hab_raw, threshold=threshold)
        sites = filter_safe_sites(annotate_sites(site_raw))
        allocation = optimize_allocation(habitations, sites)
        kpis = summarize_kpis(habitations, sites, allocation, scenario=scenario)

        return {
            "habitations": habitations,
            "safe_sites": sites,
            "infrastructure": self.infrastructure,
            "context": self.context,
            "allocation": allocation,
            "kpis": kpis,
            "scenario": scenario,
        }
