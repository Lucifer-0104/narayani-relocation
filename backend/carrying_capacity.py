"""
Site suitability filter and carrying-capacity calculator (Modules 5–6).

C_max is the tightest of land, water, power and social-infrastructure caps,
derated by residual hazard and environmental lock-up:

    C_max = min(C_land, C_water, C_power, C_school, C_health)
            * hazard_safety
            * (1 - env_constraint)
            - existing_population
"""

from __future__ import annotations

from typing import Any

from .config import (
    BEDS_PER_1000,
    CHILD_SCHOOL_SHARE,
    HEALTH_CATCHMENT_POP_PER_PHC,
    HOUSEHOLD_SIZE,
    KW_PER_HOUSEHOLD,
    MIN_HAZARD_SAFETY,
    PERSONS_PER_HA,
    WATER_LPCD,
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def hazard_safety_factor(props: dict[str, Any]) -> float:
    """Geometric mean of flood / landslide / seismic safety, floored."""
    flood = float(props.get("flood_safety", 0.8))
    slide = float(props.get("landslide_safety", 0.8))
    seis = float(props.get("seismic_safety", 0.8))
    geo = (flood * slide * seis) ** (1.0 / 3.0)
    return max(MIN_HAZARD_SAFETY, _clip01(geo))


def compute_site_capacity(props: dict[str, Any]) -> dict[str, Any]:
    """Return component caps and the residual C_max for one safe site."""
    land_ha = float(props.get("usable_land_ha", 0.0))
    water_mld = float(props.get("water_mld", 0.0))
    power_kw = float(props.get("electricity_kw", 0.0))
    school_seats = float(props.get("school_seats", 0.0))
    beds = float(props.get("hospital_beds", 0.0))
    phc = float(props.get("hospitals", 0.0))
    env = _clip01(float(props.get("env_constraint", 0.0)))
    existing = int(props.get("existing_population", 0) or 0)

    c_land = land_ha * PERSONS_PER_HA
    c_water = (water_mld * 1_000_000.0) / WATER_LPCD if WATER_LPCD else 0.0
    c_power = (power_kw / KW_PER_HOUSEHOLD) * HOUSEHOLD_SIZE if KW_PER_HOUSEHOLD else 0.0
    c_school = school_seats / CHILD_SCHOOL_SHARE if CHILD_SCHOOL_SHARE else 0.0
    c_beds = beds / BEDS_PER_1000 * 1000.0 if BEDS_PER_1000 else 0.0
    c_phc = phc * HEALTH_CATCHMENT_POP_PER_PHC
    # Health is the more generous of beds vs PHC catchment — a site with a
    # strong PHC but few inpatient beds can still support a colony.
    c_health = max(c_beds, c_phc * 0.45)

    components = {
        "c_land": c_land,
        "c_water": c_water,
        "c_power": c_power,
        "c_school": c_school,
        "c_health": c_health,
    }
    binding = min(components, key=lambda k: components[k])
    raw = max(0.0, min(components.values()))
    safety = hazard_safety_factor(props)
    gross = raw * safety * (1.0 - env)
    residual = max(0, int(round(gross - existing)))

    limiting = {
        "c_land": "usable land",
        "c_water": "water supply",
        "c_power": "electricity",
        "c_school": "school seats",
        "c_health": "healthcare",
    }[binding]

    return {
        "hazard_safety": round(safety, 3),
        "capacity_components": {k: int(round(v)) for k, v in components.items()},
        "binding_constraint": limiting,
        "gross_capacity": int(round(gross)),
        "existing_population": existing,
        "max_capacity": residual,
        "c_max": residual,
    }


def annotate_sites(fc: dict[str, Any]) -> dict[str, Any]:
    """Attach C_max and component breakdown onto each safe-site feature."""
    out = {"type": "FeatureCollection", "features": []}
    if "name" in fc:
        out["name"] = fc["name"]
    for feat in fc.get("features", []):
        props = dict(feat.get("properties") or {})
        props.update(compute_site_capacity(props))
        out["features"].append({**feat, "properties": props})
    return out


def filter_safe_sites(
    fc: dict[str, Any],
    min_capacity: int = 50,
    min_safety: float = 0.60,
) -> dict[str, Any]:
    """Drop sites that cannot host a viable colony."""
    kept = []
    for feat in fc.get("features", []):
        p = feat.get("properties") or {}
        if int(p.get("max_capacity", 0)) < min_capacity:
            continue
        if float(p.get("hazard_safety", 0)) < min_safety:
            continue
        kept.append(feat)
    return {**fc, "features": kept}
