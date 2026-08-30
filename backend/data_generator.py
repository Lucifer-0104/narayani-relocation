"""
Synthetic data layer for the Narayani relocation prototype.

This file was empty in the original repo even though simulator.py calls
ensure_data() on startup and then loads four GeoJSON files from DATA_DIR.
Nothing here is a live/real data source -- it is clearly-labelled synthetic
demo data modelled loosely on a Terai river-basin district (flood-prone
habitations along the Narayani river, candidate relocation sites on higher,
better-served ground). All numbers are fixed (not randomised) so repeated
runs and the demo UI are reproducible.

ensure_data(data_dir) writes four files into data_dir if they don't already
exist:
    mock_habitations.geojson    -- raw (unscored) habitation attributes
    mock_safe_sites.geojson     -- raw (un-annotated) candidate relocation sites
    mock_infrastructure.geojson -- hospitals / schools / grid / waterworks (context only)
    mock_context.geojson        -- district boundary + river centreline (context only)

Every habitation/site property here is a *raw input* consumed by
pipeline.py / carrying_capacity.py -- none of the derived scores (risk,
viability, C_max, etc.) are pre-computed here; those come only from the
actual backend calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SOURCE_LABEL = "SYNTHETIC_DEMO_DATA"

# ---------------------------------------------------------------------------
# Habitations (raw attributes, pre-scoring)
# ---------------------------------------------------------------------------
# lon/lat are illustrative points along/near a Terai river basin. Not surveyed.
_HABITATIONS: list[dict[str, Any]] = [
    dict(id="H01", name="Rapti Tole", block="Rapti", lon=84.312, lat=27.612,
         population=4200, households=824,
         flood_exposure=0.88, erosion_m_yr=9.5, landslide_exposure=0.05, seismic_exposure=0.55,
         historical_events_10yr=7, land_loss_ha_10yr=42.0,
         poverty_ratio=0.42, kutcha_housing_ratio=0.61, elderly_ratio=0.11, child_ratio=0.34,
         road_access_km=6.5, nearest_hospital_km=14.0,
         elevation_m=58.0, distance_to_river_m=180.0, electricity_hours=9.0),
    dict(id="H02", name="Khairahani Basti", block="Khairahani", lon=84.345, lat=27.598,
         population=3100, households=608,
         flood_exposure=0.81, erosion_m_yr=11.2, landslide_exposure=0.03, seismic_exposure=0.50,
         historical_events_10yr=6, land_loss_ha_10yr=55.0,
         poverty_ratio=0.38, kutcha_housing_ratio=0.55, elderly_ratio=0.10, child_ratio=0.33,
         road_access_km=8.2, nearest_hospital_km=17.5,
         elevation_m=54.0, distance_to_river_m=90.0, electricity_hours=8.0),
    dict(id="H03", name="Ratnanagar Chowk", block="Ratnanagar", lon=84.402, lat=27.633,
         population=5600, households=1098,
         flood_exposure=0.42, erosion_m_yr=2.1, landslide_exposure=0.04, seismic_exposure=0.48,
         historical_events_10yr=2, land_loss_ha_10yr=6.0,
         poverty_ratio=0.24, kutcha_housing_ratio=0.28, elderly_ratio=0.09, child_ratio=0.29,
         road_access_km=2.1, nearest_hospital_km=5.0,
         elevation_m=98.0, distance_to_river_m=1400.0, electricity_hours=17.0),
    dict(id="H04", name="Kalika Danda", block="Kalika", lon=84.276, lat=27.655,
         population=2100, households=412,
         flood_exposure=0.18, erosion_m_yr=0.4, landslide_exposure=0.62, seismic_exposure=0.58,
         historical_events_10yr=3, land_loss_ha_10yr=3.0,
         poverty_ratio=0.46, kutcha_housing_ratio=0.58, elderly_ratio=0.13, child_ratio=0.31,
         road_access_km=11.0, nearest_hospital_km=22.0,
         elevation_m=210.0, distance_to_river_m=4200.0, electricity_hours=7.0),
    dict(id="H05", name="Bharatpur Ward 21", block="Bharatpur", lon=84.431, lat=27.667,
         population=8900, households=1745,
         flood_exposure=0.31, erosion_m_yr=1.0, landslide_exposure=0.02, seismic_exposure=0.45,
         historical_events_10yr=1, land_loss_ha_10yr=2.0,
         poverty_ratio=0.19, kutcha_housing_ratio=0.15, elderly_ratio=0.08, child_ratio=0.27,
         road_access_km=1.0, nearest_hospital_km=2.5,
         elevation_m=115.0, distance_to_river_m=2600.0, electricity_hours=21.0),
    dict(id="H06", name="Meghauli Tar", block="Meghauli", lon=84.199, lat=27.548,
         population=2650, households=520,
         flood_exposure=0.93, erosion_m_yr=14.8, landslide_exposure=0.06, seismic_exposure=0.52,
         historical_events_10yr=9, land_loss_ha_10yr=68.0,
         poverty_ratio=0.51, kutcha_housing_ratio=0.72, elderly_ratio=0.12, child_ratio=0.36,
         road_access_km=13.5, nearest_hospital_km=26.0,
         elevation_m=51.0, distance_to_river_m=60.0, electricity_hours=6.0),
    dict(id="H07", name="Jutpani Chowk", block="Jutpani", lon=84.288, lat=27.629,
         population=3450, households=676,
         flood_exposure=0.55, erosion_m_yr=4.6, landslide_exposure=0.10, seismic_exposure=0.50,
         historical_events_10yr=4, land_loss_ha_10yr=18.0,
         poverty_ratio=0.33, kutcha_housing_ratio=0.44, elderly_ratio=0.10, child_ratio=0.32,
         road_access_km=5.0, nearest_hospital_km=11.0,
         elevation_m=72.0, distance_to_river_m=650.0, electricity_hours=12.0),
    dict(id="H08", name="Divyanagar Tole", block="Divyanagar", lon=84.365, lat=27.581,
         population=1800, households=353,
         flood_exposure=0.76, erosion_m_yr=8.0, landslide_exposure=0.04, seismic_exposure=0.49,
         historical_events_10yr=5, land_loss_ha_10yr=31.0,
         poverty_ratio=0.40, kutcha_housing_ratio=0.53, elderly_ratio=0.14, child_ratio=0.30,
         road_access_km=9.8, nearest_hospital_km=19.0,
         elevation_m=56.0, distance_to_river_m=220.0, electricity_hours=8.5),
    dict(id="H09", name="Shivanagar Basti", block="Shivanagar", lon=84.256, lat=27.601,
         population=2950, households=578,
         flood_exposure=0.69, erosion_m_yr=6.2, landslide_exposure=0.18, seismic_exposure=0.53,
         historical_events_10yr=4, land_loss_ha_10yr=22.0,
         poverty_ratio=0.37, kutcha_housing_ratio=0.49, elderly_ratio=0.11, child_ratio=0.33,
         road_access_km=7.4, nearest_hospital_km=15.5,
         elevation_m=64.0, distance_to_river_m=340.0, electricity_hours=9.5),
    dict(id="H10", name="Padampur Sadak", block="Padampur", lon=84.418, lat=27.605,
         population=6100, households=1196,
         flood_exposure=0.24, erosion_m_yr=1.4, landslide_exposure=0.03, seismic_exposure=0.46,
         historical_events_10yr=1, land_loss_ha_10yr=4.0,
         poverty_ratio=0.21, kutcha_housing_ratio=0.20, elderly_ratio=0.09, child_ratio=0.28,
         road_access_km=1.6, nearest_hospital_km=4.0,
         elevation_m=104.0, distance_to_river_m=1900.0, electricity_hours=19.0),
    dict(id="H11", name="Gitanagar Ghat", block="Gitanagar", lon=84.230, lat=27.567,
         population=1450, households=284,
         flood_exposure=0.90, erosion_m_yr=13.1, landslide_exposure=0.05, seismic_exposure=0.51,
         historical_events_10yr=8, land_loss_ha_10yr=60.0,
         poverty_ratio=0.48, kutcha_housing_ratio=0.66, elderly_ratio=0.12, child_ratio=0.35,
         road_access_km=12.2, nearest_hospital_km=24.0,
         elevation_m=52.0, distance_to_river_m=75.0, electricity_hours=6.5),
    dict(id="H12", name="Chainpur Danda", block="Chainpur", lon=84.301, lat=27.671,
         population=1980, households=388,
         flood_exposure=0.15, erosion_m_yr=0.6, landslide_exposure=0.58, seismic_exposure=0.56,
         historical_events_10yr=3, land_loss_ha_10yr=2.0,
         poverty_ratio=0.44, kutcha_housing_ratio=0.56, elderly_ratio=0.13, child_ratio=0.32,
         road_access_km=10.5, nearest_hospital_km=21.0,
         elevation_m=195.0, distance_to_river_m=3900.0, electricity_hours=7.5),
]

# ---------------------------------------------------------------------------
# Candidate relocation ("safe") sites (raw attributes, pre-annotation)
# ---------------------------------------------------------------------------
_SAFE_SITES: list[dict[str, Any]] = [
    dict(id="S01", name="Jutpani Highland Plot", lon=84.270, lat=27.660,
         usable_land_ha=45.0, water_mld=3.2, electricity_kw=1800.0,
         school_seats=900.0, hospital_beds=20.0, hospitals=1.0,
         env_constraint=0.10, existing_population=300,
         flood_safety=0.92, landslide_safety=0.80, seismic_safety=0.78),
    dict(id="S02", name="Ratnanagar Resettlement Block", lon=84.410, lat=27.648,
         usable_land_ha=60.0, water_mld=4.5, electricity_kw=2600.0,
         school_seats=1400.0, hospital_beds=35.0, hospitals=1.0,
         env_constraint=0.08, existing_population=850,
         flood_safety=0.90, landslide_safety=0.88, seismic_safety=0.75),
    dict(id="S03", name="Kalika Danda Extension", lon=84.281, lat=27.671,
         usable_land_ha=22.0, water_mld=1.4, electricity_kw=750.0,
         school_seats=400.0, hospital_beds=6.0, hospitals=0.0,
         env_constraint=0.22, existing_population=150,
         flood_safety=0.95, landslide_safety=0.55, seismic_safety=0.70),
    dict(id="S04", name="Padampur Growth Corridor", lon=84.435, lat=27.617,
         usable_land_ha=80.0, water_mld=6.0, electricity_kw=3400.0,
         school_seats=2000.0, hospital_beds=48.0, hospitals=1.0,
         env_constraint=0.06, existing_population=1200,
         flood_safety=0.88, landslide_safety=0.90, seismic_safety=0.74),
    dict(id="S05", name="Chainpur Terrace", lon=84.312, lat=27.685,
         usable_land_ha=18.0, water_mld=1.1, electricity_kw=600.0,
         school_seats=350.0, hospital_beds=4.0, hospitals=0.0,
         env_constraint=0.30, existing_population=90,
         flood_safety=0.96, landslide_safety=0.52, seismic_safety=0.68),
    dict(id="S06", name="Bharatpur Periphery Site", lon=84.447, lat=27.652,
         usable_land_ha=35.0, water_mld=2.8, electricity_kw=1600.0,
         school_seats=800.0, hospital_beds=18.0, hospitals=1.0,
         env_constraint=0.12, existing_population=500,
         flood_safety=0.85, landslide_safety=0.92, seismic_safety=0.76),
]


def _habitation_feature(h: dict[str, Any]) -> dict[str, Any]:
    props = {k: v for k, v in h.items() if k not in ("lon", "lat")}
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
    }


def _site_feature(s: dict[str, Any]) -> dict[str, Any]:
    props = {k: v for k, v in s.items() if k not in ("lon", "lat")}
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
    }


def _build_habitations() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "mock_habitations",
        "source": SOURCE_LABEL,
        "features": [_habitation_feature(h) for h in _HABITATIONS],
    }


def _build_safe_sites() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "mock_safe_sites",
        "source": SOURCE_LABEL,
        "features": [_site_feature(s) for s in _SAFE_SITES],
    }


def _build_infrastructure() -> dict[str, Any]:
    """Context-only layer (not consumed by scoring/optimization math)."""
    points = [
        ("Bharatpur District Hospital", 84.431, 27.667, "hospital"),
        ("Ratnanagar PHC", 84.402, 27.633, "hospital"),
        ("Padampur Grid Substation", 84.435, 27.617, "power"),
        ("Jutpani Community School", 84.270, 27.660, "school"),
        ("Narayani Waterworks Intake", 84.320, 27.590, "water"),
    ]
    features = [
        {
            "type": "Feature",
            "properties": {"name": name, "kind": kind, "source": SOURCE_LABEL},
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        }
        for name, lon, lat, kind in points
    ]
    return {
        "type": "FeatureCollection",
        "name": "mock_infrastructure",
        "source": SOURCE_LABEL,
        "features": features,
    }


def _build_context() -> dict[str, Any]:
    """District boundary (illustrative bbox) + Narayani river centreline (illustrative)."""
    boundary = {
        "type": "Feature",
        "properties": {"name": "Demo District Boundary", "source": SOURCE_LABEL},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [84.18, 27.53], [84.46, 27.53], [84.46, 27.70],
                [84.18, 27.70], [84.18, 27.53],
            ]],
        },
    }
    river = {
        "type": "Feature",
        "properties": {"name": "Narayani River (illustrative)", "source": SOURCE_LABEL},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [84.199, 27.548], [84.230, 27.567], [84.256, 27.601],
                [84.312, 27.612], [84.345, 27.598], [84.365, 27.581],
            ],
        },
    }
    return {
        "type": "FeatureCollection",
        "name": "mock_context",
        "source": SOURCE_LABEL,
        "features": [boundary, river],
    }


_BUILDERS = {
    "mock_habitations.geojson": _build_habitations,
    "mock_safe_sites.geojson": _build_safe_sites,
    "mock_infrastructure.geojson": _build_infrastructure,
    "mock_context.geojson": _build_context,
}


def ensure_data(data_dir: Path) -> None:
    """Write the four mock GeoJSON layers into data_dir if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, builder in _BUILDERS.items():
        path = data_dir / filename
        if path.exists():
            continue
        with path.open("w", encoding="utf-8") as fh:
            json.dump(builder(), fh, indent=2)
