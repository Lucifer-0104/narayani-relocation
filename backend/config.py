"""
Central configuration: paths + tunable weights/constants for the
Narayani relocation intelligence engine.

This file was empty in the original repo even though every other module
imports names from it. Nothing here changes the architecture described in
the other modules' docstrings -- these are exactly the constants they
already expect.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Module 1-4: risk / viability / urgency scoring
# ---------------------------------------------------------------------------

# Stay-vs-relocate cut-off on the blended viability score (0..1, 1 = safe to stay).
VIABILITY_THRESHOLD = 0.50

# Hazard index = weighted blend of flood / erosion / landslide / seismic exposure.
HAZARD_WEIGHTS = {
    "flood": 0.40,
    "erosion": 0.25,
    "landslide": 0.20,
    "seismic": 0.15,
}

# Vulnerability index = weighted blend of socio-economic exposure factors.
VULNERABILITY_WEIGHTS = {
    "poverty": 0.30,
    "kutcha": 0.30,
    "age_dependency": 0.20,
    "isolation": 0.20,
}

# Composite risk = weighted blend of hazard / vulnerability / disruption.
RISK_WEIGHTS = {
    "hazard": 0.45,
    "vulnerability": 0.30,
    "disruption": 0.25,
}

# Urgency score (relocation sequencing) weights.
URGENCY_WEIGHTS = {
    "risk": 0.40,
    "population": 0.20,
    "land_loss": 0.15,
    "trend": 0.15,
    "isolation": 0.10,
}

# ---------------------------------------------------------------------------
# Module 5-6: carrying capacity
# ---------------------------------------------------------------------------
PERSONS_PER_HA = 130.0             # settlement density cap used for C_land
WATER_LPCD = 135.0                 # litres per capita per day
KW_PER_HOUSEHOLD = 1.2             # connected electrical load per household
HOUSEHOLD_SIZE = 5.1               # persons per household (matches pipeline.py)
CHILD_SCHOOL_SHARE = 0.18          # share of population that is school-age
BEDS_PER_1000 = 2.0                # hospital beds required per 1000 residents
HEALTH_CATCHMENT_POP_PER_PHC = 30000.0  # population one PHC can realistically serve
MIN_HAZARD_SAFETY = 0.55           # floor applied to the hazard-safety derating factor

# ---------------------------------------------------------------------------
# Module 7-8: optimization / phased allocation
# ---------------------------------------------------------------------------
UNASSIGNED_PENALTY_KM = 500.0      # big-M penalty (person-km) for leaving demand unmet
PULP_MSG = 0                       # 0 = silent CBC solver, 1 = verbose
PULP_TIME_LIMIT_SEC = 20

# Urgency-based phase cut points (cumulative share of movers).
PHASE_CUTS = (0.30, 0.70)
PHASE_LABELS = {
    1: "Phase 1 -- Immediate (0-6 months)",
    2: "Phase 2 -- Near-term (6-18 months)",
    3: "Phase 3 -- Planned (18-36 months)",
}
