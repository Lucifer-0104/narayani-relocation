"""
Narayani Relocation — minimal Streamlit demo UI.

This does NOT reimplement any logic. It imports the existing backend engine
(backend.simulator.RelocationEngine) directly and displays exactly what that
engine computes:

    DATA -> RISK -> RELOCATION PRIORITY -> CANDIDATE SITES -> CAPACITY
    -> OPTIMIZATION -> RESULT (recommended relocation plan)

All figures on screen come from RelocationEngine.run_scenario() /
RelocationEngine.baseline. Nothing is hardcoded. The underlying habitation /
site data is synthetic demo data (see backend/data_generator.py,
SOURCE_LABEL = "SYNTHETIC_DEMO_DATA") and is labelled as such below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the backend package importable when run as `streamlit run frontend/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.simulator import RelocationEngine  # noqa: E402

st.set_page_config(page_title="Narayani Relocation — Demo", layout="wide")


@st.cache_resource(show_spinner="Loading data and fitting viability model...")
def get_engine() -> RelocationEngine:
    return RelocationEngine()


def props_df(fc: dict, cols: list[str]) -> pd.DataFrame:
    rows = [f["properties"] for f in fc.get("features", [])]
    df = pd.DataFrame(rows)
    keep = [c for c in cols if c in df.columns]
    return df[keep] if not df.empty else df


st.title("Narayani Relocation — Prototype")
st.caption(
    "⚠️ All habitation and site data below is **synthetic demo data** "
    "(backend/data_generator.py), not real survey/census data. "
    "All scores, capacities, and the allocation plan are computed live by the "
    "existing backend pipeline — nothing on this page is hardcoded."
)

engine = get_engine()

# ---------------------------------------------------------------------------
# Sidebar — optional what-if scenario controls (calls the existing simulator)
# ---------------------------------------------------------------------------
st.sidebar.header("Scenario (what-if)")
st.sidebar.caption("Re-runs the full backend pipeline with perturbed inputs.")
flood_mult = st.sidebar.slider("Flood frequency multiplier", 0.4, 3.0, 1.0, 0.1)
pop_growth = st.sidebar.slider("Population growth", 0.7, 1.8, 1.0, 0.05)
land_factor = st.sidebar.slider("Land availability factor", 0.4, 1.4, 1.0, 0.05)
infra_boost = st.sidebar.slider("Infrastructure boost", 0.0, 0.8, 0.0, 0.05)
viability_threshold = st.sidebar.slider("Viability threshold override", 0.2, 0.8, 0.50, 0.01)
run_scenario = st.sidebar.button("Run scenario", type="primary")

if run_scenario:
    with st.spinner("Re-running risk -> capacity -> optimization pipeline..."):
        result = engine.run_scenario(
            flood_frequency_multiplier=flood_mult,
            population_growth=pop_growth,
            land_availability_factor=land_factor,
            infrastructure_boost=infra_boost,
            viability_threshold=viability_threshold,
        )
    st.sidebar.success(f"Scenario solved — status: {result['kpis']['solver_status']}")
else:
    result = engine.baseline
    st.sidebar.info("Showing baseline (scenario = 1.0x defaults).")

habitations = result["habitations"]
safe_sites = result["safe_sites"]
allocation = result["allocation"]
kpis = result["kpis"]

# ---------------------------------------------------------------------------
# Top-line KPIs
# ---------------------------------------------------------------------------
st.subheader("District overview (computed by summarize_kpis)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total population", f"{kpis['total_population']:,}")
c2.metric("At-risk habitations", kpis["at_risk_habitations"])
c3.metric("Population to relocate", f"{kpis['relocate_population']:,}")
c4.metric("Total safe-site capacity", f"{kpis['total_capacity']:,}")
c5.metric("Mean risk score", f"{kpis['mean_risk']:.3f}")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Assigned population", f"{kpis['assigned_population']:,}")
c7.metric("Unassigned population", f"{kpis['unassigned_population']:,}")
c8.metric("Mean relocation distance", f"{kpis['mean_distance_km']:.1f} km")
c9.metric("Solver status", kpis["solver_status"])

st.divider()

# ---------------------------------------------------------------------------
# Step 1-2: Select a habitation, view its risk / relocation info
# ---------------------------------------------------------------------------
st.header("1. Habitation risk & relocation priority")

hab_df = props_df(
    habitations,
    ["id", "name", "block", "population", "risk_score", "risk_band",
     "viability_score", "is_viable", "urgency_score", "urgency_rank", "action"],
).sort_values("risk_score", ascending=False)

st.dataframe(hab_df, width='stretch', hide_index=True)

hab_names = {f["properties"]["name"]: f["properties"]["id"] for f in habitations["features"]}
selected_name = st.selectbox("Select a habitation to inspect", list(hab_names.keys()))
selected = next(
    f["properties"] for f in habitations["features"] if f["properties"]["name"] == selected_name
)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"### {selected['name']} ({selected['id']})")
    st.write(f"**Block:** {selected['block']}  |  **Population:** {selected['population']:,}")
    st.metric("Composite risk score", f"{selected['risk_score']:.3f}", selected["risk_band"])
    st.write(
        f"Hazard index: **{selected['hazard_index']:.3f}**  |  "
        f"Vulnerability index: **{selected['vulnerability_index']:.3f}**  |  "
        f"Disruption index: **{selected['disruption_index']:.3f}**"
    )
with col_b:
    st.metric("Viability score", f"{selected['viability_score']:.3f}",
               "STAY" if selected["is_viable"] else "RELOCATE")
    st.write(
        f"Physics-rule viability: **{selected['physics_viability']:.3f}**  |  "
        f"ML stay probability: **{selected['ml_stay_probability']:.3f}**"
    )
    st.write(f"**Urgency score:** {selected['urgency_score']:.3f}"
             + (f" (rank #{selected['urgency_rank']} among relocating habitations)"
                if selected.get("urgency_rank") else ""))

st.write(f"**Recommended action:** {selected['action']}")
if selected.get("reasons"):
    st.write("**Reasons:**")
    for r in selected["reasons"]:
        st.write(f"- {r}")

st.divider()

# ---------------------------------------------------------------------------
# Step 3-4: Candidate relocation sites + carrying capacity
# ---------------------------------------------------------------------------
st.header("2. Candidate relocation sites & carrying capacity")
st.caption(
    "C_max = min(land, water, power, school, health capacity) x hazard-safety "
    "x (1 - environmental constraint) − existing population. "
    "Sites below min capacity / min hazard safety are already filtered out by "
    "carrying_capacity.filter_safe_sites()."
)

site_df = props_df(
    safe_sites,
    ["id", "name", "max_capacity", "binding_constraint", "hazard_safety",
     "gross_capacity", "existing_population"],
).sort_values("max_capacity", ascending=False)
st.dataframe(site_df, width='stretch', hide_index=True)

with st.expander("Capacity component breakdown per site"):
    comp_rows = []
    for f in safe_sites["features"]:
        p = f["properties"]
        row = {"id": p["id"], "name": p["name"], **p.get("capacity_components", {})}
        comp_rows.append(row)
    st.dataframe(pd.DataFrame(comp_rows), width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Step 5-6: Optimization result / recommended relocation plan
# ---------------------------------------------------------------------------
st.header("3. Optimization result — recommended relocation plan")

solver = allocation.get("solver", {})
s1, s2, s3, s4 = st.columns(4)
s1.metric("Solver status", solver.get("status", "n/a"))
s2.metric("Habitations to place", solver.get("habitations", 0))
s3.metric("Candidate sites used", solver.get("sites", 0))
s4.metric("Solve time", f"{solver.get('solve_ms', 0):.1f} ms")

moves = allocation.get("moves", [])
if moves:
    moves_df = pd.DataFrame(moves)[
        ["habitation_name", "site_name", "population", "distance_km",
         "urgency_score", "risk_score", "phase"]
    ]
    st.subheader("Recommended moves (habitation -> site)")
    st.dataframe(moves_df, width='stretch', hide_index=True)
else:
    st.info("No relocation moves required under this scenario "
            f"(solver status: {solver.get('status', 'n/a')}).")

phases = allocation.get("phases", [])
if phases:
    st.subheader("Phased plan")
    phase_df = pd.DataFrame(
        [{"Phase": p["label"], "Habitations": p["habitation_count"],
          "Population": p["population"], "Mean distance (km)": p["mean_distance_km"]}
         for p in phases]
    )
    st.dataframe(phase_df, width='stretch', hide_index=True)

unassigned = allocation.get("unassigned", [])
if unassigned:
    st.subheader("⚠️ Unassigned population (insufficient residual capacity)")
    st.dataframe(pd.DataFrame(unassigned), width='stretch', hide_index=True)

st.subheader("Site loading after allocation")
site_load = allocation.get("site_load", [])
if site_load:
    load_df = pd.DataFrame(site_load)[
        ["site_name", "max_capacity", "assigned_population", "utilization", "binding_constraint"]
    ]
    st.dataframe(load_df, width='stretch', hide_index=True)

st.divider()
st.caption(
    "Narayani Relocation prototype — DATA -> RISK -> RELOCATION PRIORITY -> "
    "CANDIDATE SITE -> CAPACITY -> OPTIMIZATION -> RESULT. "
    "Backend: pipeline.py (risk/viability/urgency), carrying_capacity.py (C_max), "
    "optimization.py (PuLP/CBC MILP assignment)."
)
