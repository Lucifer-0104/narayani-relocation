"""
Optimization & Phased Allocation Engine (Modules 7–8).

Solves a minimum-cost population assignment of non-viable habitations onto
safe sites:

    minimise   Σ_i Σ_j  pop_i · dist_ij · x_ij  +  P · Σ_i pop_i · u_i
    s.t.       Σ_j x_ij + u_i = 1                 ∀ habitations i
               Σ_i pop_i · x_ij  ≤  C_j           ∀ sites j
               x_ij ≥ 0,  u_i ≥ 0

x_ij is the fraction of habitation i sent to site j (habitations may split
when a single site cannot absorb them). u_i captures unmet demand so the
solver always returns a feasible plan when capacity is short.

Phasing then buckets assigned habitations by urgency_score:

    Phase 1 — top 30% most urgent
    Phase 2 — next 40%
    Phase 3 — remainder
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Iterable, Optional

from .config import PHASE_CUTS, PHASE_LABELS, PULP_MSG, PULP_TIME_LIMIT_SEC, UNASSIGNED_PENALTY_KM

logger = logging.getLogger(__name__)

EARTH_KM = 6371.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def _coords(feat: dict[str, Any]) -> tuple[float, float]:
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or [0.0, 0.0]
    return float(coords[0]), float(coords[1])


def _props(feat: dict[str, Any]) -> dict[str, Any]:
    return feat.get("properties") or {}


def _non_viable(habitations: dict[str, Any]) -> list[dict[str, Any]]:
    movers = []
    for feat in habitations.get("features", []):
        p = _props(feat)
        if int(p.get("is_viable", 1)) == 0 and int(p.get("population", 0)) > 0:
            movers.append(feat)
    return movers


def _distance_matrix(
    movers: list[dict[str, Any]],
    sites: list[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    dist: dict[tuple[str, str], float] = {}
    for h in movers:
        h_lon, h_lat = _coords(h)
        hid = str(_props(h)["id"])
        for s in sites:
            s_lon, s_lat = _coords(s)
            sid = str(_props(s)["id"])
            dist[(hid, sid)] = round(haversine_km(h_lon, h_lat, s_lon, s_lat), 3)
    return dist


def _solve_pulp(
    movers: list[dict[str, Any]],
    sites: list[dict[str, Any]],
    dist: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[str, float], str, float]:
    """
    Returns (flows, unmet_fraction_by_habitation, solver_status, objective).
    flows[(hid, sid)] = assigned population (people, not fraction).
    """
    try:
       import pulp 
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PuLP is required for the allocation engine") from exc

    h_ids = [str(_props(h)["id"]) for h in movers]
    s_ids = [str(_props(s)["id"]) for s in sites]
    pop = {str(_props(h)["id"]): float(_props(h)["population"]) for h in movers}
    cap = {str(_props(s)["id"]): float(_props(s).get("max_capacity", 0)) for s in sites}

    prob = pulp.LpProblem("narayani_relocation_assignment", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (h_ids, s_ids), lowBound=0.0, upBound=1.0, cat="Continuous")
    u = pulp.LpVariable.dicts("unmet", h_ids, lowBound=0.0, upBound=1.0, cat="Continuous")

    prob += pulp.lpSum(
        pop[i] * dist[(i, j)] * x[i][j] for i in h_ids for j in s_ids
    ) + pulp.lpSum(UNASSIGNED_PENALTY_KM * pop[i] * u[i] for i in h_ids)

    for i in h_ids:
        prob += pulp.lpSum(x[i][j] for j in s_ids) + u[i] == 1.0, f"cover_{i}"

    for j in s_ids:
        prob += (
            pulp.lpSum(pop[i] * x[i][j] for i in h_ids) <= cap[j],
            f"cap_{j}",
        )

    solver = pulp.PULP_CBC_CMD(msg=PULP_MSG, timeLimit=PULP_TIME_LIMIT_SEC)
    status_code = prob.solve(solver)
    status = pulp.LpStatus.get(status_code, str(status_code))

    flows: dict[tuple[str, str], float] = {}
    unmet: dict[str, float] = {}
    for i in h_ids:
        unmet_frac = float(pulp.value(u[i]) or 0.0)
        if unmet_frac > 1e-6:
            unmet[i] = unmet_frac * pop[i]
        for j in s_ids:
            frac = float(pulp.value(x[i][j]) or 0.0)
            people = frac * pop[i]
            if people >= 1.0:
                flows[(i, j)] = people

    objective = float(pulp.value(prob.objective) or 0.0)
    return flows, unmet, status, objective


def _greedy_fallback(
    movers: list[dict[str, Any]],
    sites: list[dict[str, Any]],
    dist: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[str, float], str, float]:
    """Nearest-site greedy used only if the MILP solver fails."""
    remaining = {str(_props(s)["id"]): float(_props(s).get("max_capacity", 0)) for s in sites}
    ordered = sorted(movers, key=lambda f: float(_props(f).get("urgency_score", 0)), reverse=True)
    flows: dict[tuple[str, str], float] = {}
    unmet: dict[str, float] = {}
    person_km = 0.0
    s_ids = [str(_props(s)["id"]) for s in sites]
    for h in ordered:
        hid = str(_props(h)["id"])
        left = float(_props(h)["population"])
        for sid, _d in sorted(((s, dist[(hid, s)]) for s in s_ids), key=lambda t: t[1]):
            if left < 1 or remaining[sid] < 1:
                continue
            take = min(left, remaining[sid])
            flows[(hid, sid)] = flows.get((hid, sid), 0.0) + take
            remaining[sid] -= take
            left -= take
            person_km += take * dist[(hid, sid)]
        if left >= 1:
            unmet[hid] = left
    return flows, unmet, "Heuristic", person_km


def _index_by_id(features: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(_props(f)["id"]): f for f in features}


def _assign_phases(habitation_ids_by_urgency: list[str]) -> dict[str, int]:
    n = len(habitation_ids_by_urgency)
    if n == 0:
        return {}
    cut1 = max(1, int(math.ceil(n * PHASE_CUTS[0]))) if n > 1 else 1
    cut2 = max(cut1, int(math.ceil(n * PHASE_CUTS[1]))) if n > 2 else n
    # Keep at least one habitation in later phases when n is large enough.
    if n >= 3:
        cut1 = min(cut1, n - 2)
        cut2 = min(max(cut2, cut1 + 1), n - 1)
    mapping: dict[str, int] = {}
    for idx, hid in enumerate(habitation_ids_by_urgency):
        if idx < cut1:
            mapping[hid] = 1
        elif idx < cut2:
            mapping[hid] = 2
        else:
            mapping[hid] = 3
    return mapping


def optimize_allocation(
    habitations: dict[str, Any],
    safe_sites: dict[str, Any],
) -> dict[str, Any]:
    """
    Match displaced populations to safe sites and emit a phased relocation plan.

    Parameters
    ----------
    habitations:
        FeatureCollection that has already been scored by ``pipeline.run_pipeline``.
    safe_sites:
        FeatureCollection annotated with ``max_capacity`` by the carrying-capacity engine.

    Returns
    -------
    dict
        ``moves``, ``phases``, ``unassigned``, ``site_load``, ``kpis``, ``solver``.
    """
    t0 = time.perf_counter()
    movers = _non_viable(habitations)
    sites = [
        f
        for f in safe_sites.get("features", [])
        if float(_props(f).get("max_capacity", 0)) > 0
    ]

    empty = {
        "moves": [],
        "phases": [],
        "unassigned": [],
        "site_load": [],
        "flows_geojson": {"type": "FeatureCollection", "features": []},
        "kpis": {
            "relocate_population": 0,
            "assigned_population": 0,
            "unassigned_population": 0,
            "mean_distance_km": 0.0,
            "total_person_km": 0.0,
            "total_capacity": int(sum(float(_props(s).get("max_capacity", 0)) for s in sites)),
            "capacity_gap": 0,
            "solver_status": "Empty",
        },
        "solver": {"status": "Empty", "objective": 0.0, "solve_ms": 0.0},
    }

    if not movers:
        empty["kpis"]["solver_status"] = "NoRelocationNeeded"
        empty["solver"]["status"] = "NoRelocationNeeded"
        return empty
    if not sites:
        pop = int(sum(int(_props(h)["population"]) for h in movers))
        empty["unassigned"] = [
            {
                "habitation_id": str(_props(h)["id"]),
                "habitation_name": _props(h).get("name"),
                "population": int(_props(h)["population"]),
                "reason": "No safe sites with residual capacity",
            }
            for h in movers
        ]
        empty["kpis"].update(
            {
                "relocate_population": pop,
                "unassigned_population": pop,
                "capacity_gap": pop,
                "solver_status": "NoSites",
            }
        )
        empty["solver"]["status"] = "NoSites"
        return empty

    dist = _distance_matrix(movers, sites)
    try:
        flows, unmet, status, objective = _solve_pulp(movers, sites, dist)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PuLP solver failed (%s); falling back to greedy", exc)
        flows, unmet, status, objective = _greedy_fallback(movers, sites, dist)

    h_index = _index_by_id(movers)
    s_index = _index_by_id(sites)

    # Collapse per-habitation primary destination (largest flow) for phasing.
    assigned_ids = sorted({hid for hid, _sid in flows})
    urgency_order = sorted(
        assigned_ids,
        key=lambda hid: float(_props(h_index[hid]).get("urgency_score", 0.0)),
        reverse=True,
    )
    phase_of = _assign_phases(urgency_order)

    moves: list[dict[str, Any]] = []
    flow_features: list[dict[str, Any]] = []
    person_km = 0.0
    assigned_pop = 0.0

    for (hid, sid), people in sorted(flows.items(), key=lambda kv: -kv[1]):
        people_i = int(round(people))
        if people_i <= 0:
            continue
        h, s = h_index[hid], s_index[sid]
        hp, sp = _props(h), _props(s)
        d_km = dist[(hid, sid)]
        phase = int(phase_of.get(hid, 3))
        person_km += people_i * d_km
        assigned_pop += people_i
        h_lon, h_lat = _coords(h)
        s_lon, s_lat = _coords(s)
        move = {
            "habitation_id": hid,
            "habitation_name": hp.get("name"),
            "block": hp.get("block"),
            "site_id": sid,
            "site_name": sp.get("name"),
            "population": people_i,
            "distance_km": round(d_km, 2),
            "urgency_score": float(hp.get("urgency_score", 0.0)),
            "risk_score": float(hp.get("risk_score", 0.0)),
            "phase": phase,
            "origin": [h_lon, h_lat],
            "destination": [s_lon, s_lat],
        }
        moves.append(move)
        flow_features.append(
            {
                "type": "Feature",
                "properties": {
                    "habitation_id": hid,
                    "habitation_name": hp.get("name"),
                    "site_id": sid,
                    "site_name": sp.get("name"),
                    "population": people_i,
                    "distance_km": round(d_km, 2),
                    "phase": phase,
                    "urgency_score": float(hp.get("urgency_score", 0.0)),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[h_lon, h_lat], [s_lon, s_lat]],
                },
            }
        )

    moves.sort(key=lambda m: (m["phase"], -m["urgency_score"]))

    unassigned: list[dict[str, Any]] = []
    for hid, people in unmet.items():
        people_i = int(round(people))
        if people_i <= 0:
            continue
        hp = _props(h_index[hid])
        unassigned.append(
            {
                "habitation_id": hid,
                "habitation_name": hp.get("name"),
                "population": people_i,
                "urgency_score": float(hp.get("urgency_score", 0.0)),
                "reason": "Insufficient residual capacity across candidate sites",
            }
        )

    load_acc: dict[str, dict[str, Any]] = {}
    for s in sites:
        sp = _props(s)
        sid = str(sp["id"])
        load_acc[sid] = {
            "site_id": sid,
            "site_name": sp.get("name"),
            "max_capacity": int(sp.get("max_capacity", 0)),
            "assigned_population": 0,
            "utilization": 0.0,
            "binding_constraint": sp.get("binding_constraint"),
            "incoming": [],
        }
    for m in moves:
        bucket = load_acc[m["site_id"]]
        bucket["assigned_population"] += m["population"]
        bucket["incoming"].append(
            {
                "habitation_id": m["habitation_id"],
                "habitation_name": m["habitation_name"],
                "population": m["population"],
                "phase": m["phase"],
            }
        )
    site_load = []
    for row in load_acc.values():
        cap = max(1, row["max_capacity"])
        row["utilization"] = round(row["assigned_population"] / cap, 3)
        site_load.append(row)
    site_load.sort(key=lambda r: -r["utilization"])

    phases: list[dict[str, Any]] = []
    for phase_no in (1, 2, 3):
        phase_moves = [m for m in moves if m["phase"] == phase_no]
        hab_ids = list(dict.fromkeys(m["habitation_id"] for m in phase_moves))
        phases.append(
            {
                "phase": phase_no,
                "label": PHASE_LABELS[phase_no],
                "habitation_count": len(hab_ids),
                "population": int(sum(m["population"] for m in phase_moves)),
                "mean_distance_km": round(
                    (
                        sum(m["distance_km"] * m["population"] for m in phase_moves)
                        / max(1, sum(m["population"] for m in phase_moves))
                    ),
                    2,
                )
                if phase_moves
                else 0.0,
                "moves": phase_moves,
            }
        )

    relocate_pop = int(sum(int(_props(h)["population"]) for h in movers))
    assigned_i = int(round(assigned_pop))
    unmet_i = int(sum(u["population"] for u in unassigned))
    total_cap = int(sum(float(_props(s).get("max_capacity", 0)) for s in sites))
    mean_d = (person_km / assigned_pop) if assigned_pop else 0.0
    solve_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "moves": moves,
        "phases": phases,
        "unassigned": unassigned,
        "site_load": site_load,
        "flows_geojson": {"type": "FeatureCollection", "name": "allocation_flows", "features": flow_features},
        "kpis": {
            "relocate_population": relocate_pop,
            "assigned_population": assigned_i,
            "unassigned_population": unmet_i,
            "mean_distance_km": round(mean_d, 2),
            "total_person_km": round(person_km, 1),
            "total_capacity": total_cap,
            "capacity_gap": max(0, relocate_pop - total_cap),
            "solver_status": status,
        },
        "solver": {
            "status": status,
            "objective": round(float(objective), 2),
            "solve_ms": round(solve_ms, 1),
            "habitations": len(movers),
            "sites": len(sites),
            "arcs": len(moves),
        },
    }


def summarize_kpis(
    habitations: dict[str, Any],
    safe_sites: dict[str, Any],
    allocation: dict[str, Any],
    scenario: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Dashboard KPI block combining pipeline + allocation outputs."""
    feats = habitations.get("features", [])
    total_pop = int(sum(int((_props(f).get("population") or 0)) for f in feats))
    at_risk = [f for f in feats if int(_props(f).get("is_viable", 1)) == 0]
    relocate_pop = int(sum(int(_props(f)["population"]) for f in at_risk))
    stay_pop = total_pop - relocate_pop
    mean_risk = (
        sum(float(_props(f).get("risk_score", 0.0)) for f in feats) / max(1, len(feats))
    )
    total_cap = int(
        sum(int(_props(s).get("max_capacity", 0) or 0) for s in safe_sites.get("features", []))
    )
    alloc_kpis = allocation.get("kpis") or {}
    assigned = int(alloc_kpis.get("assigned_population", 0))
    residual = max(0, total_cap - assigned)
    return {
        "total_population": total_pop,
        "habitation_count": len(feats),
        "at_risk_habitations": len(at_risk),
        "relocate_population": relocate_pop,
        "stay_population": stay_pop,
        "total_capacity": total_cap,
        "residual_capacity": residual,
        "capacity_gap": int(alloc_kpis.get("capacity_gap", max(0, relocate_pop - total_cap))),
        "mean_risk": round(mean_risk, 3),
        "assigned_population": assigned,
        "unassigned_population": int(alloc_kpis.get("unassigned_population", 0)),
        "mean_distance_km": float(alloc_kpis.get("mean_distance_km", 0.0)),
        "solver_status": alloc_kpis.get("solver_status", "Unknown"),
        "scenario": scenario or {},
    }
