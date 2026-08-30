"""
Multi-hazard risk intelligence pipeline (Modules 1–4).

Takes raw habitation attributes (NDEM/IMD/CWC analogues + census + infra)
and produces:

  * hazard / vulnerability / disruption indices
  * a composite risk score
  * a stay-vs-relocate viability label (physics rule + sklearn model)
  * a multi-criteria urgency rank for relocation sequencing
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .config import (
    HAZARD_WEIGHTS,
    RISK_WEIGHTS,
    URGENCY_WEIGHTS,
    VIABILITY_THRESHOLD,
    VULNERABILITY_WEIGHTS,
)

logger = logging.getLogger(__name__)

FEATURE_KEYS = [
    "flood_exposure",
    "erosion_norm",
    "landslide_exposure",
    "seismic_exposure",
    "historical_events_10yr",
    "land_loss_norm",
    "poverty_ratio",
    "kutcha_housing_ratio",
    "age_dependency",
    "isolation_index",
    "elevation_norm_inv",
    "river_norm_inv",
    "electricity_norm_inv",
    "log_population",
]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _norm(value: float, hi: float) -> float:
    if hi <= 0:
        return 0.0
    return _clip01(value / hi)


def _props(feat: dict[str, Any]) -> dict[str, Any]:
    return feat.get("properties") or {}


def derive_features(p: dict[str, Any]) -> dict[str, float]:
    """Normalise raw attributes into the 0–1 feature space used by scoring + ML."""
    flood = float(p.get("flood_exposure", 0.0))
    erosion = float(p.get("erosion_m_yr", 0.0))
    slide = float(p.get("landslide_exposure", 0.0))
    seis = float(p.get("seismic_exposure", 0.0))
    events = float(p.get("historical_events_10yr", 0.0))
    land_loss = float(p.get("land_loss_ha_10yr", 0.0))
    poverty = float(p.get("poverty_ratio", 0.0))
    kutcha = float(p.get("kutcha_housing_ratio", 0.0))
    elderly = float(p.get("elderly_ratio", 0.0))
    child = float(p.get("child_ratio", 0.0))
    road = float(p.get("road_access_km", 0.0))
    hosp = float(p.get("nearest_hospital_km", 0.0))
    elev = float(p.get("elevation_m", 80.0))
    river = float(p.get("distance_to_river_m", 1000.0))
    elec = float(p.get("electricity_hours", 12.0))
    pop = max(1.0, float(p.get("population", 1)))

    isolation = _clip01(0.55 * _norm(road, 12.0) + 0.45 * _norm(hosp, 20.0))
    age_dep = _clip01((elderly / 0.18) * 0.5 + (child / 0.36) * 0.5)
    erosion_norm = _norm(erosion, 28.0)
    land_loss_norm = _norm(land_loss, 70.0)
    # Invert elevation / river-distance / power so that *higher* = more risk.
    elevation_norm_inv = _clip01(1.0 - (elev - 50.0) / 220.0)
    river_norm_inv = _clip01(1.0 - river / 5000.0)
    electricity_norm_inv = _clip01(1.0 - elec / 24.0)

    return {
        "flood_exposure": _clip01(flood),
        "erosion_norm": erosion_norm,
        "landslide_exposure": _clip01(slide),
        "seismic_exposure": _clip01(seis),
        "historical_events_10yr": events,
        "events_norm": _norm(events, 10.0),
        "land_loss_norm": land_loss_norm,
        "poverty_ratio": _clip01(poverty),
        "kutcha_housing_ratio": _clip01(kutcha),
        "age_dependency": age_dep,
        "isolation_index": isolation,
        "elevation_norm_inv": elevation_norm_inv,
        "river_norm_inv": river_norm_inv,
        "electricity_norm_inv": electricity_norm_inv,
        "log_population": math.log10(pop),
        "population": pop,
    }


def hazard_index(f: dict[str, float]) -> float:
    return _clip01(
        HAZARD_WEIGHTS["flood"] * f["flood_exposure"]
        + HAZARD_WEIGHTS["erosion"] * f["erosion_norm"]
        + HAZARD_WEIGHTS["landslide"] * f["landslide_exposure"]
        + HAZARD_WEIGHTS["seismic"] * f["seismic_exposure"]
    )


def vulnerability_index(f: dict[str, float]) -> float:
    return _clip01(
        VULNERABILITY_WEIGHTS["poverty"] * f["poverty_ratio"]
        + VULNERABILITY_WEIGHTS["kutcha"] * f["kutcha_housing_ratio"]
        + VULNERABILITY_WEIGHTS["age_dependency"] * f["age_dependency"]
        + VULNERABILITY_WEIGHTS["isolation"] * f["isolation_index"]
    )


def disruption_index(f: dict[str, float]) -> float:
    """Historical recurrence + land-loss trajectory + access collapse."""
    return _clip01(
        0.40 * f["events_norm"]
        + 0.35 * f["land_loss_norm"]
        + 0.25 * f["isolation_index"]
    )


def composite_risk(hazard: float, vuln: float, disruption: float) -> float:
    linear = (
        RISK_WEIGHTS["hazard"] * hazard
        + RISK_WEIGHTS["vulnerability"] * vuln
        + RISK_WEIGHTS["disruption"] * disruption
    )
    # Mild expansion in the upper tail so chronic hotspots separate cleanly.
    return _clip01(linear ** 0.92)


def viability_from_physics(f: dict[str, float], risk: float) -> tuple[float, list[str]]:
    """
    Transparent stay-vs-relocate score in [0, 1] (1 = safe to stay).

    Returns (score, human-readable reasons for relocation).
    """
    reasons: list[str] = []
    score = 1.0

    if f["flood_exposure"] >= 0.80:
        score -= 0.28
        reasons.append("Chronic flood exposure on the active belt")
    elif f["flood_exposure"] >= 0.65:
        score -= 0.16
        reasons.append("High flood frequency")

    if f["erosion_norm"] >= 0.55 and f["land_loss_norm"] >= 0.30:
        score -= 0.24
        reasons.append("Riverbank erosion with material land loss")
    elif f["erosion_norm"] >= 0.40:
        score -= 0.10
        reasons.append("Active bank-line retreat")

    if f["landslide_exposure"] >= 0.70:
        score -= 0.26
        reasons.append("Slope failure / landslide susceptibility")
    elif f["landslide_exposure"] >= 0.50:
        score -= 0.12
        reasons.append("Moderate landslide exposure")

    if f["events_norm"] >= 0.60:
        score -= 0.12
        reasons.append("Repeated disasters in the last decade")

    if f["isolation_index"] >= 0.70:
        score -= 0.10
        reasons.append("Poor all-weather access and distant healthcare")

    if risk >= 0.72:
        score -= 0.14
        reasons.append("Composite multi-hazard risk in the extreme band")
    elif risk >= 0.58:
        score -= 0.07

    if f["elevation_norm_inv"] >= 0.85 and f["flood_exposure"] >= 0.55:
        score -= 0.08
        reasons.append("Low-lying floodplain with no natural terrace")

    score = _clip01(score)
    if score >= 0.70 and not reasons:
        reasons.append("Within tolerable risk — in-situ strengthening preferred")
    return score, reasons


def urgency_score(f: dict[str, float], risk: float, viability: float) -> float:
    pop_term = _clip01((f["log_population"] - 3.0) / 1.3)  # 1k .. 20k
    return _clip01(
        URGENCY_WEIGHTS["risk"] * risk
        + URGENCY_WEIGHTS["population"] * pop_term
        + URGENCY_WEIGHTS["land_loss"] * f["land_loss_norm"]
        + URGENCY_WEIGHTS["trend"] * f["events_norm"]
        + URGENCY_WEIGHTS["isolation"] * f["isolation_index"]
        + 0.08 * (1.0 - viability)
    )


class ViabilityModel:
    """Gradient-boosted classifier trained on a synthetic historical analogue."""

    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
        )
        self._fitted = False

    def _sample_training_set(self, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        X = np.zeros((n, len(FEATURE_KEYS)))
        y = np.zeros(n, dtype=int)
        for i in range(n):
            flood = rng.uniform(0.05, 0.98)
            erosion = rng.uniform(0.0, 1.0)
            slide = rng.beta(1.4, 3.2)
            seis = rng.uniform(0.30, 0.85)
            events = rng.integers(0, 11)
            land_loss = rng.beta(1.5, 3.0)
            poverty = rng.uniform(0.12, 0.70)
            kutcha = rng.uniform(0.10, 0.90)
            age = rng.uniform(0.20, 0.90)
            isol = rng.uniform(0.05, 0.95)
            elev_inv = rng.uniform(0.05, 0.95)
            river_inv = rng.uniform(0.05, 0.95)
            elec_inv = rng.uniform(0.05, 0.80)
            log_pop = rng.uniform(2.8, 4.0)
            row = [
                flood,
                erosion,
                slide,
                seis,
                float(events),
                land_loss,
                poverty,
                kutcha,
                age,
                isol,
                elev_inv,
                river_inv,
                elec_inv,
                log_pop,
            ]
            X[i] = row
            f = dict(zip(FEATURE_KEYS, row))
            f["events_norm"] = _norm(f["historical_events_10yr"], 10.0)
            f["population"] = 10 ** f["log_population"]
            h = hazard_index(
                {
                    "flood_exposure": flood,
                    "erosion_norm": erosion,
                    "landslide_exposure": slide,
                    "seismic_exposure": seis,
                }
            )
            v = vulnerability_index(
                {
                    "poverty_ratio": poverty,
                    "kutcha_housing_ratio": kutcha,
                    "age_dependency": age,
                    "isolation_index": isol,
                }
            )
            d = disruption_index(
                {
                    "events_norm": f["events_norm"],
                    "land_loss_norm": land_loss,
                    "isolation_index": isol,
                }
            )
            risk = composite_risk(h, v, d)
            phys, _ = viability_from_physics({**f, "events_norm": f["events_norm"]}, risk)
            # Label: 1 = stay (viable), 0 = relocate. Flip a few for realism.
            label = 1 if phys >= 0.46 else 0
            if rng.random() < 0.06:
                label = 1 - label
            y[i] = label
        return X, y

    def fit(self) -> "ViabilityModel":
        X, y = self._sample_training_set()
        self.model.fit(X, y)
        self._fitted = True
        acc = float(self.model.score(X, y))
        logger.info("Viability model fitted on synthetic analogue (train acc=%.3f)", acc)
        return self

    def stay_probability(self, f: dict[str, float]) -> float:
        if not self._fitted:
            self.fit()
        vec = np.array([[f[k] for k in FEATURE_KEYS]], dtype=float)
        proba = self.model.predict_proba(vec)[0]
        classes = list(self.model.classes_)
        stay_idx = classes.index(1) if 1 in classes else 0
        return float(proba[stay_idx])


_MODEL: Optional[ViabilityModel] = None


def get_model() -> ViabilityModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = ViabilityModel().fit()
    return _MODEL


def score_habitation(
    props: dict[str, Any],
    threshold: float = VIABILITY_THRESHOLD,
    model: Optional[ViabilityModel] = None,
) -> dict[str, Any]:
    """Attach risk, viability and urgency onto a habitation properties dict."""
    f = derive_features(props)
    h = hazard_index(f)
    v = vulnerability_index(f)
    d = disruption_index(f)
    risk = composite_risk(h, v, d)
    phys, reasons = viability_from_physics(f, risk)
    model = model or get_model()
    ml_stay = model.stay_probability(f)
    # Blend: physics carries the policy argument, ML regularises edge cases.
    viability = _clip01(0.62 * phys + 0.38 * ml_stay)
    is_viable = 1 if viability >= threshold else 0
    urg = urgency_score(f, risk, viability)
    relocate_pop = 0 if is_viable else int(props.get("population", 0) or 0)

    if is_viable:
        action = "STAY — in-situ risk reduction"
    else:
        action = "RELOCATE — habitation not permanently viable"

    out = dict(props)
    out.update(
        {
            "hazard_index": round(h, 3),
            "vulnerability_index": round(v, 3),
            "disruption_index": round(d, 3),
            "risk_score": round(risk, 3),
            "viability_score": round(viability, 3),
            "ml_stay_probability": round(ml_stay, 3),
            "physics_viability": round(phys, 3),
            "is_viable": int(is_viable),
            "urgency_score": round(urg, 3),
            "relocate_population": relocate_pop,
            "action": action,
            "reasons": reasons,
            "risk_band": _risk_band(risk),
        }
    )
    return out


def _risk_band(risk: float) -> str:
    if risk >= 0.75:
        return "extreme"
    if risk >= 0.58:
        return "high"
    if risk >= 0.40:
        return "moderate"
    return "low"


def run_pipeline(
    habitations: dict[str, Any],
    threshold: float = VIABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Score every habitation. Returns a new FeatureCollection."""
    model = get_model()
    features = []
    for feat in habitations.get("features", []):
        props = score_habitation(_props(feat), threshold=threshold, model=model)
        features.append({**feat, "properties": props})
    # Rank urgency among those that must move (1 = most urgent).
    movers = [f for f in features if int(f["properties"].get("is_viable", 1)) == 0]
    movers.sort(key=lambda f: f["properties"]["urgency_score"], reverse=True)
    for rank, feat in enumerate(movers, start=1):
        feat["properties"]["urgency_rank"] = rank
    return {
        "type": "FeatureCollection",
        "name": habitations.get("name", "habitations"),
        "features": features,
    }


def apply_scenario_to_habitations(
    habitations: dict[str, Any],
    flood_frequency_multiplier: float = 1.0,
    population_growth: float = 1.0,
) -> dict[str, Any]:
    """Perturb raw attributes before re-scoring (what-if pre-processor)."""
    cloned = copy.deepcopy(habitations)
    for feat in cloned.get("features", []):
        p = feat["properties"]
        p["flood_exposure"] = _clip01(float(p.get("flood_exposure", 0)) * flood_frequency_multiplier)
        p["historical_events_10yr"] = int(
            round(min(12.0, float(p.get("historical_events_10yr", 0)) * flood_frequency_multiplier))
        )
        # Higher flood frequency accelerates bank erosion and land loss.
        erode_boost = 1.0 + 0.35 * max(0.0, flood_frequency_multiplier - 1.0)
        p["erosion_m_yr"] = float(p.get("erosion_m_yr", 0.0)) * erode_boost
        p["land_loss_ha_10yr"] = float(p.get("land_loss_ha_10yr", 0.0)) * erode_boost
        p["population"] = max(1, int(round(float(p.get("population", 1)) * population_growth)))
        p["households"] = max(1, int(round(p["population"] / 5.1)))
    return cloned


def apply_scenario_to_sites(
    sites: dict[str, Any],
    land_availability_factor: float = 1.0,
    infrastructure_boost: float = 0.0,
    site_id_boost: Optional[str] = None,
) -> dict[str, Any]:
    cloned = copy.deepcopy(sites)
    boost = max(0.0, float(infrastructure_boost))
    land = max(0.0, float(land_availability_factor))
    for feat in cloned.get("features", []):
        p = feat["properties"]
        p["usable_land_ha"] = float(p.get("usable_land_ha", 0.0)) * land
        apply = boost if (site_id_boost is None or p.get("id") == site_id_boost) else 0.0
        if apply:
            p["water_mld"] = float(p.get("water_mld", 0.0)) * (1.0 + apply)
            p["electricity_kw"] = float(p.get("electricity_kw", 0.0)) * (1.0 + apply)
            p["school_seats"] = float(p.get("school_seats", 0.0)) * (1.0 + apply)
            p["hospital_beds"] = float(p.get("hospital_beds", 0.0)) * (1.0 + apply)
    return cloned
