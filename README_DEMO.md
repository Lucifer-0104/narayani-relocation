# Narayani Relocation — Demo Run Instructions

## What was broken vs. what already worked

Already working, untouched:
- `schemas.py`, `main.py` (FastAPI), `simulator.py`, `pipeline.py`
  (risk/viability/urgency), `carrying_capacity.py` (C_max),
  `optimization.py` (PuLP/CBC MILP allocation + phasing).

Blocking bugs fixed (nothing else was touched):
- `config.py` was **completely empty** but every other module imports
  constants from it (`DATA_DIR`, `VIABILITY_THRESHOLD`, weight dicts, capacity
  constants, solver/phase constants). Filled in with the exact names every
  module already expects.
- `data_generator.py` was **completely empty** but `simulator.py` calls
  `ensure_data(DATA_DIR)` on startup and then loads four GeoJSON files.
  Filled in with `ensure_data()` that writes fixed (non-random), clearly
  labelled synthetic demo data (`SOURCE_LABEL = "SYNTHETIC_DEMO_DATA"`):
  12 habitations + 6 candidate relocation sites + infrastructure/context
  layers, loosely modelled on a Terai river-basin district.
- No `data/` folder and no `verify_pipeline.py` existed in the repo as
  cloned — `data/` is now created automatically by `ensure_data()` on first
  run; a manual verification was done instead (see below) since
  `verify_pipeline.py` wasn't present to run.

Everything else (risk scoring, viability model, carrying capacity, MILP
optimization, phased allocation) is the original code, unmodified.

## Verified end-to-end (baseline scenario)

- 12 habitations, 6 candidate sites loaded from generated GeoJSON.
- Risk/viability/urgency scored for all 12 habitations.
- 2 habitations flagged non-viable (Meghauli Tar, Gitanagar Ghat) → 4,100
  people to relocate.
- Carrying capacity computed for all 6 sites (total capacity 20,323).
- MILP solver (CBC via PuLP) status: **Optimal**, all 4,100 people assigned,
  0 unassigned.
- Phased plan: Phase 1 (2,650 people), Phase 2 (1,450 people).
- What-if scenario (flood x2) re-verified: at-risk habitations 2 → 6,
  relocate population 4,100 → 16,150 (solver still Optimal).

## Run it

```bash
cd narayani-relocation
pip install -r backend/req.txt streamlit
streamlit run frontend/app.py
```

First run auto-generates `backend/data/*.geojson` (synthetic demo data).
