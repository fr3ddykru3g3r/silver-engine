# Phase II runtime compatibility correction — 2026-09-11

Status: **RUNTIME REPRESENTATION CORRECTION ONLY; NO PHASE II MODEL SCORES INSPECTED BEFORE THIS CHANGE**

The first execution attempt of `IRIS_SEP_ONSET_FORECASTER_PHASE2_V1` (GitHub Actions run `34625957229`, commit `acaa4091da74e40582aa788fad338d3966266005`) passed compilation, unit tests, and the protected-outcome boundary check, then terminated before model fitting with:

`RuntimeError: no modelable rows`

The failure occurred because the reused frozen freshness feature helper interpreted a timezone-aware pandas datetime array through the legacy raw `view('int64')` path. The frozen episode benchmark already has a reviewed compatibility implementation that normalizes the `DatetimeIndex` to nanoseconds and uses `.asi8` before applying the unchanged 24-hour feature-window arithmetic.

This correction installs that existing compatibility patch before the Phase II historical acquisition/build step. It also retains the previously documented XRS threshold-feature alias correction.

## Scientific boundary

This change does **not** alter:

- the target definition;
- the 2011–2017 historical development period;
- fit/calibration/threshold/score year roles;
- onset eligibility semantics;
- model families or feature families;
- XGBoost hyperparameters or random seeds;
- calibration policy;
- blend-alpha grid;
- threshold-selection policy;
- development success gate;
- physical-unit bootstrap policy;
- protected post-2025 outcome boundary.

The failed attempt produced no modelable cohort and therefore no Phase II fitted models, probabilities, thresholds, score metrics, or bootstrap result. The correction is consequently a representation/runtime fix, not result-informed model tuning.

The frozen episode-benchmark result remains untouched and this Phase II work remains development-only.
