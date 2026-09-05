# IRIS-SEP GitHub continuation

Work branch: `codex/iris-sep-continuation-20260905`. Until its PR is merged, the project checkpoint is on that branch, not on `main`.

## READ THIS FIRST — executable simple missing-data + physics build — 2026-09-05

The current plain-language project is:

> **When solar measurements temporarily go missing, can physics fill the gap well enough to keep a 24-hour solar-radiation-storm forecast useful?**

Start with `PROJECT_IN_PLAIN_ENGLISH.md`, then `architecture/SIMPLE_MISSING_DATA_PHYSICS_BUILD_2026-09-05.md`. Exact verification is recorded in `receipts/simple_missing_data_physics_build_v1_2026-09-05.json`.

### What is built

The missing-data experiment is executable. `tools/run_missingness_forecast_benchmark.py` accepts only a safe train-only NEW-crossing package, fits and freezes a simple reference forecaster before the artificial outage, then compares mask-aware no-fill, train-fit median and causal forward-fill on deliberately hidden real score-time measurements. It rejects locked/test roles, role-crossing units, invalid chronological purge, and structural historical gaps before fitting or scoring.

The first physics candidate is also executable. `src/iris_sep/simple_physics.py` implements `ROTATE_SPREAD_2D_V1`: take the last real magnetic map, move it sideways according to an explicitly declared solar-map rotation rate, and let the magnetic pattern spread slightly using a reduced 2-D advection + diffusion equation. It is **not** full MHD and is not a solar-flare simulator.

`tools/run_simple_physics_gap_benchmark.py` deliberately hides real train-only magnetic maps and compares the physics reconstruction against last-real-map persistence. It never uses a future map, never recursively promotes a synthetic map to a real observation, and abstains if no earlier real map exists. Pixel-space improvement is only an admission screen; physics must later preserve downstream NEW-SEP forecast utility to enter the final model.

### Verification

Final source-tested head: `c7135b3a0bba84cfb39431c2be7ac6834b23e507`.

GitHub Actions run `33978887134`, job `101340379074` used Ubuntu 24.04.4, Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu and scikit-learn 1.8.0. It ran **81 tests in 0.477 seconds with zero failures and zero errors**, then passed `py_compile`. The earlier project-source scikit-learn deprecation warning was removed before this final run; remaining Node warnings are GitHub Actions runner notices, not project-source warnings.

### What has NOT run

The real-data scientific experiment is still **NOT_RUN**. Ordinary Git does not contain the verified train-only NEW-crossing package/source manifest required by `config/missingness_experiment_package_v1.json`, nor a verified train-only magnetic-map package with authoritative geometry required by `config/magnetic_map_gap_package_v1.json`. No mixed train/test table or locked data was downloaded to force a result.

Therefore no final NEW-crossing improvement, physics reconstruction advantage, accurate solar simulation, full-MHD result, operational readiness, company superiority, savings, breakthrough or award outcome is established.

### Exact next action

When the verified train-only NEW-crossing package is mounted, run `tools/run_missingness_forecast_benchmark.py` without changing the frozen experiment logic. If a verified train-only magnetic-map package with authoritative geometry is available, run `tools/run_simple_physics_gap_benchmark.py`. Physics may enter the downstream SEP forecast only after it beats persistence on hidden real maps and a causal map-to-feature path is frozen and shown to preserve forecast skill better than simpler recovery.

The older compact nonfinite replay remains a separate unresolved task and must retain its provenance gate. Final locked evaluation remains untouched.

## Prior history

For prior design decisions, negative experiments, compact replay status, Admission V2, source-era missingness analysis, and the original continuation history, read:

- `architecture/PUBLICATION_REVIEW_2026-09-05.md`
- `architecture/MISSING_DATA_PHYSICS_STRATEGY_2026-09-05.md`
- `architecture/ADR-006-structural-vs-transient-missingness.md`
- `architecture/ADR-007-observed-source-harmonization-before-reconstruction.md`
- `architecture/SOL_CONTINUATION_UPDATE_2026-09-05.md`
- `architecture/CONTINUATION_STATUS.md`
- `FUTURE_PLAN.md`
- `SOL_HANDOFF.md`
- `config/benchmark_contract_v2.json`
- `config/evaluation_policy_v1.json`
- `config/missingness_recovery_contract_v1.json`
- `evidence_checkpoint/INDEX.json`

This START_HERE intentionally points to the immutable historical documents instead of duplicating every older checkpoint inline. Failed experiments and their receipts remain preserved in the repository/history; this file is only the current continuation pointer.
