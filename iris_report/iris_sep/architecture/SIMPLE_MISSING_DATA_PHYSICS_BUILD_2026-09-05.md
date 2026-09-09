# Simple missing-data + physics build — 2026-09-05

## Status

**BUILT AND SOURCE-VERIFIED; REAL-DATA SCIENTIFIC EXPERIMENT NOT YET RUN.**

Tested source head: `c7135b3a0bba84cfb39431c2be7ac6834b23e507`.
GitHub Actions run `33978887134`, job `101340379074`:

- Ubuntu 24.04.4
- Python 3.13.5
- NumPy 2.3.5
- PyTorch 2.10.0+cpu
- scikit-learn 1.8.0
- 81 tests
- 0 failures
- 0 errors
- 0.477 s unittest time
- `py_compile`: PASS

Exact receipt: `receipts/simple_missing_data_physics_build_v1_2026-09-05.json`.

## Project in one sentence

> **When solar measurements temporarily go missing, can physics fill the gap well enough to keep a 24-hour solar-radiation-storm forecast useful?**

## What is now executable

### 1. Forecast-time missing-data benchmark

Runner: `tools/run_missingness_forecast_benchmark.py`

Safe package contract: `config/missingness_experiment_package_v1.json`

The runner:

1. accepts only a verified train-only NEW-crossing package;
2. rejects locked/test roles before model fitting;
3. fits a simple reference forecaster on the fit role;
4. freezes its calibration on the calibration role;
5. freezes its threshold on the threshold role;
6. only then hides real, nonstructural measurements in the score role;
7. compares no fill, train-fit median and causal forward fill using the same frozen forecaster;
8. evaluates the recovery arms against the complete-data reference on identical retained identities;
9. records coverage/abstention as well as TSS, FAR, Brier and ECE degradation;
10. writes immutable preregistration, holdout, prediction and receipt artifacts.

Features with zero fit-era observed support are not artificially reconstructed. They are excluded from the outage experiment because the reference forecaster never legitimately learned them.

### 2. Simple reduced-physics magnetic-map model

Source: `src/iris_sep/simple_physics.py`

Method ID: `ROTATE_SPREAD_2D_V1`

Plain-language description:

> **Take the last real magnetic map, move it sideways according to the declared solar-map rotation rate, and let the magnetic pattern spread slightly.**

The numerical model is a deliberately reduced 2-D advection + diffusion equation on an explicitly declared regular map grid. It is not full MHD, not a flare simulator and not ground truth.

The implementation requires explicit:

- longitude degrees per pixel;
- signed rotation degrees per day;
- grid-scale diffusion rate;
- numerical step ceiling;
- declared validation horizon.

No future observation is an input. If there is no earlier real map, the method cannot create one.

### 3. Hidden magnetic-map experiment

Runner: `tools/run_simple_physics_gap_benchmark.py`

Safe package contract: `config/magnetic_map_gap_package_v1.json`

The runner deliberately hides real, nonstructural train-only score-role maps and compares:

- **persistence:** reuse the last real map unchanged;
- **simple physics:** propagate the last real map with `ROTATE_SPREAD_2D_V1`.

For consecutive hidden maps, every reconstruction uses the most recent earlier **real** map. Synthetic output is not recursively promoted to observation. A hidden map without a causal prior real map is an abstention.

Pixel MAE/RMSE improvement is only an admission screen. It does not establish downstream SEP value.

## Why the physics and forecast experiments are separate

A tabular missing SHARP value is not automatically a physical state. Therefore the forecast robustness runner does **not** manufacture a `physics` value from a scalar column.

Physics is tested first where the physical object exists: magnetic maps. To enter the SEP benchmark later, the project must freeze a causal map-to-forecast-feature transformation and then demonstrate that the physics reconstruction preserves the downstream NEW-SEP forecast better than simpler recovery strategies.

This avoids the scientifically weak shortcut:

`missing scalar -> call a guessed number physics -> score it`.

## Safety and validity rules now encoded in tests

- structural historical unavailability cannot be treated as a transient outage;
- a structurally unavailable cell cannot also be observed;
- artificial holdouts may contain only genuinely observed nonstructural truth;
- hidden truth cannot affect predictions through a false observed mask;
- forward fill cannot read future values;
- the reference forecaster cannot retrain after the artificial outage;
- calibration and threshold roles are separated;
- a unit/episode cannot cross role boundaries in the executable package;
- strict chronological purge is checked from actual issue times;
- locked/test packages fail before fitting or scoring;
- magnetic physics cannot use a future map;
- consecutive hidden maps do not chain synthetic states as observations;
- no-prior-map physics case abstains;
- output directories are immutable;
- reconstruction/forecast receipts preserve claim boundaries.

## What has actually run

Only source/fixture verification has run in this GitHub environment.

The full source suite passed 81 tests and compilation at the tested source head. Synthetic fixtures prove that the executable paths, causality checks, role checks, abstention rules and receipt generation work as intended.

The real-data experiments remain **NOT_RUN** because ordinary Git does not contain:

1. the verified train-only NEW-crossing package/source manifest required by `missingness_experiment_package_v1.json`; or
2. a verified train-only magnetic-map package with authoritative source geometry required by `magnetic_map_gap_package_v1.json`.

No mixed table or locked data was downloaded to bypass that requirement.

## Exact next action

When the verified train-only NEW-crossing package exists, run:

```sh
python -m iris_report.iris_sep.tools.run_missingness_forecast_benchmark \
  --package <train_only_new_crossing.npz> \
  --metadata <train_only_new_crossing.metadata.json> \
  --output <immutable_output_dir>
```

When a verified train-only magnetic-map package with authoritative geometry exists, run:

```sh
python -m iris_report.iris_sep.tools.run_simple_physics_gap_benchmark \
  --package <train_only_magnetic_maps.npz> \
  --metadata <train_only_magnetic_maps.metadata.json> \
  --output <immutable_output_dir>
```

Do not tune physics from locked data. Do not promote physics merely because it looks plausible. Physics survives only if it first beats persistence on hidden real maps and then demonstrably preserves downstream NEW-SEP forecast utility better than the simpler recovery arms.

## Claim boundary

This build establishes executable research infrastructure, not a positive scientific result. It does not establish improved NEW-SEP skill, physics-reconstruction advantage, accurate solar simulation, MHD fidelity, operational readiness, company superiority, economic savings, breakthrough status or an award outcome.
