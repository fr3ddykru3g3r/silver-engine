# Prospective predictor pipeline implementation — 2026-09-11

## Status

`IMPLEMENTED_SOURCE_AND_PREDICTION_INFRASTRUCTURE — OUTCOMES STILL SEALED — RECURRING EXECUTION NOT AUTHORIZED`

This implementation closes a software gap; it does **not** convert the historical replay into prospective evidence. Protected post-2025 outcomes remain inaccessible to the development side.

## Pipeline

1. `config/prospective_live_input_sources_v1.json`
   - freezes the public NOAA/SWPC predictor endpoints used for source readiness;
   - fixes the required primary-GOES integral-proton semantics;
   - freezes a 15-minute maximum observation age at the 00:00 UTC issue time;
   - forbids interpolation, future fill, alternate-satellite substitution and protected-outcome access.

2. `tools/capture_prospective_input_snapshot_v1.py`
   - downloads only predictor-side NOAA/SWPC JSON resources;
   - preserves each raw response byte-for-byte;
   - records retrieval timestamps, response metadata, byte counts and SHA-256;
   - requires the proton and XRS schemas to expose their expected keys;
   - selects the latest finite >=10 MeV proton value at or before the frozen issue time;
   - rejects ambiguous latest proton measurements;
   - marks stale or post-issue snapshots as blocked;
   - never sets `features_verified_causal=true`.

3. `config/past_proton_active_proxy_rule_v1.json`
   - freezes the deterministic mechanism comparator;
   - alert = 1 only when the selected pre-issue >=10 MeV integral proton flux is >=10 pfu;
   - output probability is deliberately binary because this comparator is a state diagnostic, not a calibrated learned forecast;
   - missing, stale or invalid input means `ABSTAIN`.

4. `tools/build_past_proton_proxy_prediction_v1.py`
   - validates raw-input hashes and receipt timing before prediction;
   - refuses to create a prediction after the issue time;
   - hashes the rule, source contract, feature schema and input receipt;
   - appends the prediction to a SHA-256 hash-chained ledger;
   - rejects duplicate issue/model records;
   - never reads outcomes.

5. `tools/verify_prospective_prediction_ledger_v1.py`
   - independently replays every record hash and chain link;
   - verifies strictly increasing daily 00:00 UTC issues;
   - verifies prediction and selected-observation times do not exceed issue time;
   - rejects malformed hashes, duplicate issues and deterministic-rule inconsistencies;
   - reports only predictor-ledger metadata.

6. `.github/workflows/iris-sep-prospective-predictor-capture.yml`
   - intentionally **manual only** (`workflow_dispatch`);
   - has no schedule and no outcome/evaluation step;
   - uploads only the immutable predictor snapshot;
   - cannot alter the causal-verification flag.

7. `tools/run_custodian_prospective_episode_evaluation_v1.py`
   - remains the sealed outcome-side evaluator;
   - is separate from every predictor-capture component above;
   - requires custodian mode and an `EXECUTION_FROZEN` manifest before protected evaluation.

## Why the manual workflow is intentional

A scheduled live study must not begin accidentally before the student researchers have the correct fair/SRC approval and before the execution manifest has been frozen. GitHub Actions scheduling also does not guarantee exact wall-clock execution, so the scientific record must be based on the actual recorded retrieval and prediction timestamps rather than assuming a cron trigger was punctual.

The manual workflow therefore proves the implementation can capture timestamped public inputs without opening protected outcomes. Recurring prospective data collection should begin only after approval and a dated execution freeze.

## Freeze path before any prospective outcome access

The remaining steps are procedural rather than model development:

1. student/adult sponsor confirms the prospective phase is covered by the approved research plan;
2. collect predictor-only source-timing receipts without inspecting outcomes;
3. verify the exact rule and feature-schema hashes to enter the execution manifest;
4. set the manifest to `EXECUTION_FROZEN` only after the causal-input gate is genuinely satisfied;
5. begin append-only pre-issue predictions;
6. keep labels and episode identities under independent custody;
7. evaluate only when the preregistered information floor is reached or the frozen study endpoint is otherwise reached;
8. accept confirmation, non-confirmation or insufficient information without redesigning the cohort.

## Claim boundary

The software is now capable of creating auditable pre-issue predictor receipts and deterministic predictions. No prospective forecast-skill result exists yet, and no protected result was accessed while building this pipeline.
