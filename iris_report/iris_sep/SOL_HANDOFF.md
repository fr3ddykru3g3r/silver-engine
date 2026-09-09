# SOL continuation prompt

Continue IRIS-SEP on `codex/iris-sep-continuation-20260905`. Inspect PR #3 and the branch head first. Do not assume main contains this work.

## Current scientific direction

The freshness-crossover route is closed negative evidence and must not be rescued or retuned.

The authorized design direction is now:

`IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`

Central question:

> How much do repeated positive windows from the same physical SEP episode, already-active persistence states, and issue-time proton-history information alter apparent 24-hour SEP forecasting skill and model ranking?

Read these first:

- `CURRENT_STATUS.md`
- `architecture/EPISODE_NORMALIZED_CAUSAL_BENCHMARK_2026-09-09.md`
- `architecture/NOVELTY_BOUNDARY_EPISODE_NORMALIZED_BENCHMARK_2026-09-09.md`
- `architecture/LITERATURE_GAP_EPISODE_NORMALIZED_EVALUATION_2026-09-09.md`
- `architecture/JUDGE_9_OF_10_UPGRADE_PLAN_2026-09-09.md`
- `config/episode_normalized_causal_benchmark_v1_preregistration_2026-09-09.json`
- `config/inspected_evidence_registry_v2.json`
- `architecture/FRESHNESS_CROSSOVER_V1_CLOSURE_2026-09-09.md`
- `architecture/ONSET_CONTINUATION_FEASIBILITY_2026-09-09.md`

## What is already implemented

Core evaluator:

- `tools/episode_normalized_benchmark.py`

Tests:

- `tests/test_episode_normalized_benchmark.py`
- `tests/test_episode_normalized_benchmark_contract.py`

The evaluator currently provides:

- validation of terminal eligibility states;
- episode-normalized positive weights where each physical positive episode totals 1;
- Episode Multiplicity Factor;
- weighted confusion/TSS/FAR/POD scoring for onset-eligible rows;
- deterministic shared bootstrap draw-tensor generation;
- episode/quiet-block bootstrap unit construction;
- model ranking and pairwise rank-reversal detection.

## Protected-outcome rule — critical

Post-`2025-09-10T00:00:00Z` candidate data remain `PROTECTED_NOT_INSPECTED` under `config/inspected_evidence_registry_v2.json`.

Do NOT query, count, inspect, score, stratify, or reveal:

- protected candidate identities;
- labels/outcomes;
- event counts;
- episode durations;
- model scores;
- class balance.

Do not use protected data to decide power, thresholds, hypotheses, or study design.

Historical development/exposed data may be used to debug mechanics only. Never call it untouched final evidence.

## Next engineering milestone

Build the causal episode constructor and attrition-ledger pipeline on already-exposed development data only.

Required behavior before any model comparison:

1. construct physical threshold-crossing episodes deterministically;
2. classify issue times into the frozen terminal eligibility codes;
3. prove already-active cases never enter onset-positive scoring;
4. persist one prediction/eligibility row per issue time;
5. make attrition counts reconcile exactly;
6. generate one bootstrap draw tensor and reuse it for every paired comparison;
7. hash predictions, attrition ledger, config, environment and bootstrap draws;
8. add boundary tests for exact issue+24h endpoint, gaps, duplicates, immature windows and long events;
9. only then run the fixed comparator set on already-exposed development data as a mechanics/diagnostic result.

Fixed model families for V1:

- climatology;
- causally permitted persistence diagnostic;
- fixed elastic-net logistic regression;
- fixed XGBoost;
- existing frozen IRIS candidate only if the exact development feature interface is available.

No architecture expansion or hyperparameter rescue after observing episode-normalization shifts.

## Scientific claim boundary

Do not claim that existing SEP papers are biased or wrong before the frozen benchmark demonstrates a robust effect.

The candidate originality is the combined evaluation framework:

- quantify repeated positive-window multiplicity;
- equalize total positive weight per physical SEP episode;
- separate new onset from already-active persistence;
- test model-rank stability;
- use identical episode-level paired bootstrap draws;
- persist complete per-issue eligibility/attrition evidence.

Recent work already includes proton/XRS forecasting, XGBoost, deep learning, multimodality, 24-hour forecasts and operational onset/persistence concepts. Those are not novelty claims.

## Existing V3/fail-closed evidence

Preserve all prior V3 package, replay, missingness, filter, source-readiness and fail-closed results. They are historical evidence, not the new centerpiece.

The exact prospective V3 interface remains blocked by unavailable `CMASKL`, `MEANGBL`, and `USFLUXL` quantities affecting 18 frozen feature-vector positions. The prior prospective preflight emitted no forecast probability. Do not silently substitute or zero-fill them.

## Before ending any continuation

- run source-only CI/test suite on the exact published head;
- preserve null and negative evidence;
- update CURRENT_STATUS and this handoff with exact commit/test receipts;
- do not merge PR #3 automatically;
- do not access protected final outcomes;
- do not claim award outcome, operational superiority, economic impact, state-of-the-art performance, or breakthrough without independent evidence.
