# IRIS-SEP — START HERE

**Active branch:** `codex/iris-sep-proton-signal-audit-20260921`  
**Status date:** 2026-09-21

## Current project

> **When no >=10 pfu solar radiation storm is already active, do sub-threshold >10 MeV proton measurements from the previous 24 hours improve prediction of a new >=10 pfu SEP onset in the next 24 hours?**

Read first: [`ACTIVE_PROJECT_2026-09-21.md`](ACTIVE_PROJECT_2026-09-21.md).

This is an IB Physics-level controlled feature-ablation study. XGBoost is a standard tool, not the novelty. The physics question is whether recent proton flux contains genuine pre-onset information or whether much of its apparent value in ordinary 24-hour forecasting comes from persistence after a storm is already active.

## Novelty boundary

Prior literature already establishes that:

- onset and persistence are different SEP forecasting problems;
- preceding proton flux can be a strong predictor;
- XGBoost has been used for SEP prediction;
- persistence forecasts have been used as baselines;
- sub-threshold proton increases have been used to update SEP probabilities dynamically.

Therefore none of those observations is claimed as new.

The active novelty target is narrower: **quantify, on the same modern 24-hour benchmark and under matched model/evaluation conditions, how much apparent proton-feature value is already-active/persistence information versus genuine pre-onset information.** This is an apparently novel application based on a bounded literature search, not a “first ever” claim.

## Main experiment

1. Restrict the onset analysis to forecast issues where the previous 24 hours never crossed 10 pfu.
2. Train/evaluate a matched no-proton control.
3. Train/evaluate the same setup with sub-threshold proton min/max/mean history added.
4. Use chronological roles and the same tuning/threshold protocol.
5. Compare onset POD, FPR, FAR and TSS, with paired event-level uncertainty.
6. Keep the old full-proton vs no-proton occurrence result only as historical motivation / persistence diagnostic.

## Historical evidence retained

The repository still preserves:

- the fixed-prediction episode/onset evaluation benchmark;
- the independent audit and strict protected-boundary diagnostic;
- failed/underpowered Phase-II onset forecasters;
- missing-sensor DEGRADED/ABSTAIN engineering;
- prospective source-validation and custody work;
- negative controller/source experiments.

These remain evidence and engineering history, but they are **not** separate active IRIS projects.

Key historical references:

- `audit_20260915/AUDIT_REPORT.md`
- `audit_20260915/PAPER_RESULTS_20260920.md`
- `audit_20260915/NEGATIVE_RESULTS.md`
- `audit_20260915/REPOSITORY_TRUTH.md`
- `CURRENT_STATUS.md` (historical status as of 2026-09-11)

## Protected-data rule

The protected post-2025 outcome pool remains sealed. Do not use it for model selection, power checks, threshold tuning, feature selection, narrative rescue, or event counting. Any new confirmatory experiment must be frozen before accessing protected outcomes.

## One-sentence judge explanation

A recent 24-hour SEP model gains accuracy when historical proton flux is included, but proton flux also tells us when a radiation storm is already happening; this project tests whether proton measurements still add useful warning information **before** the 10 pfu storm threshold has been crossed.
