# Episode benchmark V1 — post-result audit correction boundary

**Date:** 2026-09-09  
**Status:** `FROZEN_POST_RESULT_ESTIMATOR_AND_EVIDENCE_CORRECTION_ONLY`

## Why this correction exists

The first successful end-to-end development artifact (`GitHub Actions run 34369442821`, head `b4fec920d7a3f58841cb14fb2c07c2e6091b17d2`) preserved correct per-row probabilities, fold-specific frozen thresholds and binary alerts, but the independent artifact audit found two implementation defects after results were available:

1. The expanding chronological OOF point-summary code used the first row's threshold when recomputing confusion metrics across the concatenated 2014–2017 prediction table. OOF rows legitimately carry different thresholds selected on each fold's immediately preceding threshold year. The authoritative confusion metrics must therefore use each persisted row's already-frozen `binary_alert` (equivalently, its own persisted probability/threshold pair), not one global threshold.
2. Physical-episode bootstrap uncertainty omitted positive occurrence windows that could not be uniquely mapped to a single physical episode. The corresponding full standard point table included those windows. The old bootstrap therefore estimated a narrower mapped-episode estimand than the full point table. It must not be presented as uncertainty for the full-window point estimand.
3. `summary.json` was hashed and then rewritten to add the hash-manifest digest, leaving the stored summary SHA stale. This is an evidence-packaging defect.

## Non-negotiable correction boundary

The following are **frozen and may not change** in response to these findings:

- source files and source definitions;
- 2011–2017 candidate issue schedule;
- physical episode construction and gap rule;
- model features;
- XGBoost and elastic-net specifications;
- random seeds;
- expanding chronological folds;
- calibration procedure for strict 2017;
- fold-specific threshold-selection procedure;
- saved per-row probabilities;
- saved per-row thresholds;
- saved per-row binary alerts;
- protected post-2025 cohort status;
- underpowering threshold;
- hypotheses and claim boundary.

No model refitting or threshold reselection is justified by this correction. A rerun may recreate the same predictions from the frozen pipeline, but the correction itself concerns only evaluation and evidence accounting.

## Authoritative point estimands after correction

### Full-window descriptive tables

The original four views remain and use all rows appropriate to each definition:

- `WINDOW_OCCURRENCE_STANDARD`
- `EPISODE_NORMALIZED_OCCURRENCE`
- `NEW_ONSET_CAUSAL`
- `EPISODE_NORMALIZED_ONSET`

Confusion metrics use the persisted row-specific `binary_alert`. Probability metrics use the persisted probabilities and weights.

These full-window point estimates are descriptive exposed-development results. They do **not** receive physical-episode bootstrap intervals when the positive population contains windows that cannot be assigned to exactly one episode.

### Mapped-episode inferential sensitivity

A separate table is required for physical-unit uncertainty. Its standard-occurrence positive cohort contains only positive windows uniquely mapped to one physical SEP episode; all standard negative windows remain. The four mapped views are:

- `MAPPED_STANDARD_OCCURRENCE`
- `MAPPED_EPISODE_NORMALIZED_OCCURRENCE`
- `MAPPED_NEW_ONSET_CAUSAL`
- `MAPPED_EPISODE_NORMALIZED_ONSET`

All bootstrap intervals, cross-view TSS shifts and bootstrap rank-reversal fractions must refer explicitly to this mapped physical-unit sensitivity, not to the full-window table.

Ambiguous/unmapped positive windows remain visible in the attrition/evidence tables and the full-window point view; they are not assigned an invented physical episode merely to make bootstrapping convenient.

## Shared bootstrap rule

Reuse one 10,000-replicate stratified draw tensor for every model and mapped evaluation view:

- positive unit = uniquely mapped physical SEP episode;
- negative unit = frozen seven-day quiet block;
- identical replicate indices across models and views;
- a resampled physical episode with no eligible onset row contributes zero onset-positive mass in that view rather than being replaced by a different event.

This preserves the paired comparison while keeping the physical episode as the independent positive unit.

## Evidence integrity correction

Finalize all result/summary files first. Then create `evidence_hashes.json` from those final immutable files. Do not rewrite any hashed file afterward. The artifact ZIP digest remains an additional outer receipt.

The independent verifier must:

- recompute row alerts as `probability >= row threshold` and match `binary_alert`;
- recompute point confusion metrics from persisted rows without assuming one global OOF threshold;
- verify mapped episode weights sum to one;
- verify attrition reconciliation;
- verify shared draw dimensions and ranges;
- verify every listed evidence hash.

## Interpretation rule

The first successful artifact is preserved as an immutable debugging/evidence checkpoint, but its OOF summary and bootstrap uncertainty are superseded for inference by the corrected artifact. Any scientific conclusion that survives must be stated using the corrected point estimands and mapped-episode uncertainty.

Because this correction was specified after seeing model results, it cannot create a new preregistered claim. It can only make the already-frozen experiment's estimator and evidence package conform to the intended design.