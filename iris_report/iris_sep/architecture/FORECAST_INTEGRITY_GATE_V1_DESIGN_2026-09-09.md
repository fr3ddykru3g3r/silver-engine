# Forecast Integrity Gate V1 — scientific design

**Status:** preregistered design; protected post-2025 outcomes remain unopened.  
**Study:** `IRIS_FORECAST_INTEGRITY_GATE_V1`  
**Primary contract:** `config/forecast_integrity_gate_v1_preregistration_2026-09-09.json`

## One-sentence question

Can a solar-particle warning system decide whether its **evidence is trustworthy enough to support a forecast**, independently of how confident the forecasting model itself appears?

## Why this is a stronger project than another SEP classifier

The crowded question is: **which model predicts SEP events best?**

Close prior work already includes:

- Ali et al. (2024), *Predicting Solar Proton Events of Solar Cycles 22–24 Using GOES Proton and Soft-X-Ray Flux Features*, ApJS 270:15, DOI `10.3847/1538-4365/ad0a6c`: proton/XRS daily statistics with SVM and XGBoost across multiple solar cycles.
- Yu et al. (2026), *Realtime forecasting of solar energetic particle event and proton flux using multi-source solar observations and multi-task deep learning*, Scientific Reports, DOI `10.1038/s41598-026-66110-2`: 24-hour multi-source SEP forecasting with magnetic, flare, CME, XRS and historical-proton evidence.
- Barnes & Barnes (2021), *Controlled Abstention Neural Networks for Identifying Skillful Predictions for Classification Problems*, JAMES, DOI `10.1029/2021MS002573`: confidence/uncertainty-based abstention for Earth-system classification.

Therefore IRIS does **not** claim novelty for proton features, XRS features, XGBoost, multimodal forecasting, or the general idea of abstention.

The candidate contribution is different:

> **Separate model confidence from evidence integrity.**
>
> A forecast can be numerically confident while its required data are stale, missing, non-causal, unauthenticated, or not schema-equivalent to the data interface on which the frozen model was developed. The proposed gate evaluates those source facts directly and can refuse to expose a forecast even when the model probability looks certain.

This distinction is motivated by the project's own live preflight: the frozen V3 model could not reproduce three required SHARP quantities (`CMASKL`, `MEANGBL`, `USFLUXL`; 18 frozen feature positions). Correct behavior was to emit **no forecast probability** rather than silently substitute values.

## Core scientific objects

For each required source at an issue time, the gate records:

1. `available` — was the required source present?
2. `authenticated` — did it come from a registered source family and pass provenance checks?
3. `schema_equivalent` — does the source expose the same scientific quantity/interface required by the frozen path?
4. `causal` — was the value available by issue time without future information or retrospective reconstruction?
5. `age_seconds / max_age_seconds` — how much of the frozen freshness budget remains?
6. `quality_fraction` — declared issue-time valid-data quality in `[0,1]`.

Any failure of 1–4, staleness at or beyond the maximum age, or zero declared quality is a **hard evidence failure** and maps source integrity to zero.

For otherwise admissible evidence, the analysis score is

`source_integrity = min(remaining_freshness_fraction, quality_fraction)`.

For a forecast path requiring several sources,

`evidence_integrity = min(source_integrity across required sources)`.

The minimum is intentionally a **weakest-link rule**. It prevents outcome-fitted source weights from allowing strong sources to hide one invalid required source.

The continuous score is an analysis/ranking variable. It does **not** override the discrete causal/schema gate.

## New measurable phenomenon: integrity–confidence discordance

Binary model confidence is defined only for comparison as

`forecast_confidence = 2 * |p - 0.5|`.

The study then measures

`integrity_confidence_discordance = forecast_confidence - evidence_integrity`.

Positive values mean the model is more confident than the source evidence is strong.

A preregistered descriptive flag is

- confidence `>= 0.8`, and
- evidence integrity `<= 0.2`.

These are called **unsafe-confidence cases** for analysis. The phrase does not mean physical harm occurred; it means a high-confidence model output was weakly supported by the required issue-time evidence.

## Primary experiment

### Candidate selector

`EVIDENCE_INTEGRITY`

Uses only issue-time source metadata. It cannot read labels or future measurements and does not use the model probability.

### Comparators

1. `FORECAST_CONFIDENCE` — `2*|p-0.5|`.
2. `AVAILABILITY_ONLY` — simple preregistered source-availability state without provenance/freshness detail.
3. `SEEDED_RANDOM` — coverage-matched negative control.

### Coverage protocol

Target coverage levels:

`50%, 60%, 70%, 80%, 90%, 100%`.

For every model/scenario/selector:

1. derive the selector cutoff on the **threshold role only**, without labels;
2. freeze the cutoff;
3. apply it unchanged to the **score role**;
4. apply the already-frozen model decision threshold to retained score rows;
5. report realized coverage, TSS, FAR, POD and Brier score.

The score cohort is never used to choose the selector cutoff.

## Degradation families

The study reuses already-exposed development evidence rather than opening protected outcomes:

- 24/72/168-hour XRS outage;
- 24/72/168-hour proton outage;
- joint XRS+proton outage;
- event-bearing terminal outages where immutable prediction receipts exist;
- live/schema-equivalence failure receipts;
- freshness/staleness degradation where source timestamps support it.

A simulated outage must change eligibility using only information still available under that outage. The freshness V1 oracle-eligibility mistake is explicitly forbidden.

## Model-agnostic generalization

A central claim requires the same direction across at least **two distinct frozen model outputs** and at least **two degradation families**. The preferred evidence package contains at least three model families if immutable prediction-level outputs are recoverable.

No new forecaster hyperparameter search is part of this study.

## Statistical correction to the failed freshness branch

Freshness V1 paired methods within a contrast but redrew samples across delays. V1 therefore could not support simultaneous crossover evidence.

Forecast Integrity Gate V1 instead creates **one shared cluster-bootstrap draw table** from complete SEP episodes / preregistered quiet blocks and reuses that exact table for:

- every selector;
- every coverage;
- every scenario;
- every model;
- every pairwise contrast.

The draw table is SHA-256 bound into the result.

Primary strong-claim uncertainty uses 10,000 paired replicates. Small positive counts and interval widths must be shown explicitly.

## Promotion / falsification rule

At coverage `>= 80%`, evidence-integrity selection is promising only if it has lower FAR and/or lower Brier score than confidence-only selection **without losing more than 0.05 absolute POD**, with the same direction across at least two degradation families and two frozen model outputs.

A statistically established advantage additionally requires the preregistered paired 95% interval for the relevant primary contrast to exclude zero.

The hypothesis is falsified or weakened if:

- confidence-only abstention matches or beats evidence integrity;
- apparent FAR improvement requires >0.05 POD loss;
- the direction reverses across models;
- outcome-fitted source weights are needed;
- result-level evidence cannot reproduce every primary metric.

## Reproducibility package required before judge-facing claims

No summary-only artifact is sufficient. The study must preserve:

- `per_issue_predictions.csv`;
- `per_issue_source_integrity.csv`;
- a complete attrition/exclusion ledger;
- shared bootstrap draw receipt or exact reproducible ordered-unit manifest + seed;
- model, source, cohort, configuration and environment hashes;
- frozen selector cutoffs;
- every negative/inconclusive cell;
- an independent recomputation audit that does not import the result runner.

## What would make this scientifically interesting even if the primary gate fails?

A null result is informative if source-integrity gating does **not** outperform confidence-based abstention. It would show that explicit provenance/freshness metadata adds little beyond the model's confidence under the tested source failures.

A positive result would support a sharper lesson:

> **Prediction confidence is not evidence validity.**

That lesson is broader than one SEP architecture and is testable, falsifiable, operationally relevant, and directly connected to the project's strongest existing live-source failure.
