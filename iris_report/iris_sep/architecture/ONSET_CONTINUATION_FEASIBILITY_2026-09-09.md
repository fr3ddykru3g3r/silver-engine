# IRIS-SEP onset-versus-continuation feasibility gate

**Date:** 2026-09-09  
**Decision:** `DO_NOT_RUN_AS_AN_INDEPENDENT_CLAIM_STUDY_YET`

## Candidate question

A scientifically meaningful next question is whether apparently strong SEP forecasting performance is driven mainly by recognizing **continuation/persistence of an already elevated proton process**, rather than by issuing useful warnings before a genuinely **new onset**.

Candidate mechanism:

> A model can show useful event-overlap or continuation skill while adding little or no warning value for first onset beyond a simple past-only proton persistence/trend baseline.

This is a separate study concept. It is not a rescue or extension of the failed freshness-crossover hypothesis.

## Why it matters

The operational decisions differ:

- **onset:** no qualifying event is currently active; the system must warn before a new threshold crossing;
- **continuation/persistence:** a qualifying event is already in progress; the system is estimating whether elevated conditions persist.

NOAA/SWPC itself distinguishes proton warnings for expected **onset** from warnings for expected **persistence** of an event already in progress. Therefore the distinction is operationally meaningful, but the distinction itself is **not novel** and must not be presented as a discovery.

Reference: NOAA/SWPC GOES Proton Flux product description: https://www.swpc.noaa.gov/products/goes-proton-flux

## Novelty boundary

Close prior work already covers:

- daily GOES proton and soft-X-ray predictors;
- XGBoost/SVM SEP forecasting and proton-dominant feature importance;
- calibrated multimodal SEP probability models;
- short-horizon proton/X-ray onset prediction;
- 24-hour multimodal event-overlap forecasting.

A defensible contribution would therefore need to be the **quantitative decomposition** of apparent headline skill into onset versus continuation under a strict common-cohort, chronological, past-only evaluation—not merely the observation that onset and persistence are different tasks.

## Exposure audit

There is no defensible historical period currently available to relabel as an untouched score set:

1. Earlier IRIS-SEP development used all eligible historical development rows before the fixed monitor start.
2. The fixed development monitor from `2023-07-31` through `2025-09-10` is explicitly already inspected development evidence.
3. Freshness V1 separately exposed its 2011–2017 fit/calibration/threshold/score roles and outcomes.
4. The evidence registry forbids relabelling any dataset that has already informed model, threshold, architecture, or policy choices as fresh final evidence.

Therefore 2018–2022, or any other pre-monitor historical slice already used in development, cannot be manufactured into an “independent test” simply because this candidate target is new.

## Prospective candidate protection

Observations after the last exposed monitor date (`2025-09-10`) are **not authorized for outcome inspection in this feasibility step**.

No candidate identities, labels, event counts, or model scores from that protected period should be opened by the development side. If a future independent evaluation is attempted, an independent custodian must first define/hash the cohort and attest non-exposure against the complete exposure registry.

## Precision planning from exposed development evidence only

For planning—not inference—the previously exposed V3 missing-feed score evidence contained `21` positive opportunities over the fixed monitor span from 2023-07-31 through 2025-09-10, approximately `773` calendar days. That corresponds to an exposed-data planning rate of about:

`21 / 773 × 365 ≈ 9.9 positive opportunities per year`.

This rate must **not** be treated as the event count of any protected future cohort. It is only a rough planning input and may not transfer because solar activity, eligibility, source coverage, and episode definitions change.

The statistical audit gives two useful scale checks:

- even `36` independent perfect detections are required before the two-sided exact 95% sensitivity lower bound reaches about 90%;
- approximately `59` independent cases with zero additional misses are needed before a one-sided exact 95% upper bound on an extra-miss probability falls below 5% under a simple binomial model.

Real SEP episodes are temporally/physically dependent, so these are optimistic intuition rather than a complete power calculation.

At roughly ten positive opportunities per year, a short prospective window is therefore unlikely to support a narrow safety/detection-harm claim before the fair deadline.

## Required contract before this study could be authorized

If adequate independent data later exists, freeze all of the following **before** protected outcomes are exposed:

### Targets

A. `EVENT_OVERLAP_24H`: any qualifying event overlaps the next 24 hours.  
B. `NEW_ONSET_24H`: no event is active at issue time and a first qualifying crossing occurs in the next 24 hours.  
C. `CONTINUATION_24H`: a qualifying event is already active and remains/reappears according to a predeclared persistence/reset rule.

B and C have different target populations. Their raw AUPRC values must not be subtracted as though they were matched samples.

### Baselines and model family

Freeze one simple past-only proton persistence/trend baseline, one regularized logistic model, and the existing tree family. No architecture search is justified as the first question.

### Primary onset estimand

Incremental `NEW_ONSET_24H` TSS over the frozen persistence/trend baseline on the same eligible onset opportunities, with paired complete-episode/quiet-block resampling.

Secondary outcomes: Brier skill, event detection, false warnings per monitored day, achieved warning lead-time distribution, and the same model's overlap/continuation results.

### Operational causality

All eligibility and state classification must be computable from information available **strictly before issue time**. Do not use the issue-time/future proton oracle behavior identified in freshness V1.

### Evidence requirements

Persist per-issue IDs, labels, probabilities/actions, fitted model hashes/exports, feature order, thresholds, eligibility/exclusion reason ledger, and a shared resampling matrix. Repeated delays/seeds/days do not count as independent solar events.

### Falsification

The proposed mechanism is falsified if the prespecified onset model reproducibly improves onset skill beyond the practical margin fixed before scoring. If uncertainty is too wide, the result is **unsupported**, not equivalent.

An engineering improvement claim must be separate: e.g. at least 20% fewer false warnings with no observed additional misses and explicit uncertainty on miss harm. A zero observed miss difference is not a safety guarantee.

## Current decision

Do **not** open or score the protected post-2025 candidate period to chase a new positive result.

Do **not** run onset-versus-continuation as an “independent” IRIS winning study until:

1. a full exposure ledger is complete;
2. an independent custodian can identify an admissible cohort without exposing its outcomes;
3. development-only power/precision simulation shows that the cohort can answer the frozen estimand with useful precision;
4. the novelty claim is kept narrow and literature-bounded;
5. the new contract is frozen before outcome access.

If these conditions cannot be met, the scientifically correct submission path is to preserve the negative freshness result and present its limitations honestly rather than manufacture a second underpowered discovery attempt.
