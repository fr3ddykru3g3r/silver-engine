# Active IRIS project — operational reproducibility of SEP forecasting

**Date frozen:** 21 September 2026

## Research question

**Can a retrospective 24-hour solar energetic particle (SEP) forecasting model be reproduced using only predictor values that were genuinely available before each forecast issue time?**

A secondary question is: **how much forecast coverage and skill remain after every predictor that lacks a verified near-real-time equivalent is removed or causes an explicit abstention?**

## Why this is the active project

Earlier candidate projects were rejected after novelty checks:

- repeated-window / onset-versus-persistence evaluation: too much prior work already distinguishes event onset, persistence and event-level verification;
- proton-history attribution: preceding proton flux, persistence baselines and sub-threshold proton rises are already used in published SEP forecasting;
- sensor forecastability horizons: Ji et al. (2025) already vary lag windows before SEP onset and study performance/feature importance versus lag;
- physics-guided counterfactual testing: Patil et al. (2026) already apply physics-guided counterfactual explanations to SEP prediction.

The present project does **not** claim that data latency or real-time availability are newly discovered problems. They are known operational concerns. The narrower novelty target is the **feature-by-feature causal availability audit and quantitative retrospective-to-operational reproducibility gap for a modern multi-source SEP machine-learning interface**.

## Core idea at IB Physics level

A forecast issued at time t may only use information that existed before t. A value that was measured earlier but processed, corrected, catalogued or published later is not a valid real-time predictor for that forecast.

For each predictor family we therefore ask:

1. What physical quantity does it measure?
2. What source generated it?
3. Was a near-real-time version available before forecast issue?
4. Does the near-real-time version have the same units and definition as the retrospective value?
5. Was it later corrected, reprocessed or derived using information unavailable at issue time?

A predictor passes only if all five questions have evidence-backed answers.

## Experimental phases

### Phase A — feature-interface audit

Start from the frozen retrospective predictor interface used by the project. For every model input, record:

- feature name and physical meaning;
- source instrument/catalog;
- measurement timestamp;
- first-seen/retrieval timestamp where available;
- retrospective processing steps;
- candidate near-real-time source;
- schema/unit equivalence;
- operational status: `VERIFIED`, `UNVERIFIED_LATENCY`, `SCHEMA_MISMATCH`, `RETROSPECTIVE_ONLY`, or `NO_EQUIVALENT`.

The main descriptive quantity is the **operationally reproducible feature fraction**:

`verified predictor columns / total predictor columns`.

This is a bookkeeping ratio, not a new mathematical theorem.

### Phase B — forecast reproducibility

For issue times where a causal input snapshot can be reconstructed, replay forecasting under two conditions:

1. **Retrospective interface:** the archived feature vector used historically.
2. **Operational interface:** only values proven to have existed before issue time; otherwise the forecast must abstain or use a separately frozen reduced model.

Report:

- fraction of issue times with a fully reproducible input vector;
- number and identity of feature families responsible for failure;
- probability difference where identical-schema replay is possible;
- TSS, POD, FPR and FAR for a separately frozen causal-only model if enough historical causal examples exist.

Do not fill a structurally unavailable feature with zero, a similarly named field, or a later definitive value and call that the same model.

### Phase C — source-product equivalence test

For source families with both definitive and near-real-time products, compare matched timestamps directly. For each common physical variable calculate simple absolute/relative differences and the fraction of timestamps whose values disagree beyond a preregistered tolerance.

This phase can produce a result even if SEP events are rare because it tests sensor/product equivalence rather than event occurrence.

## Existing evidence that motivates the study

The independent repository audit already establishes that:

- the retrospective joint predictor interface contains 259 predictor-side columns;
- historical harmonization/imputation does not prove those values existed at issue time;
- the existing prospective contract explicitly blocks the 259-column interface as a set until field-level causal availability is established;
- current GOES proton and X-ray streams exist, but their historical issue-time equivalence and publication latency are not automatically proven;
- source coverage alone is not operational equivalence.

These are motivation and partial audit evidence, not the final result of the new study.

## Hypothesis

The null hypothesis is that enforcing issue-time availability makes no material difference to the usable predictor interface or forecast output.

The alternative hypothesis is that at least one retrospective predictor family cannot be reproduced causally and that enforcing operational availability materially reduces model coverage and/or changes forecast probabilities or skill.

## What would make the project fail

The project fails as a novelty project if a prior SEP paper is found that already performs a comparable feature-by-feature issue-time availability audit and quantitative retrospective-versus-operational replay for a modern multi-source model.

It fails experimentally if the feature lineage cannot be reconstructed with enough certainty to classify the predictor interface. In that case the correct result is `UNRESOLVED`, not an invented operational score.

## Claims we will not make

- that the original model is invalid;
- that every retrospective SEP model leaks future information;
- that a source is unavailable merely because its current API differs;
- that missing-data robustness is the same as causal availability;
- that a present-day source proves historical first-seen availability;
- that a reduced operational model is the same model as the retrospective 259-feature model;
- that failure to reproduce an interface proves poor forecast skill.

## Plain-language judge explanation

> A forecast can only use measurements that actually existed when the forecast was made. I am auditing a modern solar-radiation forecast feature by feature to see whether its retrospective inputs can really be reconstructed from near-real-time data, then measuring what changes when I enforce that rule.

## Working title

**Can a Retrospective Solar-Radiation Forecast Be Reproduced in Real Time? A Causal Availability Audit of Multi-Source SEP Predictors**
