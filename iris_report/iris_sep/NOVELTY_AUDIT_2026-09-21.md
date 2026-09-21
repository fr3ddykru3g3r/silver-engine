# Novelty audit — active proton-signal project

Status: bounded literature audit, 21 September 2026. This file defines what may and may not be called novel.

## Closest prior work

### Sadykov et al. (2021)

Daily solar-proton-event prediction using preceding proton flux, soft X-ray flux and solar information. They report preceding proton flux as a highly valuable predictor. Therefore **using preceding proton flux is not novel**.

### Ali et al. (2024), ApJS 270(1):15

Uses daily GOES proton and soft-X-ray statistics with SVM and XGBoost across Solar Cycles 22–24 and compares against a persistence forecast. Therefore **XGBoost for SPE prediction, proton-flux features, cross-cycle testing and a persistence baseline are not novel**.

### Kahler (2015), Space Weather

Dynamic SEP probability forecasting explicitly uses sub-threshold increases in >10 MeV proton intensity to update the probability that the 10 pfu threshold will eventually be reached. Therefore **the idea that sub-threshold proton increases can carry precursor information is not novel**.

### Yu et al. (2026), SEPNET / SEPNET-PRISM

Modern 24-hour-ahead SEP forecasting uses multi-source observations and historical >10 MeV proton flux, and reports improved operational performance when proton-flux predictors are included. Therefore **showing that proton features improve ordinary 24-hour occurrence forecasting is not novel**.

## What remains potentially original

The active contribution is not a new forecasting algorithm or a new physical precursor. It is an **attribution experiment**:

> For a matched modern 24-hour SEP forecasting setup, how much of the apparent benefit of historical proton-flux features survives when (a) the forecast target is genuine new onset and (b) the previous 24 hours are required to remain below the 10 pfu operational storm threshold?

The experiment directly compares a no-proton control with the same model family plus sub-threshold proton history on an onset-eligible cohort. Historical full-proton/occurrence results are used only to quantify the larger apparent benefit when already-active/persistence windows are allowed.

A bounded search did not identify a prior study performing this exact controlled attribution on the same kind of daily ML forecast: same model family, matched non-proton predictors, explicit removal of already-active >=10 pfu periods, onset-only target, and direct measurement of how the proton-feature gain changes.

This supports wording such as:

- “a controlled attribution not identified in our literature search”;
- “we test whether the reported value of historical proton flux persists for genuine pre-onset forecasting”;
- “we separate persistence information from pre-onset information in a matched feature-ablation.”

It does **not** support:

- “first ever”;
- “we discovered proton precursors”;
- “we discovered persistence bias”;
- “we discovered onset forecasting”;
- “we invented a new ML model.”

## Novelty risk

Novelty confidence is **moderate, not absolute**. Dynamic pre-threshold proton forecasting is established, and persistence baselines are established. The originality depends on the controlled attribution design and its empirical result. If a paper is found that already performs the same matched active-state-versus-pre-onset proton-feature ablation for daily SEP prediction, the project must be reframed as replication/extension rather than original methodology.

## Why this can still be a strong IRIS project

For a student project, originality does not require inventing a new ML algorithm. A rigorous experiment can be novel because it isolates a confounding physical signal that changes how an existing model result should be interpreted. The experiment is falsifiable, uses an operational physical threshold, has a clear control, and can produce a useful result whether sub-threshold proton history helps or not.
