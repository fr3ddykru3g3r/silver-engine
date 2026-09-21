# IRIS-SEP active project — 21 September 2026

## Active research question

**When no >=10 pfu solar radiation storm is already active, do sub-threshold >10 MeV proton measurements from the previous 24 hours improve prediction of a new >=10 pfu SEP onset in the next 24 hours?**

This is the only active competition research question. Older missing-sensor, general occurrence-vs-onset, controller and alternate-onset-model branches are preserved as historical evidence but are not the current IRIS narrative.

## Why this question exists

Prior work already establishes several facts that must not be claimed as novel:

- SEP onset and persistence are different operational questions.
- Previous proton flux can be useful in SEP forecasting.
- XGBoost has already been used for SEP prediction.
- A simple persistence forecast has already been compared with ML models.
- Sub-threshold proton increases have already been used in dynamic SEP probability forecasts.

Therefore the project is **not** claiming discovery of persistence, proton precursors, XGBoost, or the 10 pfu threshold.

The narrower contribution is a controlled attribution question: modern 24-hour window models can appear to gain substantial skill from recent proton flux, but proton flux also directly reveals whether an event is already in progress. We test how much of the proton-feature benefit survives when the task is restricted to genuine new onsets and the previous 24 hours are constrained to remain below 10 pfu.

## IB Physics-level experiment

Use the same public SEP-PRISM source data and the same chronological train / threshold / score roles for all conditions. Keep the model family and tuning protocol fixed. Change only whether sub-threshold proton-history measurements are supplied.

### Condition A — no-proton control

Predictors exclude explicit historical >10 MeV proton-flux statistics.

### Condition B — sub-threshold proton condition

Use the same non-proton predictors plus previous-24-hour proton-flux statistics, but evaluate only forecast issues for which the previous 24 hours never crossed 10 pfu. This prevents an already-active operational storm from supplying the answer indirectly.

### Historical persistence diagnostic

Separately retain the simple `past_proton_active` diagnostic and the existing frozen joint-vs-no-proton occurrence comparison to show how much skill can be associated with already-active conditions. This diagnostic motivates the experiment; it is not the final confirmatory result.

## Primary outcome

For genuine onset decisions, compare the no-proton control with the sub-threshold-proton condition using:

- probability of detection (POD),
- false-positive rate (FPR),
- True Skill Statistic (TSS = POD - FPR),
- false-alarm ratio (FAR),
- paired physical-event bootstrap intervals.

The main quantity is the change in onset TSS when sub-threshold proton history is added.

## What would support each interpretation

If sub-threshold proton history improves onset TSS reliably, then recent proton measurements contain useful pre-onset information even before the operational threshold is crossed.

If it does not improve onset TSS while full proton history strongly improves ordinary occurrence forecasting, then much of the apparent proton-feature value is better interpreted as persistence / already-active-event information rather than genuine new-onset warning information.

Either outcome is scientifically useful.

## Existing historical clue — not the final experiment

In the strict frozen historical audit, the proton-inclusive XGBoost has occurrence TSS about 0.726 versus about 0.509 for the no-proton XGBoost, an apparent advantage of about +0.217. On genuine onset, the corresponding TSS values are about 0.437 and 0.477, so the apparent advantage is about -0.040 and its paired interval crosses zero. This is post-hoc historical motivation, not a new confirmatory result.

## Novelty boundary

The defensible novelty target is:

> **A controlled, same-benchmark attribution of the apparent value of historical proton-flux features into already-active/persistence information versus genuine pre-onset information for 24-hour >=10 pfu SEP forecasting.**

This should be described as **apparently novel based on the bounded literature search**, not “first ever.” If a prior paper is found that performs the same controlled active-state-versus-pre-onset proton-feature ablation on comparable SEP forecasts, the novelty wording must be downgraded to replication/extension.

## Claims prohibited for the current project

Do not claim that we invented XGBoost, discovered SEP persistence, discovered that proton flux can precede SEP events, discovered sub-threshold proton increases, built a state-of-the-art operational forecaster, or prospectively validated the model.

## Working title

**How Much of Proton-Flux Forecast Skill Comes From a Storm Already Being Active?**

Formal alternative:

**Separating Persistence and Pre-Onset Information in 24-Hour Solar Energetic Particle Forecasting**
