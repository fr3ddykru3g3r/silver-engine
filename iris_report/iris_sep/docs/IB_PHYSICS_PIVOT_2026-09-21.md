# IRIS-SEP IB Physics pivot — 21 September 2026

## New research question

**Does recent >10 MeV proton flux genuinely help predict the onset of a new >=10 pfu solar energetic-particle storm within 24 hours, or does it mainly help because the storm is already in progress?**

This replaces the weaker novelty story that repeated 24-hour windows themselves were newly discovered. Prior SEP work already distinguishes onset from persistence and the 2026 SEPNET-PRISM work explicitly uses historical proton flux as a predictor. The new contribution is a controlled feature-ablation experiment that separates *active-storm information* from *pre-onset, sub-threshold proton information*.

## Why this is IB Physics level

The physics idea is simple. NOAA defines an S1 solar radiation storm when >10 MeV proton flux reaches 10 pfu. If the proton flux was already above 10 pfu during the previous day, then using it to predict another positive day is partly a persistence problem. A genuine warning problem asks whether measurements taken while the storm is not yet active contain useful precursor information.

No advanced mathematics is required. The project uses:

- proton flux in pfu;
- a 10 pfu physical threshold;
- 24-hour past and future windows;
- four controlled feature sets;
- true/false positives and negatives;
- True Skill Statistic (TSS = POD - FPR);
- simple paired bootstrap intervals as an uncertainty check.

## Controlled experiment

Use the same public SEP-PRISM table, chronological fit/threshold/score periods, model family and hyperparameters for every condition. Only proton information changes.

### Model A — NO PROTON

Remove all four proton-history variables:

- ProtonFlux_label
- ProtonFlux_min
- ProtonFlux_max
- ProtonFlux_avg

This is the control.

### Model B — ACTIVE STATE ONLY

Do not give the model the actual proton values. Give it only one binary variable:

`past_proton_active = 1 if ProtonFlux_max >= 10 pfu, otherwise 0`.

This tests how much skill comes simply from knowing a radiation storm is already active.

### Model C — SUB-THRESHOLD PROTON PRECURSOR

Use past proton min/max/mean only when `ProtonFlux_max < 10 pfu`. For rows where the previous 24 hours already crossed 10 pfu, mask those proton-value features. Do not supply the active-state flag.

This asks whether proton measurements *before* the operational threshold crossing contain useful information about a new storm.

### Model D — FULL PROTON HISTORY

Use the original four proton-history variables unchanged.

This matches the broad information available to the current joint model and serves as the full-input comparison.

## Evaluation

Every model is evaluated in two clearly different tasks.

1. **Occurrence:** will the next 24 hours contain >=10 pfu proton flux?
2. **New onset:** will a new >=10 pfu SEP episode begin in the next 24 hours when it was not already active?

The central comparison is not whether XGBoost itself is new. XGBoost is only the measuring tool. The scientific result is the change in usefulness of proton information when the physical question changes from persistence/occurrence to new onset.

## Existing historical clue, not the final new experiment

The already-frozen audit gives a strong motivation. In the strict historical cohort, the XGBoost model with proton history has TSS 0.7264 for ordinary occurrence while the otherwise matched no-proton feature set has TSS 0.5090: an apparent proton-input advantage of about +0.2174 TSS.

For genuine new onset, the corresponding values are 0.4372 and 0.4775, giving an advantage of about -0.0403. A paired physical-unit bootstrap on the frozen predictions gives:

- occurrence proton-model advantage: median about +0.216, 95% interval about [+0.117, +0.317];
- onset proton-model advantage: median about -0.039, 95% interval about [-0.166, +0.084];
- change in the proton advantage from occurrence to onset: median about -0.255, 95% interval about [-0.343, -0.173].

These are post-hoc historical diagnostics and must not be presented as a newly preregistered confirmation. They motivate the four-condition ablation above.

## What would count as the strongest result

The most informative pattern would be:

- FULL PROTON strongly improves occurrence;
- ACTIVE STATE ONLY explains much of that occurrence gain;
- SUB-THRESHOLD PROTON produces little or no onset gain.

That would support the interpretation that historical proton flux is mainly useful for recognizing/persisting an already-active radiation storm rather than forecasting a genuinely new onset.

A different result would also be scientifically valuable. If SUB-THRESHOLD PROTON improves onset skill, then the data contain a real pre-onset proton precursor signal. The experiment is therefore falsifiable either way.

## Claims we will not make

- We did not invent XGBoost.
- We did not discover that SEP storms can persist for multiple days.
- We did not discover the general distinction between onset and persistence.
- We will not claim that historical proton flux is useless without testing the sub-threshold condition.
- We will not claim causation from a retrospective machine-learning ablation.
- We will not call historical development-exposed results prospective validation.

## Proposed paper title

**Does Recent Proton Flux Predict a New Solar Radiation Storm, or Just One Already in Progress?**

Alternative formal title:

**Separating Precursor and Persistence Information in 24-Hour Solar Energetic Particle Forecasting**

## One-sentence judge explanation

> A recent SEP model reports that adding historical proton flux improves forecasting, but proton flux is also the quantity that defines whether a radiation storm is already happening; I test whether that sensor still adds useful information when the task is specifically to predict a new storm before the 10 pfu threshold has been crossed.
