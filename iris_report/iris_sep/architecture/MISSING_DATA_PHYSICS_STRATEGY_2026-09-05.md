# IRIS-SEP missing-data + physics strategy — 2026-09-05

## Decision

Do **not** make a full magnetohydrodynamic (MHD) simulator the default imputer.
Do add a physics-reconstruction research track immediately, with a strict rule:
physics must demonstrate better downstream SEP forecasting under deliberately
masked forecast-time data than simpler causal recovery methods before it can
enter the frozen final model.

This is a design correction, not a retreat from physics. A simulated state is a
model hypothesis. It is not recovered sensor truth. IRIS must never silently
replace a missing observation with a simulation and present the result as if it
were observed.

The active preregistration is `config/missingness_recovery_contract_v1.json`.
The source primitives are `src/iris_sep/missingness_recovery.py`.

## Why this is the higher-value research question

Current development has two distinct missing-data problems:

1. **Historical partial features.** Some magnetic/source-era inputs are absent
   or sparse inside development data, which reduces usable support and can make
   model behavior era-dependent.
2. **Forecast-time feed loss.** A real-time source may become stale, disappear,
   change revision, or fail exactly when an operator needs a forecast.

A full MHD run cannot automatically solve either problem. It still requires
boundary/initial conditions, which may themselves be uncertain or missing, and
its output is a physically modeled state rather than the missing measurement.
The correct experiment is therefore not `missing -> MHD -> assume truth`; it is
`known truth -> deliberately hide data -> reconstruct causally -> compare with
truth -> measure downstream forecast degradation`.

## Competition review and novelty correction

The project must not claim novelty from simply predicting future solar magnetic
maps or combining them with a flare/proton predictor. An ISEF 2020 Physics and
Astronomy finalist, *Modelling Space Weather: A New Deep Learning-Based
Approach*, already paired a 24-hour solar-event predictor with a ConvLSTM that
predicted future magnetograms/dopplergrams. Another 2020 project, SWIFT, used
machine learning on HMI vector magnetograms for space-weather forecasting.

Recent top computational-physics projects instead show a more useful pattern:
physics is tied to a narrow, measurable bottleneck and validated quantitatively.
ISEF 2026 Physics first-award work included a hybrid computational framework
using Parker Solar Probe measurements for solar-Alfven/tokamak analysis, and the
2026 overall top award was a simulation program with a sharply defined
mathematical/physical question. In 2025, a Physics first-award project combined
a low-cost physical chamber with a physics-informed neural network for
turbulence. The lesson is not `use PINNs`; it is `make physics earn a measured
advantage on a controlled problem`.

IRIS's potentially distinctive question is therefore:

> When forecast-time solar observations are incomplete, can a causally
> generated physics-constrained reconstruction preserve calibrated NEW-SEP
> threshold-crossing skill better than simple causal imputation or mask-aware
> abstention, while an explicit validity boundary prevents reconstructed data
> from masquerading as observation?

That question is falsifiable and directly connected to operator use.

## Engineering architecture

### Layer A — observation truth and masks

Every modality entering the experiment has two independent objects:

- value array;
- observed/missing mask.

The mask is never destroyed by an imputer. Natural missingness is preserved.
Synthetic missingness experiments create a second held-out mask on values that
are known in the original train-only data so recovery accuracy can be measured.

### Layer B — recovery strategies

Run in increasing complexity. A more complex arm survives only if it adds value.

1. **Mask-aware / no reconstruction.** The model sees availability explicitly;
   critical loss may force abstention.
2. **Causal forward fill** where physically/temporally meaningful.
3. **Train-fit median/simple statistical recovery.** No validation/test fitting.
4. **Published-style causal KNN** where a faithful implementation is possible.
5. **Reduced physics reconstruction.** Prefer the simplest defensible physics
   model for the missing modality.
6. **Physics assimilation / MHD-derived state.** Only if Step 5 leaves a clear
   residual bottleneck and the boundary conditions, runtime, versioning and
   uncertainty are reproducible.

The final model is not entitled to use arm 5 or 6. These are candidates.

### Layer C — reconstruction provenance

Every reconstructed modality must carry:

- modality;
- method ID and class;
- train-only fit role;
- latest real observation time;
- reconstruction generation time;
- explicit future-information flag;
- declared physical constraints;
- normalized reconstruction uncertainty;
- exact artifact SHA-256.

`missingness_recovery.py` implements the source-only audit and exact
reconstruction hashing. A reconstruction that uses future data, is generated
after issue time, is not on an allowed method list, or exceeds its frozen
uncertainty boundary cannot enter a forecast experiment.

### Layer D — reconstruction benchmark

Evaluate only deliberately hidden cells. Never dilute reconstruction error with
unchanged observed cells.

Per-feature/state metrics:

- held-out MAE;
- held-out RMSE;
- bias;
- maximum absolute error;
- uncertainty interval coverage and width.

These are diagnostic metrics, not the project endpoint.

### Layer E — downstream SEP benchmark

For each missingness scenario, compare predictions on identical issue identities.
Primary decision metrics remain:

- TSS;
- FAR at matched detection;
- Brier score;
- ECE;
- retained forecast coverage;
- abstention rate;
- source-era stability.

The physics arm survives only if it reduces forecast degradation relative to the
best surviving nonphysics recovery strategy without material calibration harm.
No locked-test selection is permitted.

## Missingness scenarios

Do not tune scenarios after results. Freeze cadence-specific durations only when
the verified source-latency manifest exists.

Required scenario families:

1. **Natural missingness:** historical masks exactly as observed.
2. **Random held-out values:** deterministic seeds, useful for recovery sanity.
3. **Contiguous source outages:** block gaps matched to the source cadence.
4. **Complete modality loss:** magnetic, eruption, particle-context, and later
   XRS dropout as individually appropriate.
5. **Source-era transition:** repeat the same recovery strategy across eras to
   test whether it is merely exploiting one instrument regime.
6. **Compound fault:** missing + stale/revision/OOD conditions remain under the
   existing validity envelope.

## Physics ladder

### P0 — no simulator

Mask-aware model and causal statistical baselines. This is mandatory because a
physics model that cannot beat these should be discarded.

### P1 — reduced magnetic physics

For magnetic-map gaps, first test a lightweight, reproducible physical prior,
for example a surface-flux-transport / potential-field style reconstruction or
another source-appropriate reduced model. Exact choice is deferred until the
verified training-only data and source geometry are available; no physics term
will be invented solely to make the project sound sophisticated.

The goal is not a visually realistic Sun. The goal is to preserve the specific
forecast-time quantities that matter for NEW-SEP prediction.

### P2 — physics assimilation

Assimilate available observations into the reduced physical state and propagate
forward causally through the gap. Reconstruction uncertainty must increase with
forecast distance or be otherwise empirically calibrated from held-out gaps.

### P3 — full MHD / external physics model

Only enter this stage if P1/P2 leave a measured residual problem. A full MHD
solver is a candidate feature/reconstruction provider, not a replacement for
observations. Minimum admission requirements:

- forecast-time boundary conditions are available;
- implementation/version is reproducible and redistributable/usable under its
  license;
- runtime fits the declared research setting;
- state-to-IRIS feature mapping is frozen before score evaluation;
- ensemble or other uncertainty is exposed;
- identical masked identities are used against P0/P1/P2;
- no hidden use of later observations.

If these cannot be met, full MHD stays out of the competition claim.

## What could become the strongest result

A strong positive result would look like this:

- under a predeclared six-hour-equivalent or cadence-appropriate missing-source
  scenario, simple KNN/statistical recovery causes a measurable loss in TSS or
  calibration;
- a physics-constrained reconstruction recovers more of that downstream skill;
- its uncertainty identifies when reconstruction is too poor to trust;
- IRIS degrades or abstains instead of exposing a normal-looking probability
  outside the validated reconstruction envelope;
- the effect repeats across chronological source eras and survives paired
  uncertainty analysis.

That would connect physics, forecasting, missing-data robustness and operator
review in one causal experiment.

A negative result is also useful. If physics reconstruction gives prettier
fields but no downstream benefit, IRIS should explicitly show that visual or
physical plausibility did not translate into forecast utility and keep the
simpler system.

## Iterated build plan

### Phase 0 — completed in this continuation

- verify PR #3 remains open/unmerged and branch head before modification;
- freeze `missingness_recovery_contract_v1.json`;
- implement causal reconstruction provenance checks;
- implement deterministic random/block masks;
- implement held-out reconstruction and uncertainty-coverage metrics;
- add unit tests and source-only CI coverage.

### Phase 1 — data-bound missingness audit

Blocked until the exact verified train-only cohort/source manifest is available.
Then produce, before modeling:

- missingness rate by modality, feature, source era and calendar quarter;
- longest contiguous gaps;
- event/quiet support inside each missingness regime;
- publication latency distribution;
- whether missingness itself leaks event/outcome information;
- which values are truly absent versus structurally unavailable in an era.

Output: one immutable missingness manifest and a decision on which modalities
are eligible for reconstruction.

### Phase 2 — nonphysics baselines

Freeze the exact inner chronological folds and run P0 strategies only. This
establishes the floor that physics has to beat. No architecture search.

### Phase 3 — reduced physics arm

Implement exactly one magnetic reconstruction method chosen from the Phase-1
source geometry and available open implementations/equations. Validate it first
on deliberately hidden known observations. Then measure downstream SEP impact.

Stop if it fails the V1 survival gate.

### Phase 4 — assimilation / optional MHD escalation

Only if Phase 3 passes but leaves meaningful gap-length degradation, test one
assimilation extension. Escalate to full MHD only after a written residual-gap
argument and feasibility receipt. Avoid building a global MHD solver from
scratch merely for presentation value.

### Phase 5 — integrate the surviving recovery path

A surviving recovery arm becomes a separately identified input source, never an
`observed` value. Extend the admission/inference bundle so model, calibration,
threshold, reconstruction method, source observations and reconstructed payload
are receipt-bound. Forecasts using reconstruction are at least `DEGRADED` unless
a later frozen policy establishes a narrower validated status.

### Phase 6 — final forecast experiment

After the primary verified NEW-crossing cohort exists:

1. climatology;
2. eligible persistence;
3. elastic net;
4. XGBoost;
5. faithful reproduced published comparator;
6. compact IRIS baseline;
7. + causal proton context;
8. + XRS;
9. + the single surviving missingness/reconstruction path, if any.

Freeze the simplest surviving candidate before independent evaluation.

### Phase 7 — operator evidence

Replay real historical outages/recovery periods where data provenance permits.
Measure how often IRIS would emit VALID, DEGRADED or ABSTAIN and whether
reconstruction uncertainty correctly predicts downstream error. Do not invent
company costs or protection thresholds.

## Stop/go rules

Stop a physics arm when any one of these is true:

- uses post-issue information;
- cannot be reproduced from a pinned implementation/artifact;
- reconstructs held-out observations worse than a materially simpler causal
  baseline without downstream compensation;
- downstream TSS/FAR/calibration is not better than the best simpler recovery
  arm within paired uncertainty;
- benefit appears in only one unsupported source era;
- runtime/latency makes the declared experiment infeasible;
- uncertainty cannot identify high-error reconstructions.

Go forward only on evidence, not architecture prestige.

## Claim boundary

This strategy does not establish that IRIS accurately simulates solar flares,
solves the MHD equations, beats any company, improves the final NEW-crossing
forecast, saves money, is operationally ready, constitutes a breakthrough, or
will win IRIS/ISEF. Those require completed experiments.
