# Daily modality-outage evidence extraction — 2026-09-06

## Decision

The preregistered daily aggregate-interface outage benchmark and the preregistered selective handling-policy analysis both completed with independent metric audit. The experiment is scientifically usable as **quiet-window reliability evidence**, but it is **not yet event-preservation evidence** because every deliberately selected outage block in the score role contains zero NEW-crossing positives.

That limitation is material. It means this run can measure probability drift and false-alert behavior when XRS/proton families disappear, but it cannot establish that reconstruction or abstention preserves SEP detection during event-bearing outages.

## Immutable execution

- GitHub Actions run: `34050680276`
- job: `101533612567`
- immutable artifact ID: `9994585108`
- artifact name: `iris-sep-daily-outage-34050680276`
- artifact ZIP SHA256: `269d53a2c10bd0e11afcf3cb0c87602122eb139d289c0a9193a7c759f015af01`
- prediction CSV SHA256: `033fdad48e87df8993f95eac78584f2fbbb8bc3b0d23324731f5a76b6847928f`
- status: `COMPLETED_DAILY_MODALITY_OUTAGE_DEVELOPMENT_ONLY`
- locked test accessed: **false**
- monitor used: **false**
- retraining/recalibration/rethresholding after outage: **false/false/false**
- all 9 modality-duration scenarios reported: **yes**
- all 3 recovery arms reported: **yes**
- all 162 preregistered handling-policy evaluations reported: **yes**
- independent audit: **passed**

The earlier workflow run `34047881188` is retained as a failed audit attempt. Forecast execution completed there, but the independent checker disagreed with the runner about reporting Brier skill on one-class affected subsets. The correction changed only the independent checker's one-class reporting semantics; it did not change outage locations, labels, model outputs, recovery arms, thresholds or predictions.

## Frozen clean forecast

Score role: **3,219 daily issue rows, 21 NEW-crossing positives**.

Model: `IRIS_CROSSFIT_EVIDENCE_STACK_V1`.

| Policy | Threshold | TSS | POD | FPR | FAR | Brier | ECE | AUPRC | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benchmark primary `MAX_TSS` | 0.0190993 | 0.4257 | 0.5714 | 0.1457 | 0.9749 | 0.006348 | 0.003780 | 0.07672 | 0.86882 |
| Reliability `POD80_MIN_FAR` | 0.0160292 | 0.5120 | 0.7143 | 0.2023 | 0.9773 | 0.006348 | 0.003780 | 0.07672 | 0.86882 |

The very high FAR is itself operationally important: this development stack is not an operationally certified alerting system. Any operator-facing use must present probability and validity state separately and must not imply that the current threshold policy is deployment-ready.

## Exposure support — critical limitation

The deterministic label-blind placement rule selected five blocks per modality-duration scenario. All affected score rows are negative in every scenario:

| Scenario | Affected rows | Affected positives |
|---|---:|---:|
| PROTON 24 h | 5 | 0 |
| PROTON 72 h | 15 | 0 |
| PROTON 168 h | 35 | 0 |
| XRS 24 h | 5 | 0 |
| XRS 72 h | 15 | 0 |
| XRS 168 h | 35 | 0 |
| XRS + PROTON 24 h | 5 | 0 |
| XRS + PROTON 72 h | 15 | 0 |
| XRS + PROTON 168 h | 35 | 0 |

Therefore affected-subset TSS, POD, AUPRC and AUROC are undefined. No statement such as “abstention preserves detection” or “reconstruction avoids missed SEP events” is supported by this run.

## Recovery results

Three frozen arms were compared without model refitting:

1. `MASK_AWARE_NO_FILL`
2. `TRAIN_FIT_MEDIAN`
3. `CAUSAL_FORWARD_FILL`

Mean absolute probability drift is measured only on outage-affected rows relative to the clean frozen forecast.

| Scenario | Forward-fill drift | No-fill drift | Train-median drift |
|---|---:|---:|---:|
| PROTON 24 h | **0.001129** | 0.002706 | 0.001736 |
| PROTON 72 h | **0.002021** | 0.003135 | 0.002660 |
| PROTON 168 h | **0.001776** | 0.003065 | 0.003226 |
| XRS 24 h | **0.002907** | 0.003033 | 0.009309 |
| XRS 72 h | **0.002435** | 0.004425 | 0.008952 |
| XRS 168 h | **0.002745** | 0.004617 | 0.007620 |
| XRS + PROTON 24 h | **0.002908** | 0.003943 | 0.013860 |
| XRS + PROTON 72 h | **0.003762** | 0.005209 | 0.013318 |
| XRS + PROTON 168 h | **0.003239** | 0.005561 | 0.013227 |

**Causal forward-fill has the smallest probability drift in all nine contiguous daily-outage scenarios.** This agrees with the earlier random-cell missingness transfer, but it remains development-only evidence.

### Benchmark-primary decision effect (`MAX_TSS`)

Each cell shows `candidate false alerts on affected quiet rows / clean-reference false alerts`, followed by whole-score delta TSS.

| Scenario | Forward fill | No fill | Train median |
|---|---:|---:|---:|
| PROTON 24 h | 0/1; +0.000313 | 0/1; +0.000313 | 1/1; +0.000000 |
| PROTON 72 h | 2/3; +0.000313 | 1/3; +0.000625 | 2/3; +0.000313 |
| PROTON 168 h | 5/6; +0.000313 | 1/6; +0.001563 | 9/6; -0.000938 |
| XRS 24 h | 0/1; +0.000313 | 0/1; +0.000313 | 3/1; -0.000625 |
| XRS 72 h | 3/3; +0.000000 | 0/3; +0.000938 | 7/3; -0.001251 |
| XRS 168 h | 7/6; -0.000313 | 0/6; +0.001876 | 13/6; -0.002189 |
| XRS + PROTON 24 h | 0/1; +0.000313 | 0/1; +0.000313 | 5/1; -0.001251 |
| XRS + PROTON 72 h | 2/3; +0.000313 | 0/3; +0.000938 | 13/3; -0.003127 |
| XRS + PROTON 168 h | 6/6; +0.000000 | 0/6; +0.001876 | **27/6; -0.006567** |

The apparent no-fill improvement is not evidence that no-fill is generally superior. These are all-negative outage blocks, so suppressing probabilities can only help false-alert accounting here and cannot reveal the event-detection cost.

Train-fit median is particularly unsafe for XRS-family gaps in this diagnostic. In the 168 h combined XRS+proton scenario it increases affected quiet-row false alerts from 6 to 27 under MAX_TSS. The paired whole-score TSS interval is entirely below zero for this scenario, and similarly adverse non-zero-crossing intervals occur for several XRS/combined train-median cases.

Causal forward-fill changes whole-score TSS only very slightly in these quiet-window blocks. All nine forward-fill paired TSS intervals cross zero under both threshold policies; it should therefore be described as probability-preserving here, not as forecast-skill improving.

## Selective handling policy

Three preregistered output policies were evaluated for every scenario x recovery arm x threshold policy:

- `ALWAYS_EXPOSE_NORMAL`
- `EXPOSE_DEGRADED_ON_DECLARED_OUTAGE`
- `ABSTAIN_ON_DECLARED_OUTAGE`

`DEGRADED` does not alter probability or thresholding; it changes the validity state shown to an operator. `ABSTAIN` withholds affected rows and counts any abstained positive as a missed positive in full-cohort accounting.

Because these particular affected rows contain zero positives, abstention removes affected-row false alerts without incurring an observed missed positive. That is a property of these selected blocks, **not a validated safety result**.

Coverage if all affected outage rows are abstained:

| Outage duration | Rows withheld per scenario | Forecast coverage |
|---|---:|---:|
| 24 h | 5 / 3219 | 99.8447% |
| 72 h | 15 / 3219 | 99.5340% |
| 168 h | 35 / 3219 | 98.9127% |

For causal forward-fill, abstaining on the affected rows would remove the following candidate false alerts from exposed output under `MAX_TSS`: 0/2/5 for proton 24/72/168 h, 0/3/7 for XRS, and 0/2/6 for combined XRS+proton. Under `POD80_MIN_FAR` the corresponding counts are 1/3/7, 0/4/9, and 0/3/7.

Again, this does not establish the missed-event cost because no affected positive exists in the original label-blind block set.

## What is useful for operators / companies now

The evidence supports a **failure-aware input pipeline**, not an “always reconstruct” product claim:

1. **OBSERVED** — use the authoritative forecast-time observation.
2. **ALTERNATE_OBSERVED** — if the primary feed is unavailable, prefer a valid redundant/alternate observed source after source-specific harmonization.
3. **RECOVERED** — reconstruct only a transient gap for a modality/gap length whose recovery method has been validated on deliberately hidden real values; preserve method, source age and uncertainty provenance.
4. **DEGRADED** — expose the probability with an explicit degraded-data state when evidence says the probability remains informative but support is outside the normal envelope.
5. **ABSTAIN** — withhold a normal-looking probability when source era, gap length, recovery uncertainty, revision, freshness or evidence bindings fail.

A reconstructed value must never be relabelled as observed. Structural historical unavailability must never be filled as if a sensor had existed. Recursive synthetic-to-synthetic carry-forward is not automatically allowed beyond a validated horizon.

This state machine addresses a real operational failure mode: a downstream decision system should be able to distinguish “the model genuinely saw current data” from “the model is running on a recovered or unsupported input state.”

Relevant operational context:

- NASA's 2026 SEP validation work explicitly evaluates models for operator decision making: https://ntrs.nasa.gov/citations/20260000463
- NASA CCMC's SAWS-ASPECS describes SEP-warning outputs tailored to spacecraft and launch operators: https://ccmc.gsfc.nasa.gov/models/SAWS-ASPECS~1.2/
- ESA spacecraft-operation services treat SEPs as a sudden operational hazard for spacecraft: https://swe.ssa.esa.int/sco_services

These sources establish the operator problem, not superiority of IRIS-SEP over existing operational systems.

## Required next experiment: event-bearing outages

The zero-positive support failure motivates a new, separately preregistered development extension. It must be labelled post-hoc relative to the original daily-outage run but preregistered before its own result-capable code exists.

The new experiment must deliberately stress **all eligible NEW-crossing positive score issue rows** and matched quiet controls. For 24/72/168 h gaps, the outage window ends at the forecast issue row so every recovery arm remains causal. The same frozen promoted model, calibration, thresholds and three recovery arms are retained.

Required event-bearing outputs:

- number and identity hash of eligible positive issue rows;
- event-day probability shift relative to clean forecast;
- TP/FN and POD on affected positive rows under both declared threshold policies;
- quiet-control FP/FAR/FPR;
- full-score TSS/Brier/ECE/AUPRC/AUROC;
- coverage and missed-positive accounting for `NORMAL`, `DEGRADED`, and `ABSTAIN` handling;
- gap-length degradation curve;
- all scenarios, including failures, reported.

No locked test or previously protected final evaluation identities may be accessed.

## Claim boundary

This evidence supports a development claim that causal forward-fill is substantially more probability-stable than train-median/no-fill across the tested contiguous quiet-window outages, while severe or unsupported gaps should carry an explicit validity state. It does **not** yet establish event-detection preservation, physics-reconstruction advantage, raw-sensor outage robustness, operational certification, company superiority, economic savings, or locked-test performance.
