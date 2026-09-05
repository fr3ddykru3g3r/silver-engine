# IRIS-SEP substantial contribution strategy

## The real problem

Satellite operators do not consume a TSS score. They decide whether to continue
normal operations, monitor conditions, prepare protection, or protect a
spacecraft. A forecast can fail them in two distinct ways: it can predict the
wrong event outcome, or it can look valid when its input data are stale,
missing, incompatible, or outside the model's supported era.

The current evidence says IRIS does not yet solve the first problem better than
the best local reference. It has begun to solve the second, and the train-only
rolling experiment shows why that matters: a conventional compact model emitted
nonfinite forecasts under a later-era shift while its aggregate development
score had appeared normal.

## Proposed contribution

IRIS-SEP is an availability-aware SEP decision-support system with two linked
research contributions:

1. **A frozen, forecast-time NEW-crossing benchmark.** One daily calibrated
   probability of a new >10 MeV, >=10 pfu crossing in the following 24 hours,
   compared on identical issue identities against strong classical models and
   a reproduced published comparator.
2. **An operational-validity envelope.** Each forecast is admitted, degraded,
   or rejected using cryptographically bound evidence, source publication time,
   freshness, schema, modality availability, uncertainty, and supported-era
   checks. Reliability is evaluated under declared feed-failure and
   distribution-shift scenarios, alongside event skill.

The scientific hypothesis is not that a larger network wins. It is that adding
strictly forecast-time particle/radiative context and explicit source-regime
handling improves generalization and false-alarm performance, while the
validity envelope prevents unsupported forecasts from being presented as safe.

## Why companies should care

NOAA's S1 radiation-storm threshold is the same >10 MeV, 10 pfu boundary used
here, and NOAA documents satellite impacts ranging from operational disruption
to memory errors, loss of control, sensor noise, and solar-panel damage at
higher storm levels. NASA's CCMC validates SEP probability, all-clear,
threshold-crossing time, false alarms, skill scores, and flux quantities for
human-spaceflight and operational contexts. NASA also distinguishes forecasts
made only with pre-event information from scientifically interesting forecasts
that use later data.

Real feed failure is not hypothetical. The 2024 HESPERIA REleASE+ work states
that gaps affect real-time SOHO, ACE, and STEREO streams and implements fallback
behavior when radio-data gaps exceed a threshold. IRIS generalizes that
principle into an auditable model-independent admission contract.

Primary sources:

- https://www.swpc.noaa.gov/node/1085
- https://ccmc.gsfc.nasa.gov/challenges/sep/validation/
- https://ccmc.gsfc.nasa.gov/assessment/topics/SEP/campaign2020/next_steps.php
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW004013
- https://arxiv.org/html/2606.14440

## What would count as a substantial result

The final result needs two tables, neither substituting for the other.

### Forecast table

- identical frozen NEW-crossing cohort;
- reproduced comparator plus climatology, elastic net, and XGBoost;
- paired episode/quiet-block TSS interval;
- FAR at matched detection;
- Brier score, ECE, and lead time;
- performance and coverage by source era.

The superiority claim requires the existing frozen gate. A point estimate or a
large relative percentage from a weak denominator is not substantial evidence.

### Validity-envelope table

- valid fresh input;
- stale source;
- publication after issue time;
- missing critical and optional modalities;
- schema mutation;
- evidence-receipt mutation;
- model/policy/calibration mismatch;
- unsupported source era;
- out-of-distribution feature magnitude;
- recovery and fallback behavior.

Report unsafe-valid rate, safe abstention rate, retained valid coverage, time to
detection, and operator-state suppression. An “astronomical difference” is
credible only if it is categorical and measured—for example, zero unsafe valid
outputs over a large adversarial test battery versus a precisely defined
unguarded wrapper. It must not be described as superiority over named systems
unless their public interfaces are tested under identical faults.

## Immediate build sequence

1. Add source-era and feature-support detection to the admission contract.
2. Build a deterministic fault-injection benchmark with thousands of mutations,
   including magnitude shift that reproduces the observed nonfinite-logit case.
3. Measure coverage-risk curves and a predeclared utility sensitivity grid;
   never pick operator costs after seeing results.
4. On verified NEW-crossing training data, test only fixed additions: historical
   proton context, then XRS, then one verified connectivity feature.
5. Freeze the simplest candidate before blinded evaluation.
6. Produce the paper and video from the forecast and validity receipts.

## Current verdict

The project is not presently award-ready. It now has a concrete model, a full
train-only comparison, and a discovered failure that motivates a stronger and
more useful contribution. The competitive path is a forecast plus a formally
tested validity envelope. Visual polish begins after these two evidence tables
exist; the paper structure and video narrative can be prepared now.

## First validity-envelope result

The deterministic V1 fault benchmark executed 10,000 synthetic contract cases
and produced zero unsafe valid outputs across 7,830 injected-invalid cases. A
precisely defined unguarded serializer would emit 7,830/7,830. The stronger V3
benchmark added unsupported eras, unknown source revisions, magnitude shift,
nonfinite outputs, compound failures, and recovery. It completed 10,000 trials
across 299 unique fault combinations with zero status errors, zero unsafe-valid
outputs, and zero failures in 12 fault-to-recovery sequences. Magnitude and
finiteness are computed inside the admission boundary from supplied arrays,
rather than trusted caller flags. This is a large software-safety result, not
SEP forecast skill and not a claim against named competitors. Real feed replays
and an independently reviewed fault oracle remain required.
