# IRIS-SEP: Availability-aware forecasting of new solar energetic particle threshold crossings

## Abstract — evidence-constrained draft

Solar energetic particle forecasts support decisions that protect spacecraft,
instruments, astronauts, and high-latitude operations, but retrospective skill
alone does not establish whether a forecast is safe to use when real-time input
streams are delayed, missing, or distribution-shifted. We develop IRIS-SEP, a
research decision-support framework for one daily probability of a new >10 MeV,
>=10 pfu threshold crossing within 24 hours. The framework combines a frozen,
episode-disjoint forecast benchmark with a receipt-bound operational-validity
envelope that returns VALID, DEGRADED, or ABSTAIN before exposing an advisory
state. In a preliminary experiment restricted to the training portion of a
legacy-label development cohort, fixed XGBoost achieved pooled chronological
TSS 0.287, compared with 0.276 for elastic net and 0.258 for a numerically
stabilized compact neural model. Neither candidate showed a positive paired
advantage over XGBoost. The original neural model produced nonfinite logits in
the latest chronological fold, revealing a failure hidden by aggregate
development evaluation. Structural analysis found severe source-era
missingness and uneven event support. These findings motivate forecast-time
particle and radiative context, explicit source-regime handling, and systematic
fault-injection validation. In 10,000 deterministic synthetic contract trials,
the V1 validity envelope returned the expected status in every case and emitted
zero unsafe valid outputs across 7,830 injected-invalid cases; this is a
software-safety result against a defined unguarded serializer, not SEP skill or
named-competitor superiority. Final scientific comparison awaits an immutable
training-only NEW-crossing cohort and blinded evaluation. No superiority or
operational-readiness claim is made.

## 1. Research question

Can availability-aware, forecast-time particle and solar context improve a
daily 24-hour NEW-crossing forecast over reproduced competitors while a formal
validity envelope prevents unsupported forecasts from reaching satellite
operators?

## 2. Operator decision

The model supplies evidence for a human decision among NORMAL, MONITOR, PREPARE,
and PROTECT. It does not command a spacecraft. The decision record includes
probability, all-clear probability, uncertainty, model and policy versions,
input observation/publication times, missingness, and an evidence-receipt hash.

## 3. Methods

### 3.1 Forecast target and cohort

Define a positive issue only when the >10 MeV integral proton flux is below
10 pfu at issue time and newly crosses 10 pfu during the following 24 hours.
Keep complete episodes together and purge overlapping forecast horizons.

### 3.2 Predictors

The first validated model uses aggregate magnetic state, eruption evidence,
historical proton context, and XRS context only when publication-time metadata
proves availability. AIA/HMI image fusion and secondary heads remain excluded.

### 3.3 Models and evaluation

Compare climatology, causally available persistence, elastic net, XGBoost,
reproduced SEPNET/SEPNET-O, and compact IRIS on identical identities. Fit
preprocessing, early stopping, calibration, and thresholds only in their
declared roles. Evaluate TSS, paired uncertainty, matched-detection FAR, Brier,
ECE, lead time, and source-era stability.

### 3.4 Operational-validity envelope

Before releasing a probability, verify schema, model/policy/calibration
bindings, evidence receipt, source publication time, freshness, missing critical
modalities, uncertainty completeness, supported source era, and numerical
finiteness. Failure produces ABSTAIN with no probability or advisory state.

## 4. Preliminary results

The current evidence belongs exclusively to the pinned outer training role and
uses a legacy operational-window label. Across four chronological score blocks,
XGBoost TSS was 0.013, 0.432, 0.295, and 0.020. Pooled XGBoost TSS was 0.287.
Elastic net scored 0.276; its paired TSS difference was -0.0111 with 95% interval
[-0.0451, 0.0225]. Signed-log compact IRIS scored 0.258; its difference was
-0.0289 with interval [-0.0928, 0.0285]. The untransformed compact model failed
with nonfinite logits in the latest fold. No candidate passed selection.

## 5. Interpretation

The experiment rejects the idea that architecture complexity alone is the next
step. It supports testing source-regime robustness and additional physical
context. It also provides a concrete operational failure for the validity
envelope to detect. The result does not answer the final research question
because the current target is not verified NEW-crossing data and the published
comparator has not been reproduced.

The initial deterministic validity-envelope benchmark covered nine status and
fault types over 10,000 trials. IRIS produced no unsafe valid output in 7,830
injected-invalid cases, while the defined unguarded serializer emitted a
probability in every invalid case. The result establishes implementation
behavior for these synthetic mutations only. It does not measure event skill,
real outage frequency, economic utility, or the behavior of another SEP model.
The subsequent V3 compound benchmark covered 299 unique fault combinations,
including unsupported eras, source-revision mismatch, extreme standardized
feature magnitude, nonfinite output, and 12 recovery sequences. All 10,000
statuses and all recovery sequences matched the declared oracle. These remain
synthetic software tests pending independently designed and real-feed faults.

## 6. Planned decisive experiments

- immutable publisher training-only intake and label/latency audit;
- fixed proton-context and XRS ablations using train-only chronological folds;
- reproduced published comparator on identical inputs and identities;
- one blinded evaluation after model freeze;
- adversarial validity-envelope benchmark and coverage-risk curves;
- historical event, quiet interval, outage, and recovery replays.

## 7. Limitations

Current development data have severe era-dependent missingness, sparse event
support in some chronological blocks, no proton/XRS context, and legacy target
semantics. The validity-envelope fixtures establish software behavior, not
operational certification. Satellite-company costs and protection thresholds
require independent operator input and must not be invented.

## Evidence ledger

- `architecture/TRAIN_ONLY_PERFORMANCE_DIAGNOSIS_2026-09-05.md`
- `artifacts/train_inner_diagnostic_v4/analysis_receipt.json`
- `workstreams/luna_training_diagnosis_20260905/training_diagnosis.json`
- `replays/synthetic_continuation_20260905/receipt.json`
- `config/benchmark_contract_v2.json`
- `config/evaluation_policy_v1.json`
