# IRIS v2 — pre-paper 90% readiness gate

This document defines what must be complete **before** writing the competition paper or scripting the 90-second pitch. It is intentionally result-agnostic: no criterion requires a particular large benchmark score.

## Scientific core

- [x] Research question is mechanistic: which magnetic structures must synthetic LOS magnetograms reproduce to transfer useful information to 24 h M1+ forecasting?
- [x] Whole connected physical-region grouping prevents HARP/NOAA chain leakage.
- [x] Chronological evaluation with >=36 h buffer.
- [x] Primary metric TSS; calibration/discrimination secondary; region-cluster bootstrap.
- [x] Real-only and duplicated-real-positive controls.
- [x] Physics manipulation measured independently from differentiable training losses.
- [x] Destructive controls implemented (PIL blur, geometry flip, block shuffle).
- [ ] At least one physics constraint passes the independent real-data manipulation gate.
- [ ] Selected constrained synthetic arm beats duplicate-real control in repeated future-facing folds, or produces a clear reproducible mechanistic null result.
- [ ] Physical-fidelity -> forecast-utility relationship quantified across realized fidelity levels.

## Generator validity

A generator arm is not eligible for forecasting unless ALL are true on TRAIN ONLY:

1. synthetic unsigned-flux proxy median is within 0.5x–2.0x of real-positive median;
2. synthetic strong-field-area median is within 0.4x–2.5x of real-positive median;
3. saturation fraction |B|>2900 G < 5%;
4. non-trivial sample diversity;
5. for a constrained arm, target hard-physics distance improves >=20% vs matched base;
6. no validation/test forecasting information used to choose physics lambda.

## Replication

- [x] Four outcome-blind rolling-origin fold manifests generated from region chronology only.
- [x] Fold boundaries/hashes frozen before v2 forecast results.
- [ ] Chosen generator/control matrix evaluated on >=3 rolling folds.
- [ ] Direction of key effect is stable across folds/seeds.
- [ ] Aggregate paired connected-region CI reported.

## Blind/prospective evidence

- [x] Prospective prediction freeze/receipt utility implemented.
- [ ] Final model/config/hash frozen.
- [ ] At least one batch of predictions timestamped before its 24 h GOES outcome window completes.
- [ ] Outcomes appended only after horizon elapses; frozen prediction file retained unchanged.

## Scope discipline

Primary story: **constraint -> realized physics -> selective destruction -> transfer utility**.

CDR/SHAP remains secondary robustness evidence and must not displace the principal physical-fidelity experiment.

## Integrity / audit

- [x] Original v1 terminal test is explicitly labelled exploratory after exposure.
- [x] v2 does not tune to exposed v1 test numbers.
- [x] Training and evaluation physics metrics are separate implementations.
- [x] Workflow self-tests fail correctly under shell pipefail.
- [ ] AI-use/provenance ledger updated through final experiment freeze.
- [ ] Two-student manual flare-label audit completed and signed.
- [ ] Real-PIL visual audit completed and signed.
- [ ] Written IRIS clarification on archival spacecraft-observation eligibility stored, OR submission switches to the <=12-month contingency analysis.

## 90% threshold

The project may be called >=90% pre-paper ready only when:

- generator validity gate passes;
- rolling replication is complete;
- destructive-control result is complete;
- prospective predictions are frozen (outcomes may still be accruing if clearly labelled);
- human audits and archival-data eligibility are closed;
- all code/data hashes and final protocol are frozen.

A high TSS by itself never satisfies this gate.
