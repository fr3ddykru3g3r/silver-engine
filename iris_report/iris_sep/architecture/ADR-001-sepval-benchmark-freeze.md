# ADR-001: Freeze the causal SEPVAL benchmark before model implementation

Status: accepted on 2026-09-04; locked-test outcomes not accessed.

## Decision

IRIS-SEP predicts a **new** >10 MeV, >=10 pfu threshold crossing in the next
24 hours. Inputs stop at the forecast issue time and incorporate realistic
publication latency. Windows already above threshold are excluded from the
primary cohort or reported separately. Complete SEP episodes remain in one
chronological partition, and overlapping 24-hour windows are purged across
partition boundaries.

The five roles are train, validation monitor, validation calibration,
validation threshold, and locked test. Imputation, scaling, feature selection,
dimensionality reduction, architecture selection, calibration, and operating
threshold selection are all completed without locked-test access. The locked
test is opened once for final evaluation after configuration and prediction
interfaces are hashed.

All models use the same frozen issue identities. The paired resampling unit is
a complete SEP episode or a predeclared quiet block, never an individual
overlapping hourly row.

## Claim gate

A positive headline requires all of the following: higher median TSS than the
reproduced and operationally recalibrated SEPNET-O; a paired 95% bootstrap
interval for the TSS difference wholly above zero; lower false-alarm ratio at
matched detection probability; no material calibration or lead-time
degradation; identical evaluation identities; and every leakage/data-contract
audit passing. Otherwise the result is negative or inconclusive.

The machine-readable source of truth is `config/benchmark_contract.json`.
