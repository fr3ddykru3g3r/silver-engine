# IRIS-SEP GitHub continuation

Work branch: `codex/iris-sep-continuation-20260905`. Until its PR is merged, the project checkpoint is on that branch, not on `main`.

## Read this first

**Authoritative current status:** `CURRENT_STATUS.md`.

This file is retained as a stable entry point and historical pointer. Older architecture reports and receipts remain preserved for audit continuity, but they are not authoritative for “what is current” when they conflict with `CURRENT_STATUS.md`.

## Current project in one sentence

> **Can a causally defensible daily model forecast a NEW >10 MeV, >=10 pfu SEP threshold crossing within 24 hours, while explicitly degrading or abstaining when the available inputs do not justify a normal forecast?**

## Current development candidate

`IRIS_CROSSFIT_EVIDENCE_STACK_V1` remains the development candidate. It uses separate solar, XRS and historical-proton specialists, out-of-fold fit-era evidence fusion, calibration on the calibration role and threshold selection on the threshold role.

No architecture expansion on the current daily aggregate table is authorized during the two-week submission sprint.

## Important updated evidence boundary

Real development experiments **have run**. Existing score/monitor results are retrospective development evidence and have already informed model choices. They cannot later be relabelled as fresh final evidence.

Strict forecast-time causality for every predictor in the released aggregate table is **not yet established**. The pinned upstream preprocessing contains retrospective interpolation/harmonization steps, so a finite aggregate cell is not automatically a native observation. See the provenance audit before making any sensor-reconstruction or prospective-replay claim.

The locked test remains untouched during development.

## Current reading order

1. `CURRENT_STATUS.md`
2. `config/current_development_architecture_v1.json`
3. `CHANGELOG.md`
4. `architecture/THRESHOLD_POLICY_RECONCILIATION_2026-09-06.md`
5. `provenance/SOURCE_PROVENANCE_AUDIT_2026-09-06.md`
6. `config/source_provenance_contract_v1.json`
7. `config/inspected_evidence_registry_v1.json`
8. `config/benchmark_contract_v2.json`
9. `config/evaluation_policy_v1.json`
10. `FUTURE_PLAN.md`

## Preserved historical material

For prior design decisions, rejected models, missing-data physics work, compact replay status, source-era analysis and previous continuation checkpoints, read the dated files under `architecture/`, `receipts/`, `evidence_checkpoint/` and the full `CHANGELOG.md`.

Failed experiments and negative results are part of the evidence trail and must not be deleted or rewritten because later work supersedes their interpretation.
