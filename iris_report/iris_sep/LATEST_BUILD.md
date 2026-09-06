# Latest IRIS-SEP build

The authoritative current status is `CURRENT_STATUS.md`.

Do **not** treat older “latest” prose in historical architecture reports as current when it conflicts with that page. Historical reports and receipts remain preserved for audit continuity.

## Current development candidate

`IRIS_CROSSFIT_EVIDENCE_STACK_V1` is the current development candidate. Its architecture is frozen on the current daily aggregate interface; no new architecture search is authorized during the two-week submission sprint.

See:

- `CURRENT_STATUS.md`
- `config/current_development_architecture_v1.json`
- `CHANGELOG.md`
- `architecture/THRESHOLD_POLICY_RECONCILIATION_2026-09-06.md`
- `provenance/SOURCE_PROVENANCE_AUDIT_2026-09-06.md`
- `config/inspected_evidence_registry_v1.json`

## Current evidence boundary

Real development experiments have run and are preserved. They are retrospective development evidence on the released aggregate dataset, not untouched final superiority evidence.

Strict forecast-time causality for every aggregate predictor is not yet established because upstream preprocessing includes retrospective interpolation/harmonization operations and complete row/cell provenance is not preserved in the reviewed model-ready table.

The locked test remains untouched during development.
