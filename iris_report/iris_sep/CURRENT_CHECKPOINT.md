# Current IRIS-SEP checkpoint

Read `CURRENT_STATUS.md` first. It is the authoritative current-state page.

Then read, in order:

1. `config/current_development_architecture_v1.json`
2. `CHANGELOG.md`
3. `architecture/THRESHOLD_POLICY_RECONCILIATION_2026-09-06.md`
4. `provenance/SOURCE_PROVENANCE_AUDIT_2026-09-06.md`
5. `config/source_provenance_contract_v1.json`
6. `config/inspected_evidence_registry_v1.json`
7. `config/evaluation_policy_v1.json`

Historical `START_HERE.md`, architecture reports and receipts remain preserved for audit continuity, but any historical claim that real-data experiments are still unrun is superseded by `CURRENT_STATUS.md`.

The current model is development-only. Existing score/monitor evidence has been inspected. The locked test remains forbidden during development, and strict prospective causality for all released aggregate predictors remains unresolved pending lineage reconstruction.
