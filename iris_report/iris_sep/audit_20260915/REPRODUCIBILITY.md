# Reproduction and audit scope

Audit base before edits: `13f692a90fc3006181d1c4488da78ff135f24f87` on the remote award-validation branch. Work is isolated on `codex/iris-sep-independent-audit-20260915`; the separate existing dirty working copy was untouched.

From the repository root with numpy, pandas, scipy, scikit-learn, pytest installed:

```sh
PYTHONPATH=iris_report/iris_sep python -m tools.audit_estimands_v1 --archive audit_evidence/archives/10137507101.zip --output audit_evidence/new-strict-run
PYTHONPATH=iris_report/iris_sep python -m tools.audit_followups_v1 --archives audit_evidence/archives --output audit_evidence/new-followups-run
PYTHONPATH=iris_report/iris_sep python -m pytest -q iris_report/iris_sep/tests/test_audit_estimands_v1.py iris_report/iris_sep/tests/test_audit_followups_v1.py
```

Output folders must be new; runners fail rather than overwrite evidence. Archive hashes are pinned. Do not substitute regenerated model predictions. The new draws use seeds 20260915 and 20260916 and are explicitly not the frozen primary bootstrap draws. No training is required.

The downloaded archive ZIPs remain locally under `audit_evidence/archives/`; they are not included in the student-facing package because some contain protected-boundary rows. `archive_manifest.json` records cryptographic identities. Preserve these files securely before GitHub artifact expiry; a hash does not recover an expired artifact. Redistribution/licensing was not certified.

`file_inventory.csv` inventories hashes of current architecture/config/tools/tests/submission files. AST parsing and JSON validation establish syntax/inventory coverage, not expert review of every line or correctness of every experiment. Deep audit concentrated on the episode benchmark, estimator/label code, bootstrap, protected contracts, phase-II/controller lineage, source contracts and evidence receipts. No claim that every historical branch was rerun is made.

The mathematical tests include zero-covariance dependence, arbitrary weighting, bounds, rank reversal, unequal-negative corrections, singleton/persistence behavior, AUC/Brier identities and boundary refusal. Controller tests mutate unresolved labels to check causal timing. The first version of that test incorrectly demanded that a discrete quantile change immediately upon one label resolving; it was corrected to test invariance before resolution and a change later. This was a test expectation failure, not evidence of future-label use.

Synthetic tests do not certify raw satellite causality, all original workflow environments, independent prospective validation, or the assumptions behind a physical event catalog. See `validation.json` for the actual environment and test results.
