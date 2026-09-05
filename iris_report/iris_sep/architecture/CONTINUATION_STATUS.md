# IRIS-SEP continuation status

The research objective is a daily forecast of a new >10 MeV, >=10 pfu SEP
threshold crossing within the following 24 hours. The final claim requires
the frozen paired benchmark gate, including lower false-alarm ratio at matched
detection and no material calibration or lead-time degradation.

## Current evidence

The compact neural model and XGBoost have completed five-seed local development
runs. The fixed probability blend has validation-monitor TSS 0.276 versus
0.257 for XGBoost. Its paired difference interval includes zero, and its
matched-detection false-alarm ratio is slightly worse. This is an inconclusive
development result. That monitor has already informed model development.

The approved V6 dual-target table preserves the selected V3 development
identities, partitions and operational labels, adding general SEP and flux
targets for comparator development. It does not resolve the final target or
dataset provenance requirements.

## Comparator correction

The corrected dense SEPNET adapter's V2 experiments have not passed scientific
review. In particular, episode weights were derived from operational-label
groups despite a claim that those labels did not affect training. Restart
equivalence, persisted-metric recomputation and several artifact bindings also
required corrections. Do not use those experiments as an approved comparator
or as evidence of superiority. Preserve their receipts as audit history.

The V5 adapter uses general-target groups for training weights. Nine local tests
pass, including operational-label mutation, validation-feature and label
mutation, fitted-preprocessor equality, and interrupted versus uninterrupted
predictions and checkpoint-state comparisons. Restored early-stopped seeds skip
further training; run configuration binds the implementation hash.

Five-seed V5 runs completed locally. Row-weighted adapter monitor TSS was
0.2367; general-episode weighting reduced it to 0.0740. The latter is rejected
as an improvement candidate. These experiments use the legacy development
target and the already inspected monitor. Public SEPNET-O target/configuration
equivalence remains unestablished. The fixed partition itself used operational
labels, which is disclosed independently from training-weight isolation.

## External dependency

The inspected publisher distribution offers a full table/archive. A verified
training-only release, source latency information and a frozen evaluation
arrangement remain outstanding. The user reports that the training-only /
blinded-evaluation request was sent to the publisher. No sent timestamp or
message ID was supplied, and no reply has been supplied. Do not resend.
See `receipts/publisher_request_user_report_2026-09-05.json`.

Local implementation and small CPU experiments can continue. Final evaluation
must wait for the data contract and model-selection freeze. No locked test
access is authorized during tuning.

## Continuation integration: 2026-09-05

Completed: publisher sent-status correction; Luna A safe-intake checklist;
Luna B primary-literature comparator matrix and conditional train-only ablation
proposal; Luna E independent integrity review; ADR-005 scope reconciliation;
receipt-bound synthetic replay tooling and 11 failure/status cases.

Verification: the full local gate passed (82 unittest executions plus static,
contract and pinned-artifact checks). The fixture boundary also passed targeted
tests after final hardening. Commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/private/tmp/iris_sep_pydeps python3 -m iris_report.iris_sep.tests.verify_local
python3 -m unittest iris_report.iris_sep.tests.test_pilot_replay -v
python3 -m iris_report.iris_sep.tools.build_fixture_replays --output iris_report/iris_sep/replays/synthetic_continuation_20260905
```

Authoritative continuation artifacts:

- `receipts/continuation_verification_2026-09-05.json` and adjacent `.log`;
- `receipts/publisher_request_user_report_2026-09-05.json`;
- `architecture/CONTINUATION_AUDIT_DISPOSITION_2026-09-05.md`;
- `workstreams/luna_a_continuation_20260905/README.md`;
- `workstreams/luna_b_continuation_20260905/README.md` and
  `preregistered_inner_fold_batch.md` (proposal, not yet executable/frozen);
- `workstreams/luna_e_continuation_20260905/RED_TEAM_REPORT.md`;
- `replays/synthetic_continuation_20260905/receipt.json` and SVG;
- `receipts/source_sync_comparison_2026-09-05.json`.

No new model optimization or final training occurred. The V1 table lacks the
proton/XRS inputs motivating the next ablations. Final work remains blocked by
a verified training-only release, new-crossing and complete-episode semantics,
latency/licensing manifests, reproduced comparator and frozen blinded evaluation.
Sealed cohort/prediction validation and an evidence registry also remain required
before independent evaluation or real pilot admission. The used monitor remains
selection evidence. No locked test was accessed and the frozen policy is intact.

GitHub synchronization uses a separate shallow, blob-filtered checkout and a
`codex/` branch. No `.git` is recreated in this workspace. Git contains source
and small audit receipts; full verification requires hash-matching local
artifacts and notebooks. Retain historical files referenced by receipts and
failed experiments; no large artifact upload or destructive cleanup is allowed
without verified preservation. The GitHub PR is not merged automatically.
