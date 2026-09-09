# IRIS-SEP continuation status

For the current GitHub plan and SOL instructions, start with `START_HERE.md`.
The source checkpoint is published in PR #3:
https://github.com/fr3ddykru3g3r/silver-engine/pull/3
`architecture/PUBLICATION_REVIEW_2026-09-05.md` supersedes stronger claims below.
The prior upload rejection at the end of this history was followed by explicit
user authorization to publish. Check the PR/head and publication receipt for
current synchronization state rather than treating that historical rejection as
an active authorization blocker.

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

### Train-only performance diagnosis

A four-fold expanding chronological comparison now exists under
`artifacts/train_inner_diagnostic_v4/`. It used only the pinned outer train role.
Across 2,441 disjoint score windows, fixed XGBoost had pooled TSS 0.287,
elastic net 0.276, and signed-log compact IRIS 0.258. Paired unit-bootstrap
intervals for both challengers versus XGBoost included zero. The original
compact model produced nonfinite logits on the latest fold and is incomplete;
the signed-log transformation fixed execution but not skill. XGBoost's fold
TSS ranged from 0.013 to 0.432, exposing severe temporal instability.

The training audit found extreme SHARP missingness and uneven quarterly event
support. These results redirect work toward a verified NEW-crossing cohort,
forecast-time proton/XRS context, source-era handling, and minimum fold-support
rules. See `architecture/TRAIN_ONLY_PERFORMANCE_DIAGNOSIS_2026-09-05.md`.

### Substantial operator contribution checkpoint

The research contribution is now explicitly two-gated: forecast improvement on
the frozen NEW-crossing cohort and an operational-validity envelope that
suppresses unsupported decisions. The latter is implemented in
`src/iris_sep/pilot_admission_v2.py` and computes feature-magnitude and output
finiteness inside its admission boundary. It also checks supported issue era and
allowed source revisions before calling the receipt-bound replay layer.

The authoritative compound synthetic benchmark is
`artifacts/compound_validity_benchmark_v3/`. It ran 10,000 deterministic trials
covering 299 unique combinations of twelve fault types, with zero status errors,
zero unsafe-valid outputs, and zero failures across twelve fault-to-recovery
sequences. This is software-safety evidence, not SEP skill, real outage evidence,
economic benefit, or named-competitor superiority. The earlier V1 and V2 runs
are retained as development history.

`architecture/SUBSTANTIAL_CONTRIBUTION_STRATEGY.md`,
`paper/RESEARCH_PAPER_DRAFT.md`, and `video/VIDEO_NARRATIVE_V1.md` now align the
research, operator decision, evidence boundaries, and presentation. The latest
full local gate passed 92 unittest executions plus static, contract, checkpoint,
and persisted-artifact checks. No locked test was accessed.

GitHub synchronization uses a separate shallow, blob-filtered checkout and a
`codex/` branch. No `.git` is recreated in this workspace. Git contains source
and small audit receipts; full verification requires hash-matching local
artifacts and notebooks. Retain historical files referenced by receipts and
failed experiments; no large artifact upload or destructive cleanup is allowed
without verified preservation. The GitHub PR is not merged automatically.

### GitHub submission blocker

Source-only commit `69e8382db60f8a831a56bc2e8da398cc14006af2` is prepared
on `codex/iris-sep-continuation-20260905` in the separate temporary checkout.
Automatic approval review rejected the push twice, including after destination
ownership and exact-payload scan checks. Nothing was pushed and no PR exists.
The next action requires explicit approval of the 232-file, 1.45 MB upload to
`fr3ddykru3g3r/silver-engine`. See
`receipts/source_sync_submission_status_2026-09-05.json` and the prepared PR body
at `/private/tmp/iris-sep-source-sync-pr-body.md`. Do not bypass the rejection.
