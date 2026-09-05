# IRIS-SEP GitHub continuation

Work branch: `codex/iris-sep-continuation-20260905`. Until its PR is merged,
the project checkpoint is on that branch, not on `main`.

Read these in order:

1. `architecture/PUBLICATION_REVIEW_2026-09-05.md` — corrections to earlier claims.
2. `architecture/SOL_CONTINUATION_UPDATE_2026-09-05.md` — latest source-only continuation, negative results and exact next action.
3. `architecture/CONTINUATION_STATUS.md` — prior history, evidence and blockers.
4. `FUTURE_PLAN.md` — milestones and stop/go decisions.
5. `SOL_HANDOFF.md` — executable continuation instructions.
6. `config/benchmark_contract_v2.json` and `config/evaluation_policy_v1.json`.
7. `evidence_checkpoint/INDEX.json` — selected small receipts copied byte-for-byte
   from local experiments; hashes and original paths are recorded.

## SOL continuation checkpoint — 2026-09-05

PR #3 was verified open and unmerged before this continuation. Its starting head
was `0a06c8a82b9a60d6801e052ea3c0378ae44e2ab8`; work continued only on
`codex/iris-sep-continuation-20260905`.

Source-only verification in the continuation runtime used Python 3.13.5 with
NumPy 2.3.5, PyTorch 2.10.0+cpu and scikit-learn 1.8.0. This command completed
17 tests with zero failures or errors:

```sh
python3 -m unittest \
  iris_report.iris_sep.tests.test_pilot_replay \
  iris_report.iris_sep.tests.test_pilot_admission_v2 \
  iris_report.iris_sep.tests.test_validity_envelope_benchmark \
  iris_report.iris_sep.tests.test_compound_validity_benchmark \
  iris_report.iris_sep.tests.test_inference_bundle \
  iris_report.iris_sep.tests.test_compact_layer_replay -v
```

The five new source/test modules also passed `python3 -m py_compile`. Exact
runtime/results are recorded in
`receipts/sol_continuation_source_only_v2_2026-09-05.json`.

A later compact-replay hardening patch added an explicit audit of loaded
checkpoint parameter tensors, zero-support feature counts, and regressions for a
nonfinite checkpoint parameter plus nonbinary missing masks. Because shell DNS
could not resolve `github.com`, the affected files were hydrated from the
verified branch into an isolated source-only tree. Python 3.13.5 / NumPy 2.3.5 /
PyTorch 2.10.0+cpu passed `py_compile` and 3 focused unittest executions with
zero failures/errors. This focused post-patch check supplements, rather than
replaces, the earlier 17-test run. See
`receipts/compact_checkpoint_audit_patch_2026-09-05.json`.

Full artifact verification was **NOT_RUN** in this execution environment because
the preserved development datasets, folds, checkpoints and notebooks were not
present. Do not download a replacement or mixed training/test table to make it
run.

The controlled original-compact replay is implemented in
`tools/run_compact_nonfinite_replay.py`, but its causal run was also **NOT_RUN**
here because the exact fold-3 source/folds/seed-7 logits/checkpoint/preprocessing
artifacts were absent. The actual first nonfinite layer and root cause remain
unknown. A synthetic test demonstrates a possible float64-to-float32 cast
overflow hazard, but that is not evidence that it caused the retained failure.
Synthetic all-missing inputs remained finite. Helper mutation and distribution
shift are likewise not established causes. The hardened replay now also records
whether the loaded checkpoint itself already contains any nonfinite parameter
values, so an input-side explanation is not inferred from the first bad layer
output alone. See `receipts/compact_nonfinite_replay_2026-09-05_sol.json`.

Admission V2 now has a canonical offline inference bundle in
`src/iris_sep/inference_bundle.py`. It binds policy, source revisions,
transformed feature arrays, raw model output, logit-intercept calibration,
operator thresholds, request metadata and the evidence receipt. The evidence
receipt must itself carry a static inference-binding digest; the independent
bounded adversarial review found and closed the weaker design where only the
outer bundle hash protected those fields. See
`workstreams/sol_adversarial_bundle_20260905/RED_TEAM_PLAN.md`.

No locked test/identity/outcome was accessed, no outer monitor was rescored, no
new model was trained, the publisher request was not resent, and no external
message was sent. No deletion was performed because no artifact/cache deletion
could be backed by a verified hash-matching recovery copy; see
`receipts/cleanup_manifest_2026-09-05_sol.json`.

**Exact next action:** in the preserved artifact workspace, run the controlled
compact replay against the pinned train-only source and retained fold-3 seed-7
artifacts. Accept a first-nonfinite-stage conclusion only if the saved logits are
reproduced exactly. Inspect the checkpoint-parameter audit alongside feature
support, train-fitted scaling/cast behavior, missing masks, branch outputs, gate
tensors and final logits before assigning causality. Do not change preprocessing
or model behavior before that proof. In parallel, wait for verified training-only
NEW-crossing data with latency, licensing, episode semantics, comparator fidelity
and an independent frozen evaluation arrangement. Final model training remains
blocked until those requirements are satisfied.

## Reproduction levels

Source-only checks require NumPy and the repo root on the Python path. Verify
that the interpreter matches compiled dependencies first. On the original
laptop `/private/tmp/iris_sep_pydeps` contains CPython 3.14 binaries, so use
`/opt/homebrew/bin/python3`; Apple `/usr/bin/python3` is CPython 3.9 and is
incompatible with that directory.

```sh
/opt/homebrew/bin/python3 -m unittest \
  iris_report.iris_sep.tests.test_pilot_replay \
  iris_report.iris_sep.tests.test_pilot_admission_v2 \
  iris_report.iris_sep.tests.test_validity_envelope_benchmark \
  iris_report.iris_sep.tests.test_compound_validity_benchmark \
  iris_report.iris_sep.tests.test_inference_bundle \
  iris_report.iris_sep.tests.test_compact_layer_replay -v
```

Full local artifact verification additionally requires the preserved development
datasets, third-party source snapshots, notebooks and model checkpoints. These
are intentionally outside ordinary Git. Do not download a mixed train/test table
to satisfy missing files. On the original laptop, first check dependencies, then:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/private/tmp/iris_sep_pydeps python3 -m iris_report.iris_sep.tests.verify_local
```

The temporary dependency directory is not portable. Recreate dependencies in a
small isolated environment when absent; do not treat missing dependencies or
artifacts as failed scientific results. Report source-only and full verification
separately. A fresh GitHub checkout cannot reproduce full artifact checks yet.

## Storage and synchronization

Fetch only the work branch with shallow, blob-filtered access. Compare branch
heads and local changes before copying anything. Preserve existing root
`iris-model/`; newer local source is namespaced under `iris_report/`.
Never blindly stage all files, force-push, rewrite history, merge automatically,
or materialize large data. Do not create a large `.git` in the original workspace.

For cleanup, inventory exact paths and sizes; check references in receipts;
verify a remote/source commit or external artifact backup by hash; then remove
only unreferenced duplicates or reproducible caches. Record a cleanup manifest.
Old versions required by receipts and the only copy of failed experiments must
remain. Source publication is not an artifact backup.
