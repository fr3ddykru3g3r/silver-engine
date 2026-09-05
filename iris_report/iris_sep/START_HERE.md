# IRIS-SEP GitHub continuation

Work branch: `codex/iris-sep-continuation-20260905`. Until its PR is merged,
the project checkpoint is on that branch, not on `main`.

Read these in order:

1. `architecture/PUBLICATION_REVIEW_2026-09-05.md` — corrections to earlier claims.
2. `architecture/CONTINUATION_STATUS.md` — history, evidence, blockers.
3. `FUTURE_PLAN.md` — milestones and stop/go decisions.
4. `SOL_HANDOFF.md` — executable continuation instructions.
5. `config/benchmark_contract_v2.json` and `config/evaluation_policy_v1.json`.
6. `evidence_checkpoint/INDEX.json` — selected small receipts copied byte-for-byte
   from local experiments; hashes and original paths are recorded.

## Reproduction levels

Source-only checks require NumPy and the repo root on the Python path. Verify
that the interpreter matches compiled dependencies first. On the original
laptop `/private/tmp/iris_sep_pydeps` contains CPython 3.14 binaries, so use
`/opt/homebrew/bin/python3`; Apple `/usr/bin/python3` is CPython 3.9 and is
incompatible with that directory.

```sh
/opt/homebrew/bin/python3 -m unittest iris_report.iris_sep.tests.test_pilot_replay iris_report.iris_sep.tests.test_pilot_admission_v2 iris_report.iris_sep.tests.test_validity_envelope_benchmark iris_report.iris_sep.tests.test_compound_validity_benchmark -v
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
