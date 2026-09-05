# SOL continuation update — 2026-09-05

This update continues PR #3 on `codex/iris-sep-continuation-20260905` and does
not supersede the frozen benchmark/evaluation contracts. It records only the
bounded source hardening completed after the publication checkpoint.

## Remote and execution state

At continuation start, PR #3 was open and unmerged at
`0a06c8a82b9a60d6801e052ea3c0378ae44e2ab8`. The connector-reported PR had 265
changed files and no deletions. A direct shallow clone was unavailable in the
execution container because DNS for `github.com` was unavailable, so only the
specific source/test files needed for this bounded work were hydrated through
the connected GitHub API. No data cohort was downloaded.

The source-only runtime was Python 3.13.5 with NumPy 2.3.5, PyTorch 2.10.0+cpu,
and scikit-learn 1.8.0. Seventeen targeted tests passed with zero failures or
errors. The new Python files also passed `py_compile`. See
`receipts/sol_continuation_source_only_v2_2026-09-05.json`.

Full artifact verification was **NOT_RUN** because this container did not contain
the preserved development datasets, checkpoints, folds, notebooks, or complete
artifact dependency layout required by `verify_local`. That is an environment
blocker, not a failed scientific result.

## Original compact nonfinite diagnosis

The retained evidence still says fold 3 / original compact / seed 7 produced
only 1,062 finite logits out of 2,120. No previous cause attribution is promoted.

New diagnostic source provides two separate checks:

- `modeling/compact_layer_replay.py` replays every branch, gate, fusion, shared
  layer and final logit with missing-modality dropout disabled, recording the
  first tensor stage containing a nonfinite value. It also reports per-feature
  support and all-missing rows.
- `tools/run_compact_nonfinite_replay.py` applies the saved train-fitted
  preprocessing, checks feature support/scales/missing masks, verifies the
  pinned train-only source and retained failure-logit hashes, loads the retained
  checkpoint, and requires exact saved-logit reproduction before it permits a
  first-nonfinite-stage conclusion.

The exact artifact replay was **NOT_RUN** here because the train-only CSV,
`folds.json`, retained `seed_7.npz`, checkpoint, preprocessing receipt, and model
receipt are not present in this execution environment. The actual first failing
stage and root cause therefore remain unknown.

A synthetic source-only regression demonstrates a narrower code hazard: a value
can be finite after float64 standardization but overflow during the subsequent
float32 cast. This is not evidence that such a cast caused the retained fold-3
failure. A separate all-modalities-missing synthetic replay stays finite, so an
all-missing row is not intrinsically sufficient to produce the observed failure.
Helper mutation and distribution shift likewise remain unproven causes. See
`receipts/compact_nonfinite_replay_2026-09-05_sol.json`.

## Immutable admission-V2 inference bundle

`src/iris_sep/inference_bundle.py` adds an offline, canonical inference envelope
that binds in one artifact:

- admission-V2 source-era, source-revision and magnitude policy;
- runtime schema/policy/model identifiers and source-revision snapshot;
- exact transformed feature array and raw model-output array bytes + digests;
- logit-intercept calibration identifier and parameter;
- operator threshold policy and threshold values;
- forecast-time freshness/missingness/uncertainty request metadata; and
- embedded evidence receipt bytes and digest.

Replay accepts only the immutable bundle bytes plus a separately trusted bundle
SHA-256; the caller no longer supplies a replacement policy or inference arrays.
Derived probability is recomputed from bound raw model output and calibration.

The independent adversarial review found one initial design weakness before
publication: a trusted *new* outer bundle hash could have wrapped a changed
admission/calibration/threshold contract around an unchanged evidence receipt.
The corrected design therefore requires the evidence receipt itself to contain
`inference_binding_sha256`, binding the static admission policy, runtime policy,
calibration parameter, thresholds, model version, and schema. Bundle construction
and replay reject mismatch. This still proves only integrity under the tested
software contract; it does not authenticate a future publisher/model receipt by
itself. See `workstreams/sol_adversarial_bundle_20260905/RED_TEAM_PLAN.md`.

## Storage, cleanup, and evidence boundary

No dataset, model, notebook output, cache, or environment was added to ordinary
Git. No file was deleted: this runtime could not demonstrate a hash-matching
external/remote artifact backup plus non-reference for any candidate deletion.
The cleanup manifest therefore records `NO_DELETION_PERFORMED`.

No locked identity or outcome was accessed, no outer monitor was rescored, no
new model was trained, no mixed training/test table was downloaded, and the
publisher request was not resent. No external message was sent.

## Remaining blockers and next action

The immediate executable next action is to run the controlled compact replay in
the preserved artifact workspace. If and only if it reproduces the retained
seed-7 logits exactly, use its first-nonfinite-stage record to establish the
numerical failure location before changing preprocessing/model behavior.

Final model work remains blocked on a verified training-only NEW-crossing cohort,
source publication latency, licensing, complete episode semantics, faithful
same-cohort comparator reproduction, and an independent frozen evaluation path.
Once those are available, freeze the exact train-only chronological batch before
running climatology, eligible persistence, elastic net, XGBoost, reproduced
SEPNET, compact baseline, causal proton context, then XRS. The inspected monitor
and inner score blocks are not fresh evidence.

No final NEW-crossing improvement, operational readiness, economic savings,
breakthrough, or superiority over a named space company is established by this
continuation.
