# SOL continuation update — 2026-09-05

This update continues PR #3 on `codex/iris-sep-continuation-20260905` and does
not supersede the frozen benchmark/evaluation contracts. It records only the
bounded source hardening completed after the publication checkpoint.

## Follow-up: provenance-gated replay and expanded adversarial verification

A later continuation began from verified live remote head
`0aa786edca5942db324263b993770301e2a36088`; PR #3 remained open and unmerged.
The principal finding was not a new model-performance result but a reproducibility
gap in the standalone compact-failure replay: the original train-only diagnostic
had frozen helper/trainer/model/evaluation source hashes in its preregistration,
but the replay tool did not yet require those exact source revisions before
interpreting a replay mismatch or first nonfinite stage.

`tools/run_compact_nonfinite_replay.py` now fails closed unless the published
frozen diagnostic preregistration SHA-256 matches and every preregistered source
dependency hash matches. It additionally requires the retained model receipt to
bind seed 7, train-only fit/stop roles, exact fit/stop/predict indices, checkpoint
SHA-256 and preprocessing SHA-256; preprocessing feature order must match the
preregistered feature list. Only after those checks, the pinned train-only source
and retained failure-logit hashes, and exact saved-logit reproduction may the
layer replay identify a first nonfinite tensor stage. This prevents later helper,
model or preprocessing revisions from being silently substituted for the
original execution graph.

The exact fold-3 artifact replay remains **NOT_RUN** because the preserved
train-only CSV, folds, seed-7 logits, checkpoint, preprocessing and model receipt
are intentionally outside ordinary Git and were not present in the clean source
checkout. The retained failure remains 1,062 finite logits out of 2,120. No
causal attribution is made: helper mutation, float32 cast overflow and
distribution shift remain hypotheses until a hash-matched replay proves or
excludes them. Checkpoint parameter finiteness, feature support, train-fitted
scaling/cast behavior, missing masks, branch tensors, gates, shared tensors and
final logits are all audited by the controlled replay.

The Admission-V2 inference bundle adversarial matrix was also expanded. In
addition to existing array/source/calibration/threshold cases, tests now exercise
admission support-boundary mutation, model-version and schema mutation, embedded
evidence-receipt mutation, derived-probability mutation, and empty arrays. The
red-team document now states its independence boundary precisely: it is an
isolated bounded source/test workstream, not an external human or third-party
security audit. The externally retained bundle SHA-256 remains the trust anchor
for dynamic inference content, while the embedded static inference binding
prevents reuse of the same scientific receipt with a changed
policy/calibration/threshold/model/schema.

A repository-run source-only workflow verified tested source head
`b7d86e520d33cf55c74f7c8d88a1fed8904740e4`. GitHub Actions run `33975465640`
(job `101331265966`) used Ubuntu 24.04, Python 3.13.5, NumPy 2.3.5, PyTorch
2.10.0+cpu and scikit-learn 1.8.0. It ran 27 unittests in 0.196 seconds with zero
failures/errors and passed `py_compile` for the changed source/test files. See
`receipts/sol_continuation_source_only_v3_2026-09-05.json`.

Full artifact verification remains **NOT_RUN**: the clean checkout does not
contain the gitignored preserved datasets, model checkpoints, folds, notebooks,
caches or complete artifact dependency layout required by `verify_local`. No
replacement or mixed training/test table was downloaded to force it to run.

No locked identity/outcome was accessed, no outer monitor or inspected inner
score block was reused as fresh evidence, no new model was trained, the publisher
request was not resent, no external message was sent, and no cleanup deletion was
performed. No candidate deletion had both proof of non-reference and a verified
hash-matching recovery copy.

The exact next action is therefore unchanged in scientific substance but stricter
in provenance: run the compact replay in the preserved artifact workspace only
after restoring the preregistered implementation revisions if needed; accept a
first-nonfinite-stage conclusion only after every provenance gate and exact
saved-logit reproduction pass. Final model work remains blocked on verified
training-only NEW-crossing data, source latency/licensing, complete episode
semantics, faithful comparator reproduction and an independent frozen evaluation
path. With those available, freeze the bounded train-only chronological batch
before climatology, eligible persistence, elastic net, XGBoost, reproduced
SEPNET, compact baseline, causal proton context and then XRS.

No final NEW-crossing improvement, operational readiness, economic savings,
breakthrough, award outcome, or superiority over a named space company is
established by this follow-up.

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
