# IRIS-SEP

IRIS-SEP is the availability-constrained, locked-benchmark continuation of the IRIS project. It
forecasts whether a new operational solar energetic particle threshold crossing
will occur within 24 hours and translates that forecast into calibrated,
operator-facing evidence. It does not control spacecraft.

The first deliverable is a same-cohort comparison against SEPNET-O. The model is
not a breakthrough unless every condition in the frozen paired benchmark gate
passes. The failed synthetic physics gate and the temporal next-magnetogram
simulation remain separate results.

Current source-of-truth files:

- `architecture/CONTINUATION_STATUS.md`: latest integrated evidence and blockers.
- `tests/verify_corrected_sepnet_v5.py`: hash-pinned development comparator
  verification; V5 is a corrected dense adapter, not a verified reproduction
  of published SEPNET-O. General-episode weighting was rejected after lower
  development TSS (0.074 versus 0.237 for row weighting).
- `config/benchmark_contract.json`: pre-data-access benchmark freeze.
- `config/benchmark_contract_v2.json`: active, evidence-driven daily-cadence
  amendment made before locked-test access; v1 is retained for audit history.
- `architecture/ADR-001-sepval-benchmark-freeze.md`: rationale and claim gate.
- `architecture/ARCHITECTURE_READINESS_AUDIT.md`: pre-training corrections and
  competition-readiness gates.
- `architecture/ADR-002-primary-first-operator-pilot.md`: narrowed validated
  product and company-facing safety boundary.
- `architecture/ADR-003-schema-aligned-tabular-primary.md`: header-evidenced
  decision to use aggregate feature branches rather than inventing a sequence.
- `architecture/ADR-004-daily-benchmark-cadence-amendment.md`: narrows the
  retrospective claim to the daily cadence actually supplied by the benchmark.
- `config/product_contract_v1.json`: fail-closed research-pilot output contract.
- `config/evaluation_policy_v1.json`: calibration, threshold, matched-detection,
  bootstrap, and non-degradation rules frozen before locked evaluation.
- `architecture/WINNER_READY_GATE.md`: evidence required before competition or
  operator claims.
- `src/iris_sep/contracts.py`: fail-closed integrity and role-access checks.
- `workstreams/`: isolated Luna research and implementation receipts.

No locked-test identities, outcomes, thresholds, or predictions belong in this
directory during tuning.

## Execution policy

Run metadata inspection, contract checks, unit tests, preprocessing smoke tests,
and small-model experiments locally first. Use Colab only when a measured local
constraint—GPU requirement, memory, runtime, or dataset transfer locality—makes
the run materially unsuitable for the local machine. Browser/computer-use
automation is not part of the routine workflow.

## Integrated checkpoint: 2026-09-05

- The SEP benchmark contract is frozen before primary-table or locked-test
  access.
- Luna A pinned the small provenance artifacts and recorded unresolved source
  disagreements without downloading the multi-gigabyte archive.
- Luna B froze leakage-safe baseline interfaces and source revisions. The
  authoritative local classical artifact is `artifacts/local_baselines_v4/`:
  it is a weak development-only floor on the publisher's legacy training label,
  not a SEPVAL or new-crossing result.
- Luna C implemented the compact availability-constrained three-expert model
  and restart-safe checkpoints. CPU forward, backward, deterministic replay,
  strict-mask, schema-binding, and checkpoint-restore tests pass. The safe V1
  five-seed development run completed locally in about 24 seconds, so spending
  a Colab GPU on that run is unnecessary.
- Luna D rejects AIA/HMI fusion in the current state because there is no
  authoritative AARP-to-HMI crosswalk. AIA auxiliary pretraining remains
  label-free and separate.
- No locked SEP test identity, outcome, prediction, threshold, or score has
  been accessed.

Current primary scope:

- one forecast per available daily benchmark window;
- probability of a new >10 MeV, >=10 pfu crossing in the following 24 hours;
- compact magnetic, eruption, and particle-context aggregate branches;
- primary output only; no AIA, HMI image embedding, auxiliary head, peak-flux,
  onset, >100 MeV, hourly-operation, or spacecraft-control claim.

The safe V1 development table does not contain proton/XRS context. Its neural
run therefore exercises magnetic and eruption branches while marking the
particle branch unavailable. A final three-branch claim requires the audited
V2/CLEAR training cohort and feature-latency manifest.

Final V2/CLEAR training is deliberately blocked until the publisher supplies a
training-only immutable artifact (or a controlled server-side export) with
version, units, lineage, new-crossing semantics, already-enhanced exclusions,
and publication-latency metadata. The available V2 delivery exposes a full
table/archive rather than a safe training-only artifact, and CLEAR version
lineage is unresolved. IRIS will not download that table and filter it locally,
because merely inspecting locked identities would violate the benchmark.
The official repository publishes no training-only split or opaque evaluator,
so `provenance/SEP_PRISM_TRAINING_ONLY_REQUEST.md` preserves the request text (the user reports it was sent; no reply supplied)
for publisher-side extraction, same-cohort comparison predictions, blinded
scoring, and explicit reuse terms.

For safe comparator development, the approved V1 dual-target cohort is
`data_processed/sepnet_v1_development_v6_dual_target.csv`. It preserves the
general training label, operational evaluation label, and maximum-flux target
while cryptographically binding every identity, role, schema, source, and
partition implementation to the selected V3 development cohort. It remains a
legacy-label development artifact, never final evidence.

Current development evidence is negative/inconclusive, as required to be
reported: the authoritative neural ensemble's validation-monitor TSS is 0.232
versus 0.257 for XGBoost, with a paired 95% unit-bootstrap interval for the TSS
difference of [-0.143, 0.103]. This role was used for early stopping, the label
is the publisher's legacy target, and none of these values are headline or
SEPVAL results. See `receipts/local_neural_selection_2026-09-05.json`.

A predeclared 50/50 probability blend of the neural ensemble and XGBoost raises
the monitor TSS point estimate to 0.276, but its paired 95% interval versus
XGBoost is [-0.072, 0.097] and its matched-detection false-alarm ratio is
slightly worse. It therefore remains an inconclusive development candidate,
not a winning result. The exploratory blend grid is retained and explicitly
marked selection-biased.

Pre-training hardening workstreams:

- `workstreams/luna_g_data_pipeline/`: synthetic-only cohort, causality,
  partition, train-only transformation, and immutable-manifest contracts.
- `workstreams/luna_h_model_hardening/`: primary-first loss, feature-mask,
  censoring, missing-modality, uncertainty, and abstention semantics.
- `workstreams/luna_i_eval_ops/`: role-safe calibration/threshold selection,
  metrics, paired unit bootstrap, and advisory operator forecast contract.

Run the dependency-free local integration gate with:

```bash
python3 iris_report/iris_sep/tests/verify_local.py
```

The Colab development package is
`colab/IRIS_SEP_Development_Training_2026-09-04.ipynb`. It embeds the exact
trainer and tests, downloads only the pinned publisher training file, resumes
five-seed checkpoints from Drive, and refuses test-like roles. It is built
locally and is retained for reproducibility or future heavier runs; the V1
development training itself has already completed locally.
