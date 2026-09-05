# Luna E red-team continuation: SEP evaluation and pilot integrity

Date: 2026-09-05  
Scope: daily probability of a **new >10 MeV, >=10 pfu SEP crossing in the next 24 hours**.  
Boundary: independent review only; no locked-test identities, outcomes, predictions, downloads, tuning, or external messages.

## Evidence reviewed

The active daily contract and evaluation policy are frozen in
`config/benchmark_contract_v2.json` and `config/evaluation_policy_v1.json`.
The current development-only status, legacy-label warning, and unresolved
publisher release are correctly recorded in `architecture/CONTINUATION_STATUS.md`,
`receipts/sepnet_reproduction_status_2026-09-05.json`, and
`receipts/v2_clear_training_access_status_2026-09-05.json`.

The synthetic pipeline and evaluation tests were run without touching protected
data:

```text
PYTHONPATH=. python3 -m unittest \
  iris_report.iris_sep.workstreams.luna_g_data_pipeline.test_pipeline_unittest \
  iris_report.iris_sep.workstreams.luna_i_eval_ops.test_eval_ops_unittest -v
```

Result: **15/15 passed**.  These tests establish useful local behavior, but they
do not prove that a future real-data run will satisfy the locked gate.

## Findings requiring action

### E1 — High: cohort-unit integrity is not fail-closed at the evaluator boundary

`build_cohort_units` verifies that every issue has one target and rejects
duplicate target records (`workstreams/luna_g_data_pipeline/iris_sep_pipeline/cohort.py:125-135`).
It does not verify that the returned units have disjoint issue IDs, that every
eligible issue occurs exactly once, that event and quiet units do not overlap,
or that each unit's stored label agrees with all of its member targets.  The
`CohortUnit` constructor only checks kind, interval ordering, and nonempty IDs
(`.../schemas.py:147-162`).  A hand-built or corrupted manifest can therefore
feed duplicate observations into the paired comparison while still passing the
current unit-level checks.

Action: add a pre-evaluation cohort audit that checks canonical issue identity,
one-and-only-one assignment across active roles and `purged`, no issue overlap,
unit label homogeneity, complete eligible-row coverage, and non-overlapping
unit intervals.  Bind the audit receipt and its hash to the partition/cohort
manifest.  Add synthetic tests for duplicate issue IDs across two units,
missing eligible IDs, mixed-label units, and an issue present in both a role and
`purged`.

### E2 — High: episode construction is a threshold-run heuristic, not an audited complete-episode definition

`_episode_runs` creates an episode from contiguous above-threshold observations
and closes it at any below-threshold observation (`cohort.py:77-107`).  It uses a
cadence-derived gap, but has no missing-observation, measurement-quality,
hysteresis, recovery, or source-specific episode semantics.  A one-sample dip or
gap can split one physical storm, while a source revision can change grouping.
Positive issue rows are then assigned to the first run containing the crossing
(`cohort.py:144-150`).  This is especially material because the frozen gate
requires complete SEP episodes to stay together and the bootstrap unit is an
episode or quiet block.

Action: require the publisher's authoritative episode IDs/algorithm and quality
flags in the training-only release.  Until then, label the local grouping as a
development fixture only.  On intake, produce a versioned episode manifest with
algorithm parameters, observation coverage/missingness, revision, and a
boundary audit showing every crossing maps to exactly one episode.  Include
adversarial tests for a one-sample dip, a missing interval, duplicate timestamps,
and adjacent episodes exactly at the merge boundary.

### E3 — High: the 24-hour purge is tested, but role assignment lacks a full coverage receipt

`assign_chronological_roles` correctly rejects units at the inclusive purge
endpoint and checks adjacent kept-role spacing (`cohort.py:227-246`).  However,
it slices provisional counts before purging and silently drops boundary units;
there is no assertion that all units are represented once in a role or in the
purged audit, nor a minimum event/quiet count per role.  An empty calibration,
threshold, or locked role can reach later code and fail late or produce an
underpowered result.  The current tests cover spacing and endpoint behavior,
but not role coverage or statistical power.

Action: serialize an explicit assignment table for every unit with one status
(`train`, `validation_monitor`, `validation_calibration`, `validation_threshold`,
or `purged`), reject duplicate/missing statuses, and fail before model fitting
if required roles lack both classes or predeclared minimum independent units.
Record the requested counts, retained counts, purge losses, class counts, and
power warning in the partition receipt.

### E4 — High: latency limits are mutable input fields rather than frozen policy

`FeatureRecord` accepts `max_latency_hours` from each caller and validates
against that value (`schemas.py:73-96`, `113-135`).  This allows a producer to
raise the limit per row and still pass validation.  The benchmark contract
requires a latency manifest, but the synthetic validator does not bind a
feature/modality limit, publication timestamp convention, or source revision
to a hashed manifest.

Action: remove caller authority over the limit in the real intake path.  Load a
frozen modality-specific latency policy from the signed feature manifest, reject
unknown source revisions and missing publication metadata, and receipt the
maximum observed age/latency by role.  Test an inflated row limit and a
publication timestamp exactly at and just after the issue time.

### E5 — High: independent locked evaluation is a convention, not an enforced API boundary

Calibration and threshold fitting require their expected roles
(`workstreams/luna_i_eval_ops/evaluation.py:42-65`, `159-172`), and direct
contract tests reject calibration/threshold misuse.  But `probability_metrics`,
`minimum_far_at_pod`, and `paired_unit_bootstrap_tss_difference` accept arbitrary
arrays and no role/phase or frozen-manifest token (`evaluation.py:115-156`,
175-204).  `assert_role_access` only checks a caller-supplied string and phase
(`src/iris_sep/contracts.py:98-110`); it is not wired into prediction loading
or metric execution.  A future orchestration error could score locked rows
during tuning without a library exception.

Action: make evaluation consume a sealed prediction bundle carrying role,
phase, contract hash, cohort hash, partition hash, and prediction hash; reject
locked roles unless phase is final evaluation and the model-selection receipt
is frozen.  Keep a separate explicit final-evaluation entry point.  Add a test
that attempts every metric and bootstrap function with a locked bundle during
tuning and expects a contract violation.

### E6 — Medium: paired bootstrap trusts `unit_ids` and does not validate unit semantics

The bootstrap resamples unique strings and skips replicates with one class
(`evaluation.py:175-204`).  It does not check that each unit is a predeclared
episode/quiet block, that a unit has one label, that issue rows are unique, or
that the supplied units match the frozen cohort manifest.  Consequently, a
caller can accidentally pass row IDs, mixed-label groups, or a different
grouping while still receiving a numeric interval.  The current positive test
only demonstrates that a supplied grouping changes the estimate.

Action: accept a typed unit manifest or validated mapping, enforce one label per
unit and exact cohort membership, reject row-level IDs, and receipt the number
of event and quiet units plus invalid-replicate reasons.  Keep the 95% valid
replicate requirement, and fail on class/power insufficiency before reporting a
confidence interval.

### E7 — Medium: the operator contract validates hash syntax, not evidence identity or uncertainty completeness

`build_operator_forecast` requires a 64-character evidence hash and matching
schema hash (`workstreams/luna_i_eval_ops/operator.py:76-95`), then emits the
hash without checking that the receipt exists, is the expected model/policy
receipt, or binds the input schema and calibration (`operator.py:108-131`).
The `uncertainty` mapping is copied without requiring the contract's
`between_seed_spread`, `calibration_uncertainty`, and `input_quality` components.
The result can therefore be marked `VALID` with a syntactically valid but
unrelated evidence hash or incomplete uncertainty record.

Action: resolve evidence hashes against an immutable receipt registry before
emitting `VALID`/`DEGRADED`; verify receipt type, model version, policy ID,
calibration ID, schema hash, cohort/role, and prediction hash.  Require the
three uncertainty components and finite bounds.  Any registry, binding, or
uncertainty failure must produce `ABSTAIN` with a deterministic reason.

### E8 — Medium: stale-feed handling lacks timestamp lineage and clock policy

The operator path checks only caller-provided numeric `age_minutes` against
per-modality limits (`operator.py:82-107`).  It does not validate the source
publication timestamp, the age calculation reference clock, clock skew, source
revision, or freshness of the evidence receipt itself.  Negative ages are
treated as stale, which is safe, but the provenance of a nonnegative age is
unverifiable.

Action: carry `source_time`, `publication_time`, `age_reference_time`, and
`source_revision` in each freshness record; recompute age inside the contract,
apply a frozen clock-skew bound, and reject future publication times.  Add
replays for future timestamps, clock skew, publication-after-issue, and a feed
whose receipt is older than the forecast issue.

### E9 — Medium: schema validation does not require complete modality presence

The operator policy requires all three modality names and identifies critical
modalities (`operator.py:28-38`), but `build_operator_forecast` allows any
noncritical modality to be absent from `missing_modalities`; it becomes
`DEGRADED` only if explicitly listed (`operator.py:82-107`).  A malformed input
that omits a modality from both `data_freshness` and `missing_modalities` is
treated as stale and abstains today, which is safe, but there is no explicit
schema-level distinction between an intentional unavailable feed and malformed
payload.  More importantly, the model-side feature validator checks duplicate
feature identities but does not require the expected feature names/modality
schema (`schemas.py:113-135`).

Action: require exact modality sets in every forecast request, separately mark
`MISSING` versus `SCHEMA_FAILURE`, and validate the ordered feature manifest and
modality membership before inference.  Add tests for omitted, duplicated, and
unknown feature names and for a noncritical feed omitted from the freshness map.

## Failure-path matrix for the next integration gate

| Path | Current local behavior | Independent-evidence requirement |
|---|---|---|
| stale feed | Numeric age over limit yields `ABSTAIN` | Recompute from signed publication timestamps and frozen clock policy |
| missing critical input | `ABSTAIN` with `CRITICAL_INPUT_MISSING` | Exact modality/schema validation and receipt of missingness |
| missing noncritical input | Explicit missing yields `DEGRADED`; omitted freshness becomes stale | Distinguish intentional missingness from malformed payload |
| schema mismatch | Hash mismatch adds `SCHEMA_FAILURE` and abstains | Verify ordered schema against the immutable feature manifest |
| missing/invalid evidence | `None` probability or caller reason abstains | Resolve evidence hash and verify model/policy/calibration bindings |
| duplicate issue | Duplicate issue records rejected in synthetic pipeline | Audit duplicate IDs across units, roles, and manifests |
| episode leakage | Basic threshold-run grouping and 24h purge | Publisher episode IDs, complete episode manifest, and boundary audit |
| locked access | Direct role helper rejects tuning call | Sealed bundle and evaluator-wide phase enforcement |
| underpowered role | No minimum class/unit count enforced | Predeclared power/count gate before fitting or scoring |

## Disposition

The current architecture is suitable for continued local engineering and
synthetic contract testing. It is **not yet independent-evaluation ready** and
does not support a final SEPVAL or superiority claim. The principal blockers
are the missing publisher training-only artifact and provenance metadata, the
unverified episode definition, and the absence of evaluator-wide sealed-bundle
enforcement. The existing development monitor remains development-only and
must not be relabeled as independent evidence.

Recommended order before any final training/evaluation: (1) obtain and hash the
training-only release with episode, target, units, latency, and exclusion
manifests; (2) implement cohort and sealed-bundle audits; (3) add stale,
missing, schema, evidence, and power replays; (4) freeze model selection and
only then open the locked evaluation once.
