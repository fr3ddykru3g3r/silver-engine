# SOL continuation prompt

Continue IRIS-SEP from the **episode benchmark result**, not from the older freshness or V3 modeling loops.

## Branches

Primary benchmark branch:

`codex/iris-sep-episode-benchmark-v1-20260909`

Umbrella continuation branch:

`codex/iris-sep-continuation-20260905`

PR #3 is the umbrella continuation PR and must not be merged automatically.

Always inspect the exact branch head and current PRs before writing.

## Central research question

> Are 24-hour SEP forecasting systems forecasting a new radiation storm, or do ordinary scores partly reward recognition of already-active storms and repeated counting of one physical SEP episode?

Working benchmark:

`IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`

Read first:

- `CURRENT_STATUS.md`
- `architecture/EPISODE_NORMALIZED_CAUSAL_BENCHMARK_RESULT_2026-09-10.md`
- `architecture/PUBLISHED_SEPNET_TARGET_AND_EPISODE_AUDIT_RESULT_2026-09-09.md`
- `architecture/EPISODE_BENCHMARK_POST_RESULT_AUDIT_CORRECTION_2026-09-09.md`
- `config/episode_normalized_causal_benchmark_v1_preregistration_2026-09-09.json`
- `config/episode_normalized_causal_benchmark_v1_mechanism_decomposition_2026-09-09.json`
- `config/episode_normalized_causal_benchmark_v1_expanding_oof_2026-09-09.json`
- `config/inspected_evidence_registry_v2.json`

## Authoritative benchmark receipt

Audit-corrected workflow:

- run `34371418428`
- commit `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact ID `10112207048`
- artifact digest `sha256:418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`

The workflow passed:

- pinned environment;
- compile checks;
- synthetic/property/compatibility/audit tests;
- frozen scientific-contract gate before data access;
- frozen model execution;
- audit-corrected result estimator;
- independent V2 verification from persisted CSV/NPZ evidence;
- final evidence-hash generation;
- second manifest verification;
- immutable artifact upload.

## Main development result

2014–2017 expanding OOF:

- 936 rows;
- 27 standard positives;
- 10 uniquely mapped physical positive episodes;
- 5 new-onset episodes;
- 11 persistence windows;
- 11 ambiguous positive windows.

TSS:

- current-proton-active diagnostic: `0.556 standard -> 0.000 onset`
- elastic-net joint: `0.642 -> 0.665`
- XGBoost joint: `0.657 -> 0.043`
- XGBoost XRS-only: `0.320 -> 0.216`

Mapped 10,000-draw shared bootstrap:

- joint XGB multiplicity effect: median `-0.133`, 95% `[-0.225,-0.041]`
- joint XGB persistence exclusion: median `-0.400`, 95% `[-0.800,-0.125]`
- current-proton-active persistence exclusion: median `-0.567`, 95% `[-0.867,-0.267]`
- joint minus XRS-only onset: median `-0.168`, 95% `[-0.746,+0.244]`; rank reversal is not statistically secure.

Do not hide the five-event onset limitation.

## Public benchmark audit

Pinned `yuyian/SEP-Prediction` audit:

- 11,773 windows;
- 1,726 stored operational positives;
- 1,083 stored positives with no >=10 pfu episode overlap under the audited event-table semantics;
- 411 persistence vs 227 onset windows;
- 610 mapped positive windows / 256 physical episodes;
- multiplicity factor `2.3828125`.

This is motivation/methodology evidence. Do not claim the published final SEPVAL score is wrong.

## Post-result correction boundary

The first successful artifact had:

1. an OOF summary bug using one threshold rather than persisted row-specific frozen thresholds;
2. a bootstrap estimand that needed to be explicitly separated from the full descriptive cohort.

Correction was evidence-layer only:

- no refit;
- no feature changes;
- no threshold reselection;
- no hyperparameter tuning;
- no protected outcome access.

Legacy result files remain preserved.

## Protected outcome rule

Post-`2025-09-10T00:00:00Z` candidate outcomes remain sealed.

Do not query, count, inspect, score, stratify or otherwise reveal them.

Do not use protected data for power rescue, model selection, threshold changes or narrative tuning.

## Historical V3 evidence

Preserve it.

The exact prospective V3 interface remains blocked because `CMASKL`, `MEANGBL`, and `USFLUXL` are missing, affecting 18 frozen feature-vector positions. The prior preflight emitted no forecast probability. That is a valid fail-closed engineering result but no longer the research centerpiece.

## Next legitimate work

1. Finalize the dedicated episode-benchmark PR into the continuation branch; do not merge automatically.
2. Build the IRIS research paper and judge-facing package from verified results.
3. Generate figures only from authoritative corrected files.
4. Keep the protected final cohort sealed.
5. If increasing statistical power, use only a preregistered independent extension/public dataset and freeze it before score inspection.
6. No post-hoc architecture rescue on the 5-event onset cohort.

## Claim to use

> On exposed development data, conventional window-level SEP occurrence scoring can reward repeated representations of physical episodes and recognition of already-active storms. A physical-episode, new-onset evaluation changes measured skill for some fixed models and yields a more causally interpretable estimate of pre-onset warning ability.

Do not strengthen this into a universal literature claim without independent confirmation.
