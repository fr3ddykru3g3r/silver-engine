# IRIS-SEP authoritative current status

**Status date:** 2026-09-10  
**Work branch:** `codex/iris-sep-episode-benchmark-v1-20260909`  
**Umbrella continuation:** `codex/iris-sep-continuation-20260905`  
**Purpose:** scientific source of truth. Historical inspected results remain development evidence and are never relabelled as untouched final evidence.

## Project in one sentence

**Are 24-hour SEP forecasting scores measuring prediction of a new radiation storm, or partly rewarding recognition of a storm that is already active and repeated counting of the same physical event?**

The centerpiece is now the episode-normalized causal evaluation benchmark, not a claim of a new state-of-the-art forecasting architecture.

## Current disposition

`SEP_PRISM_FIXED_MODEL_REPLAY_V1 — STRONG_MODEL_CONFIRMATION_ON_PREVIOUSLY_EXPOSED_PUBLIC_DATA`

The evaluation effect now reproduces in a preregistered, higher-powered chronological replay on public historical data. The source hashes were already recorded as development-inspected, so this is strong methodological confirmation rather than untouched final evidence. The protected post-2025 cohort remains sealed.

### SEP-PRISM model-free confirmation

The pinned 1986–2025 table contains 14,464 daily windows and 650 stored positives, with zero stored-versus-reconstructed target mismatches. The onset-state audit finds 418 already-active persistence windows, 228 genuine new-onset windows and four ambiguous positives. The Episode Multiplicity Factor is **614 uniquely mapped positive windows / 257 represented physical episodes = 2.389**, because 36 multi-episode-overlap positives are excluded from that calculation. Persistence exceeds onset in 1986–2010, 2011–2017 and 2018–2025; their respective EMFs are 2.46, 2.52 and 1.95.

### SEP-PRISM frozen fixed-model replay

The replay covers 7,558 unique daily score issues across three chronological out-of-fold periods from 2005–2025. Six fixed comparators were evaluated under unchanged alerts. The matched inferential cohort contains 85 onset episodes and 1,080 Monday-anchored quiet blocks, with 10,000 shared physical-unit bootstrap draws.

| Frozen directional TSS contrast | Point change | Paired 95% interval |
|---|---:|---:|
| Joint XGBoost: episode-normalized minus mapped occurrence | -0.105 | [-0.146, -0.063] |
| Joint XGBoost: onset minus episode-normalized occurrence | -0.184 | [-0.247, -0.125] |
| Past-proton proxy: onset minus mapped occurrence | -0.508 | [-0.567, -0.444] |

All three intervals lie below zero, satisfying the frozen `STRONG_MODEL_CONFIRMATION` rule. On the same matched population, joint-XGBoost TSS changes **0.726 → 0.621 → 0.437** across mapped occurrence, episode normalization and onset. Its onset confusion table is TP=41, FN=44, FP=330 and TN=6,984 (sensitivity 48.24%, FAR 88.95%), so the result is evidence about evaluation sensitivity, not operational readiness.

The joint-XGBoost minus no-proton onset contrast is -0.040 with interval [-0.163, +0.083], which does not establish a reliable rank reversal. Elastic-net's corresponding contrast is positive, but both elastic-net onset TSS values are negative. No broad proton-feature or model-superiority claim is supported.

### External public-data arm

A pinned audit of `yuyian/SEP-Prediction` found:

- 11,773 24-hour windows;
- 1,726 stored operational positives;
- 1,083 stored positives that do not overlap a reconstructed >=10 pfu operational episode under the audited event-table semantics;
- 411 already-active persistence windows versus 227 new-onset windows;
- 610 uniquely mapped positive windows representing 256 physical episodes;
- Episode Multiplicity Factor **2.3828125**.

This motivates the benchmark. It does not prove the paper's final SEPVAL score is wrong.

### Internal fixed-model development arm

Exposed 2014–2017 expanding OOF cohort:

- 936 scored issue rows;
- 27 standard positive windows;
- 10 uniquely mapped positive physical episodes;
- 5 distinct new-onset episodes;
- 11 persistence windows;
- 11 ambiguous positive windows;
- mapped Episode Multiplicity Factor **1.6**.

Headline TSS:

| Model | Standard | Episode-normalized occurrence | New onset |
|---|---:|---:|---:|
| Current-proton-active diagnostic | 0.556 | 0.567 | 0.000 |
| Elastic-net joint | 0.642 | 0.765 | 0.665 |
| XGBoost joint | 0.657 | 0.443 | 0.043 |
| XGBoost XRS-only | 0.320 | 0.266 | 0.216 |

Shared mapped-episode bootstrap, 10,000 draws:

- joint-XGBoost multiplicity-only TSS shift median **-0.133**, 95% **[-0.225, -0.041]**;
- joint-XGBoost persistence-exclusion shift median **-0.400**, 95% **[-0.800, -0.125]**;
- current-proton-active persistence-exclusion shift median **-0.567**, 95% **[-0.867, -0.267]**;
- joint-vs-XRS-only onset difference median **-0.168**, 95% **[-0.746, +0.244]**: point reversal observed, statistically secure reversal not established.

The onset arm has only five positive episodes and is explicitly underpowered.

## Authoritative evidence receipt

Final source-only compatibility verification:

- workflow run `34509843220`
- commit `03c912842652e7a4ef4c273ebaff9555fa920bc7`
- pandas 2.3.2: PASS
- pandas 3.0.1: PASS
- both jobs passed environment setup, full Python compilation, every registered source-only test, JSON parsing and data-dependent-test inventory checks

The pandas 3 compatibility fixes explicitly normalize timestamps to nanoseconds in feature-window indexing and physical-episode gap detection. They do not refit models, reselect thresholds, alter persisted predictions, reopen the negative freshness experiment or access protected outcomes.

Audit-corrected development workflow:

- run `34371418428`
- commit `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact `10112207048`
- artifact SHA-256 `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`

The workflow passed compile/tests, frozen-contract validation, benchmark execution, independent evidence recomputation, final hash generation, a second manifest verification and artifact upload.

Read:
`architecture/EPISODE_NORMALIZED_CAUSAL_BENCHMARK_RESULT_2026-09-10.md`

## Post-result correction boundary

Independent audit found an OOF threshold-summary bug and an estimand-labeling ambiguity in the first successful artifact.

The correction changed **only** the result/evidence layer:

- no refit;
- no threshold reselection;
- no feature change;
- no hyperparameter search;
- no protected-outcome access.

Legacy uncorrected tables remain preserved in the artifact.

## Protected evidence rule

`config/inspected_evidence_registry_v2.json` remains authoritative.

Post-`2025-09-10T00:00:00Z` candidate outcomes remain protected. Development-side work must not query, count, inspect, score or stratify them, including for power rescue.

The current result is **development/methodology evidence**, not independent final evidence.

## Historical V3 reliability engineering — preserved, not the centerpiece

The frozen V3 work remains valid historical evidence.

**Status date:** 2026-09-08

The V3 model remains **RETROSPECTIVE** for skill because the exact prospective interface cannot be reproduced. `CMASKL`, `MEANGBL`, and `USFLUXL` are unavailable in the required near-real-time interface, affecting **18 frozen feature-vector positions**. The prospective fail-closed run emitted **no forecast probability**.

This historical fact must not be mixed with the new episode-normalized model-score result.

The V3 `NO_XRS_OR_PROTON` state remains `ABSTAIN`, with no alert. Prior post-hoc alert filters remain rejected because false-alert reductions came with lost detections.

## Scientific claim boundary

Supported:

- repeated physical-event representation exists in a current public SEP rolling benchmark;
- already-active persistence can score highly under occurrence evaluation while providing no new-onset detections;
- in the exposed development cohort, multiplicity and persistence materially affect some fixed-model TSS values;
- point model ranking can change under causal/onset evaluation;
- physical-episode weighting and onset/persistence separation are scientifically justified.

Not supported:

- all prior SEP papers are wrong or biased;
- SEPNET's final published SEPVAL score is wrong;
- independent prospective superiority;
- operational readiness/certification;
- state-of-the-art model performance;
- economic savings;
- award outcome.

## 9/10 competition target

The project now has a credible high-level competition structure:

- **Scientific thought:** falsifiable mechanism + causal estimands + paired physical-unit uncertainty.
- **Creativity:** benchmark contribution rather than another classifier.
- **Thoroughness:** public external audit, attrition ledger, multiple fixed comparators, preserved negative/legacy evidence, shared bootstrap tensors.
- **Skill:** pinned environment, immutable receipts, model serialization, independent replay/recomputation, double hash verification.
- **Clarity:** one question — “forecasting a new storm, or recognizing one already happening?”

The earlier five-onset internal limitation has been substantially reduced by an 85-episode matched replay. Remaining weaknesses concern evidence independence, causal feature availability, missing model/threshold-search objects, and prospective operational validation. Do not disguise them.

## Immediate next actions

1. Keep PR #5's verified replay documentation and final green CI receipt synchronized.
2. Preserve the replay archive digest, independent receipts and exact matched-population denominators.
3. Keep the protected final cohort sealed.
4. Obtain a custodian-controlled prospective confirmation only after freezing an operational feature-availability contract.
5. Do not tune or rescue the frozen replay after score inspection.
