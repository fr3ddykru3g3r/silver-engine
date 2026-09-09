# IRIS-SEP authoritative current status

**Status date:** 2026-09-10  
**Work branch:** `codex/iris-sep-episode-benchmark-v1-20260909`  
**Umbrella continuation:** `codex/iris-sep-continuation-20260905`  
**Purpose:** scientific source of truth. Historical inspected results remain development evidence and are never relabelled as untouched final evidence.

## Project in one sentence

**Are 24-hour SEP forecasting scores measuring prediction of a new radiation storm, or partly rewarding recognition of a storm that is already active and repeated counting of the same physical event?**

The centerpiece is now the episode-normalized causal evaluation benchmark, not a claim of a new state-of-the-art forecasting architecture.

## Current disposition

`EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1 — AUDIT_CORRECTED_AND_INDEPENDENTLY_VERIFIED_DEVELOPMENT_RESULT`

The methodological result is real and reproducible on exposed development data, but the internal new-onset cohort is small and final independent evidence has not been released.

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

The remaining weakness is statistical power for new onset. Do not disguise it.

## Immediate next actions

1. Publish the benchmark-result documentation and dedicated PR into the continuation branch.
2. Build the IRIS paper, synopsis, figures and 90-second explanation from the verified result.
3. Keep the protected final cohort sealed.
4. Seek a higher-powered **independent** confirmation only through a preregistered, non-tuned extension or external/public benchmark whose labels are already legitimately exposed.
5. Do not perform post-result architecture rescue on the five-event onset cohort.
