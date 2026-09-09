# Episode-normalized causal SEP benchmark — authoritative development result

**Result date:** 2026-09-10  
**Benchmark:** `IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`  
**Status:** `AUDIT_CORRECTED_AND_INDEPENDENTLY_VERIFIED — DEVELOPMENT EVIDENCE`

## Executive result

The benchmark tests whether ordinary 24-hour SEP occurrence scoring can mix three scientifically different effects:

1. forecasting a **new** >10 MeV, >=10 pfu threshold crossing;
2. recognizing that a threshold episode is **already active** at forecast time;
3. counting one physical SEP episode multiple times because it appears in multiple 24-hour windows.

The development result supports the mechanism, but not a universal literature-wide claim.

On the exposed 2014–2017 expanding out-of-fold cohort:

- score issue rows: **936**
- ordinary positive windows: **27**
- uniquely mapped physical positive episodes: **10**
- new-onset positive windows / distinct onset episodes: **5**
- already-active persistence windows: **11**
- ambiguous positive windows retained in full-window descriptive scoring: **11**
- mapped episode multiplicity factor: **1.6**

The onset arm is therefore **underpowered** and remains development-only.

## Main fixed-model point results

| Fixed model | Standard occurrence TSS | Episode-normalized occurrence TSS | New-onset TSS |
|---|---:|---:|---:|
| Current-proton-active diagnostic | 0.556 | 0.567 | 0.000 |
| Elastic-net joint | 0.642 | 0.765 | 0.665 |
| XGBoost joint | 0.657 | 0.443 | 0.043 |
| XGBoost XRS-only | 0.320 | 0.266 | 0.216 |
| Fit-prevalence climatology | 0.000 | 0.000 | 0.000 |

The simplest mechanism demonstration is the current-proton-active diagnostic: it obtains **TSS 0.556** under ordinary occurrence scoring with zero false positives, yet **TSS 0.000** on genuine new-onset opportunities. It recognizes persistence; it does not forecast new onset.

The joint XGBoost has the largest headline collapse: **0.657 standard occurrence TSS -> 0.443 episode-normalized occurrence TSS -> 0.043 new-onset TSS**.

## Physical-episode sensitivity and paired uncertainty

For inference, ambiguous positive windows are not silently assigned to physical episodes. Bootstrap claims use the separately persisted **uniquely mapped physical-episode sensitivity cohort** with identical shared draws across models and contrasts.

Ten thousand shared episode/quiet-block bootstrap draws were used.

### Multiplicity-only effect

For joint XGBoost:

- median TSS shift: **-0.133**
- 95% interval: **[-0.225, -0.041]**
- probability of a negative shift: **0.9883**

Thus the mapped development cohort supports a negative multiplicity effect for joint XGBoost.

### Persistence-removal effect

For joint XGBoost:

- median TSS shift: **-0.400**
- 95% interval: **[-0.800, -0.125]**

For the current-proton-active diagnostic:

- median TSS shift: **-0.567**
- 95% interval: **[-0.867, -0.267]**

These intervals exclude zero. Removing already-active persistence materially lowers the measured skill of these proton-aware occurrence predictors on this development cohort.

### Rank stability

Full-window point ranks change between standard and onset scoring.

For joint XGBoost versus XRS-only XGBoost, the mapped physical-unit bootstrap gives:

- median onset TSS difference (joint - XRS-only): **-0.168**
- 95% interval: **[-0.746, +0.244]**
- standard-to-onset ordering reversal fraction: **0.718**

Because the interval crosses zero, a statistically secure rank reversal is **not established**.

Elastic-net joint versus XRS-only has a positive mapped onset difference:

- median: **+0.432**
- 95% interval: **[+0.195, +0.894]**

## Public benchmark audit

A separate pinned audit of the public `yuyian/SEP-Prediction` benchmark is the higher-powered motivation arm.

Public rolling table:

- fixed 24-hour windows: **11,773**
- stored operational-positive rows: **1,726**
- rows with an independently reconstructed >=10 pfu episode overlap: **643**
- stored-positive rows without such overlap under the audited event-table semantics: **1,083**
- already-active persistence windows: **411**
- new-onset windows: **227**
- uniquely mapped positive windows: **610**
- physical episodes represented by those windows: **256**
- Episode Multiplicity Factor: **2.3828125**

This establishes a reproducible public-data target/multiplicity/persistence issue worth testing. It does **not** establish that the published paper's final SEPVAL score is wrong.

Pinned public audit receipt:

- upstream commit: `d0eb54e46b7dd6c760325e123d2ad86f9420fbff`
- workflow run: `34362893938`
- audit commit: `8b7ccf1fa2bc4aae20a3b12e0838270b12f8379b`
- artifact ID: `10108593291`
- artifact digest: `sha256:498104f66efb59ebd85795d628d799006f60266a8e3454bd2df3b127b988673d`

## Reproducibility receipt

Authoritative audit-corrected development workflow:

- workflow run: `34371418428`
- Git commit: `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact ID: `10112207048`
- artifact digest: `sha256:418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`

The workflow passed:

1. pinned-environment installation;
2. compile checks;
3. synthetic, property, compatibility and audit-correction tests;
4. frozen scientific-contract validation before data access;
5. frozen model execution;
6. post-result estimator correction without refitting or threshold reselection;
7. independent recomputation from persisted CSV/NPZ evidence;
8. evidence-hash finalization;
9. a second verification of the final evidence manifest;
10. immutable artifact upload.

`independent_verification_v2.json` passed.

## Post-result audit correction

The first successful artifact exposed two evidence-estimator defects during independent audit:

1. the concatenated expanding-OOF summary used one fold's threshold instead of the row-specific frozen thresholds already persisted in the prediction table;
2. the physical-episode bootstrap used a uniquely mapped cohort but the distinction from the full descriptive point cohort was not explicit enough.

Corrections were restricted to the estimator/evidence layer:

- **no model refit**
- **no feature change**
- **no cohort change caused by score inspection**
- **no threshold reselection**
- **no hyperparameter tuning**
- **no protected-outcome access**

The original result tables are preserved with `legacy_uncorrected_` names.

## Claim boundary

Supported:

> On the exposed development cohorts studied here, conventional window-level SEP occurrence evaluation can reward repeated representations of physical episodes and recognition of already-active storms, and these choices can materially change measured TSS and point model ranking.

Also supported:

> A forecast-evaluation framework that distinguishes new onset from persistence and gives each physical event equal positive statistical mass provides a more causally interpretable measurement of pre-onset warning skill.

Not supported:

- all published SEP forecasting results are inflated;
- the SEPNET published final SEPVAL score is wrong;
- a statistically secure rank reversal for joint versus XRS-only is established;
- the current 5-event onset cohort is sufficient for final external inference;
- state-of-the-art, operational superiority, certification, economic impact or award-outcome claims.

## Protected final-evidence rule

Post-`2025-09-10T00:00:00Z` candidate outcomes remain protected. They were not accessed by this development study. Final independent evaluation requires the already-frozen contract and custodian-controlled release; development-side code must not inspect those outcomes for power rescue, threshold selection, model selection or narrative tuning.

## Judge-facing one-sentence story

**Are SEP systems forecasting a new radiation storm, or are standard scores partly rewarding them for recognizing a storm that is already happening and counting the same physical event more than once?**
