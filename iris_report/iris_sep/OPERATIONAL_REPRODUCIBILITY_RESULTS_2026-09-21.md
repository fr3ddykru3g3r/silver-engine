# Operational reproducibility audit — results

**Study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Execution date:** 21 September 2026  
**Status:** `PHASE_A_COMPLETE — PHASE_B_BLOCKED_BY_PREREGISTERED_INTERFACE_GATE`

## Research question

> Can the frozen retrospective 24-hour SEP forecasting interface be reproduced using only predictor values demonstrated to have an equivalent form available before forecast issue time?

## Frozen evidence

The audit did not rediscover features from filenames or a later model script. It used the exact ordered 259-predictor schema stored in the immutable external-model replay artifact.

| Item | Frozen value |
|---|---|
| Predictor count | 259 |
| Ordered predictor-list SHA-256 | `cf0fc9e07b1e9b173ad0c330fb021452b5a527ab796dfdbd9282c29a5e3047d4` |
| Artifact `feature_schema.json` SHA-256 | `b70c1b9137cfe7493787f8ddcc328ce81e1153314bbcade8d12013c94948fcf1` |
| Upstream 24-hour table SHA-256 | `4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb` |
| Artifact digest | `sha256:81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0` |
| Upstream revision | `yuyian/SEP-Prediction-V2@e138dcd72c1952a00e11e1a0b025337f9e7c93fb` |

The 274-column archived table reproduces the same 259 predictors exactly when the original replay exclusions are applied: exclude `window_begin`, `window_end`, `OSEP_label`, `GSEP_label`, and all `Future_*` fields. Present-time fields such as `SHARP_label` and `ProtonFlux_label` remain predictors.

## Main result

| Status | Predictor count | Fraction of 259 | Interpretation |
|---|---:|---:|---|
| `VERIFIED` | 0 | 0.00% | No exact archived predictor currently satisfies every preregistered issue-time-equivalence criterion. |
| `SCHEMA_MISMATCH` | 222 | 85.71% | An operational/NRT physical source may exist, but the retrospective feature construction is not the same issue-time object. |
| `RETROSPECTIVE_ONLY` | 26 | 10.04% | The frozen feature explicitly depends on retrospective cataloguing/reconstruction/imputation. |
| `UNVERIFIED_LATENCY` | 11 | 4.25% | Historical issue-time availability is unresolved rather than disproven. |

Therefore:

- **248 / 259 = 95.75%** of the exact frozen predictors are already directly tied to a construction classified as schema-mismatched or retrospective-only.
- **11 / 259 = 4.25%** remain unresolved (the flare family).
- **0 / 259** currently meet the stricter `VERIFIED` exact-equivalence standard.
- Even if all 11 unresolved flare predictors later pass, the maximum verified fraction for this **unchanged exact interface** would be **4.25%**; the remaining 248 would still require the feature construction itself to be replaced/rebuilt.

### Family decomposition

| Feature family | Count | Status |
|---|---:|---|
| SHARP | 107 | `SCHEMA_MISMATCH` |
| SHARP_AR | 107 | `SCHEMA_MISMATCH` |
| Flare | 11 | `UNVERIFIED_LATENCY` |
| DONKI_CME | 12 | `RETROSPECTIVE_ONLY` |
| CDAW_CME | 14 | `RETROSPECTIVE_ONLY` |
| ProtonFlux | 4 | `SCHEMA_MISMATCH` |
| XRS | 4 | `SCHEMA_MISMATCH` |

The result summary is machine-readable in `audit_operational_reproducibility_20260921/interface_audit_v1.json`; per-feature rows are deterministically regenerated from the frozen schema, manifest and audit tool.

## Physical reason the result matters

Forecast causality is simple: if a forecast is issued at time `t`, the predictor vector may only contain information that could exist by `t`.

The audit found four different ways a retrospective scientific data product can fail to be identical to an operational input:

1. **future-aware product definition:** definitive SHARP/HARP geometry uses the active region's complete lifetime;
2. **retrospective reconstruction:** SHARP-from-SMARP and DONKI-like CME features are statistically reconstructed across historical eras;
3. **two-sided gap filling:** linearly interpolated proton/XRS values may use samples after the missing timestamp;
4. **catalogue latency/revision:** an event property can describe something that physically happened before `t` but only be catalogued or finalized after `t`.

Those are data-product causality issues, not claims about the laws of solar physics.

## Why the headline is not "0% real-time data"

That statement would be wrong. GOES proton and X-ray measurements have operational streams. HMI provides NRT SHARP data. DONKI supports operational space-weather analysis. The issue is that **the archived feature values used in the frozen retrospective interface are not automatically identical to those live products**.

The defensible headline is:

> In the frozen 259-input interface, 248 predictors (95.75%) use a construction that is already demonstrated to be retrospective or non-equivalent to the corresponding issue-time/NRT construction; the remaining 11 require historical latency evidence. No predictor currently meets every criterion for exact issue-time equivalence as archived.

## Phase B decision: do not manufacture a replay score

The preregistration requires the operational replay to use only predictors with demonstrated issue-time availability **and** matching field/schema semantics. It explicitly forbids replacing a structurally unavailable feature with zero, a later definitive value, or a similarly named NRT field and then calling it the same model.

Because the exact interface contains no `VERIFIED` columns under the frozen evidence rule, an identical-schema 259-feature causal replay is not legitimate. No reduced causal model was frozen before inspecting outcomes either.

**Therefore Phase B is stopped at the preregistered interface gate.** This is the experimental result of the gate: the existing retrospective interface must first be rebuilt as a distinct causal/NRT interface before operational skill can be measured honestly.

Protected outcomes from `2025-09-10T00:00:00Z` onward remain sealed and were not used to choose sources, statuses, tolerances, features, thresholds or narrative.

## Code correction found during execution

The first audit implementation incorrectly treated every `*_label` field as an outcome. The immutable schema showed that present-time indicators such as `SHARP_label`, `Flare_label`, `ProtonFlux_label` and `XRS_label` are part of the 259 predictor interface.

The audit was corrected **before a result was accepted**. The corrected rule uses the frozen schema directly; when extracting from the raw table it removes only the exact non-predictors and `Future_*` fields specified by the replay contract. Regression tests now protect this invariant.

Local test result for the corrected audit package:

```text
4 passed in 0.09s
```

## What follows scientifically

The next experiment is not to force the original model to run. It is to build and freeze a **new causal interface** using only near-real-time products and one-sided preprocessing, then validate that interface separately.

1. **Product-equivalence experiment:** matched-time NRT versus definitive SHARP/GOES fields, with tolerances frozen before viewing the error distribution.
2. **Causal forecasting experiment:** train a separately declared NRT-only model with one-sided gap handling and archived issue-time snapshots, then evaluate it prospectively or on a predeclared historical causal cohort.

Any skill score from that model belongs to the **new causal model**, not to the original 259-feature retrospective interface.

## Claim limits

Allowed: the exact frozen interface is not demonstrated to be operationally reproducible; 248/259 exact predictors have a direct retrospective/schema-mismatch mechanism in the pinned construction; 11/259 remain unresolved under the evidence standard; a separate causal interface must be rebuilt before operational forecast skill is claimed.

Not allowed: the original paper is wrong or invalid; 95.75% of underlying physical measurements were unavailable in real time; the model necessarily leaks future SEP outcomes; an NRT-rebuilt model would have poor skill; this is the first such audit ever performed.
