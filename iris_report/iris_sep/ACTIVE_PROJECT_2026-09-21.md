# Active IRIS project — operational reproducibility of SEP forecasting

**Date frozen:** 21 September 2026  
**Active branch:** `codex/iris-sep-operational-reproducibility-20260921`  
**Study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Current state:** `PHASE_A_COMPLETE — PHASE_B BLOCKED BY PREREGISTERED INTERFACE GATE`

## Research question

**Can a retrospective 24-hour solar energetic particle (SEP) forecasting interface be reproduced using only predictor values that were genuinely available in an equivalent form before each forecast issue time?**

## One-sentence result

**Of the frozen model's 259 exact predictors, 248 (95.75%) use a construction that is already shown to be retrospective or non-equivalent to the corresponding issue-time/NRT construction; the remaining 11 (4.25%) still lack historical latency proof, so the exact 259-input interface cannot yet be replayed causally without changing the model input definition.**

This does **not** mean 95.75% of the underlying physical measurements were unavailable. It means the archived feature values and the live values are not the same data product/construction.

## Why this is Physics, not just software auditing

The model inputs represent physical states of the Sun and near-Earth particle environment: magnetic active-region quantities from HMI/MDI; flare timing and X-ray strength; coronal mass-ejection geometry/speed; >10 MeV proton flux; and soft X-ray flux.

The physical principle being enforced is causality: at forecast issue time `t`, no input may depend on measurements, region evolution, catalogue decisions or interpolation samples that only become known after `t`.

The most intuitive example is definitive SHARP geometry. Definitive HARP/SHARP products can use the active region's complete lifetime to define the region consistently. That is excellent for science, but a forecaster at the beginning of that lifetime does not know the future evolution yet. The NRT product is therefore a different operational object.

## Frozen interface

The project is not auditing a guessed feature list. The 259 predictors come from the immutable external-model replay artifact:

- artifact id `10137507101`;
- artifact digest `sha256:81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`;
- ordered predictor-list SHA-256 `cf0fc9e07b1e9b173ad0c330fb021452b5a527ab796dfdbd9282c29a5e3047d4`;
- pinned upstream revision `yuyian/SEP-Prediction-V2@e138dcd72c1952a00e11e1a0b025337f9e7c93fb`.

The exact list is in `config/frozen_joint_feature_schema_v1.json`.

## Phase A — completed

| Family | Count | Status |
|---|---:|---|
| SHARP | 107 | `SCHEMA_MISMATCH` |
| SHARP_AR | 107 | `SCHEMA_MISMATCH` |
| Flare | 11 | `UNVERIFIED_LATENCY` |
| DONKI_CME | 12 | `RETROSPECTIVE_ONLY` |
| CDAW_CME | 14 | `RETROSPECTIVE_ONLY` |
| ProtonFlux | 4 | `SCHEMA_MISMATCH` |
| XRS | 4 | `SCHEMA_MISMATCH` |

Summary: exact predictors audited **259**; directly retrospective/schema-mismatched **248 (95.75%)**; unresolved **11 (4.25%)**; fully `VERIFIED` under the strict exact-equivalence rule **0**.

## Why those families fail the exact-equivalence test

1. **SHARP/SHARP_AR:** definitive and NRT SHARP are not identical products; definitive geometry can use full region history. The upstream fusion also uses nearest-time imputation and SHARP-from-SMARP statistical reconstruction.
2. **DONKI/CDAW CME:** the archived fusion uses kNN imputation, cross-catalog event pairing and regression reconstruction of DONKI-like fields from CDAW.
3. **Proton flux:** the archived series stitches satellites, performs linear interpolation and fills gaps from HAPI archival values.
4. **XRS:** the archived series stitches satellites and linearly interpolates gaps; NOAA explicitly distinguishes operational from retrospectively reprocessed science-quality XRS.
5. **Flare:** the physical information is plausibly operational, but the audit has not yet established historical first-seen/revision timing for every exact archived HEK field, so it remains unresolved rather than failed.

## Phase B — deliberately not forced

The preregistration says an operational replay may use only exact predictor values with demonstrated issue-time availability and schema equivalence. It forbids later definitive substitution, zero-filling structural absence, or replacing a feature with a merely similar NRT field while calling it the same model.

Because the unchanged 259-input interface does not pass that gate, **there is no honest same-model operational skill score to report yet**. The next step must be a separately declared NRT/causal interface and model. Its performance would answer a new question rather than retroactively certify the old interface.

## Protected outcome rule

All post-`2025-09-10T00:00:00Z` protected outcomes remain sealed. They were not used to choose source statuses, latency assumptions, feature mappings, thresholds or the narrative.

## Novelty boundary

The project does not claim that latency, data gaps, NRT products, or pre-event cutoffs are new. The bounded novelty claim is the **pinned complete-interface + code-lineage + fail-closed quantitative audit + preregistered replay gate** applied to this modern multi-source SEP forecasting system. Current confidence is moderate and the project carries an explicit kill condition if an exact prior study is found.

## Plain-language judge explanation

> A forecast can only use information that actually existed when it was made. I froze all 259 inputs of a recent solar-radiation forecast and traced how each one was built. I found that 248 are constructed differently from what a forecaster could have had live, while 11 still need historical timing proof. So instead of pretending the old model can be replayed in real time, the experiment tells us exactly what must be rebuilt before its operational performance can be tested fairly.

## Working title

**Can a “Real-Time” Solar-Radiation Forecast Be Reproduced in Real Time? A Causal Audit of 259 SEP Forecast Inputs**
