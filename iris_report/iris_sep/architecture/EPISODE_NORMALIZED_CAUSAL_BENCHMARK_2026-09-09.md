# Episode-normalized causal SEP benchmark — scientific design

**Design status:** PREREGISTRATION CANDIDATE / NO PROTECTED OUTCOMES ACCESSED  
**Date:** 2026-09-09  
**Working system name:** `IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`

## 1. Central research question

How much do repeated positive windows from the same physical SEP episode, persistence/already-active states, and issue-time proton-history information alter the apparent performance and model ranking of 24-hour SEP forecasts?

The benchmark does **not** assume that conventional metrics are inflated. It tests that hypothesis.

## 2. Motivation

Recent state-of-the-art SEP forecasting work uses fixed 24-hour predictor windows and acknowledges that one physical SEP event can persist for longer than 24 hours and therefore contribute multiple positive prediction windows. This creates a potential mismatch between the statistical unit used by ordinary window metrics and the physical unit of interest: a distinct radiation-storm episode.

Operationally, a warning that a new event is about to begin is also different from persistence of an event already underway. NOAA/SWPC explicitly distinguishes onset warnings from persistence warnings. The scientific question here is whether mixing or repeatedly weighting these states changes measured forecast skill.

## 3. Hypotheses

### H1 — event multiplicity

Window-level positive counts over-weight long-duration SEP episodes relative to short episodes. Under episode-normalized weighting, at least one fixed model's measured TSS or matched-detection FAR will change materially.

### H2 — rank instability

The ordering of at least two fixed model families will differ between standard window-level occurrence scoring and episode-normalized new-onset scoring.

### H3 — proton-state dependence

Removing direct issue-time/in-window proton-state features will reduce onset-skill more than persistence-skill, showing that historical particle measurements encode different information in the two regimes.

### H0 / falsification

The study is scientifically informative if all three hypotheses fail. A null result means standard window scoring is empirically robust to episode normalization for the tested frozen models/cohorts. No architecture tuning follows a null result.

## 4. Units of analysis

### 4.1 Window

A frozen issue timestamp with a 24-hour predictor interval ending at issue time and a 24-hour outcome interval after issue time.

### 4.2 Physical SEP episode

A continuous threshold-exceedance episode under the frozen event semantics. Episode identity must be constructed once from the outcome flux series and remain independent of model prediction.

### 4.3 New-onset eligible issue

An issue is new-onset eligible only when:

- proton flux at issue time is below the frozen operational threshold;
- no unresolved observation gap invalidates the state determination;
- the outcome window is fully mature and observable;
- the first qualifying crossing in the outcome window belongs to a new episode rather than continuation of an already-active episode.

Already-active cases are never treated as onset positives.

## 5. Five frozen evaluation views

### A. `WINDOW_OCCURRENCE_STANDARD`

Literature-compatible reference. Each eligible 24-hour window has unit weight.

### B. `NEW_ONSET_CAUSAL`

Only causally onset-eligible windows. Each window still has unit weight.

### C. `EPISODE_NORMALIZED_ONSET`

Same cohort as B, but positive weights are normalized so that for every physical SEP episode e:

`sum_i w_i(e) = 1`

where i ranges over eligible positive issue windows associated with episode e.

This prevents a long episode from contributing more total positive mass only because it persists across more candidate windows.

### D. `PROTON_STATE_BLIND_ONSET`

Same onset eligibility, using separately fitted frozen models that exclude direct proton-history features. This is a mechanism test, not a production replacement.

### E. `PERSISTENCE_DIAGNOSTIC`

Already-active issues are isolated and reported separately. These cases can quantify continuation recognition but cannot increase onset skill.

## 6. Primary metrics

Primary:

- TSS under each evaluation view;
- episode-level POD;
- FAR at matched episode-level POD;
- model rank by TSS;
- `N_positive_windows / N_distinct_positive_episodes` multiplicity ratio.

Secondary:

- HSS;
- AUROC;
- AUPRC;
- Brier score;
- calibration slope/intercept;
- reliability curve;
- warning lead time for correctly detected new episodes.

No single point metric establishes the claim.

## 7. New benchmark quantities

### 7.1 Episode Multiplicity Factor (EMF)

`EMF = number_of_positive_windows / number_of_distinct_positive_episodes`

EMF = 1 means no positive episode is represented more than once. EMF > 1 quantifies repeated representation but is not itself evidence of bias.

### 7.2 Episode-Normalization Shift (ENS)

For metric M:

`ENS_M = M_episode_normalized_onset - M_window_occurrence_standard`

The sign is not preregistered. The scientific question is magnitude, uncertainty, and cross-model consistency.

### 7.3 Rank Stability Matrix

For every model pair (a,b), record whether the ordering under standard scoring agrees with the ordering under episode-normalized onset scoring. Report the full matrix; no cherry-picked pair.

## 8. Models — fixed and intentionally simple

Use fixed comparator families before any complex architecture:

1. climatology;
2. persistence diagnostic where causally permitted;
3. elastic-net logistic regression;
4. frozen XGBoost specification;
5. one existing frozen IRIS development candidate if its exact features are available on the development cohort.

No hyperparameter search after observing benchmark deltas.

The point is to test evaluation mechanics across model families, not build the highest-scoring network.

## 9. Bootstrap and uncertainty

Bootstrap unit: complete physical SEP episode or predeclared quiet block.

Critical implementation rule:

- generate one shared bootstrap index tensor;
- reuse the exact same draws across all models, evaluation views, delays, ablations, and pairwise contrasts;
- preserve episode membership within every draw;
- never redraw independently across conditions.

Primary interval: percentile 95% paired bootstrap interval with 10,000 replicates, plus point estimate.

If positive episodes are too sparse for stable inference, the result is `UNDERPOWERED`, not a rescued analysis.

## 10. Attrition ledger

Every candidate issue time receives exactly one terminal status:

- `ELIGIBLE_ONSET_POSITIVE`
- `ELIGIBLE_ONSET_NEGATIVE`
- `ALREADY_ACTIVE_PERSISTENCE`
- `UNRESOLVED_GAP`
- `IMMATURE_OUTCOME_WINDOW`
- `DUPLICATE_OR_AMBIGUOUS`
- `SOURCE_UNAVAILABLE`
- `EXCLUDED_BY_FROZEN_BOUNDARY`

Counts must reconcile exactly to the candidate cohort size.

## 11. Required persisted evidence

For every development prediction row:

- issue timestamp;
- frozen cohort role;
- eligibility code;
- physical episode identifier if label access is authorized;
- model name/version;
- probability;
- threshold;
- binary alert;
- sample weight under each evaluation view;
- source/provenance flags.

Persist exact prediction files, model files where permitted, environment manifest, source hashes, configuration hash, bootstrap-draw hash, and result-table hash.

## 12. Synthetic/property tests before real scoring

The evaluator must pass constructed cases including:

1. one 72-hour event represented by three positive windows and one 24-hour event represented by one positive window — each episode must total weight 1;
2. an issue already above threshold — must be persistence, never onset;
3. crossing exactly inside the 24-hour outcome horizon — onset positive;
4. crossing after the horizon — onset negative;
5. observation gap at issue time — unresolved/not onset eligible;
6. duplicate timestamps — reject;
7. shared bootstrap draws — all model contrasts use identical episode indices;
8. model-rank reversal synthetic fixture — verifies the rank-stability analysis detects a deliberate reversal;
9. null fixture — standard and normalized metrics are identical when every event has exactly one positive window.

## 13. Protected cohort rule

Development and validation mechanics use already-exposed data only.

No post-2025-09-10 outcome identities, labels, event counts, episode durations, model scores, or class balance may be inspected by the development side to choose the study design or rescue power.

A final untouched claim requires independent custodian selection, hash/attestation, overlap check against the inspected-evidence registry, and proof that the complete contract was frozen first.

## 14. Competition-level decision gate

A strong result requires all of the following:

- evaluator passes all synthetic/property tests;
- prediction-level evidence is complete;
- attrition ledger reconciles exactly;
- at least two nontrivial fixed model families are evaluated;
- episode-level uncertainty is reported;
- effect direction is not selected post hoc;
- any rank change or metric shift survives the paired shared-draw analysis;
- null and negative results remain visible;
- no untouched claim is made from exposed development data.

## 15. Judge-facing explanation

> If one radiation storm lasts three days, should it count as three successful predictions while a one-day storm counts once? Our benchmark asks whether that evaluation choice changes which forecasting system appears best—and separates predicting a new storm from recognizing one already underway.

That is the entire story. All technical machinery exists to answer it rigorously.