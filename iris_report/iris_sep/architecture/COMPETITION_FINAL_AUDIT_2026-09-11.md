# IRIS / ISEF competition final audit — 2026-09-11

**Project:** Are We Forecasting a New Solar Radiation Storm? An Episode-Normalized Causal Benchmark for SEP Prediction  
**Evidence class:** strong methodological confirmation on development-exposed historical public data; protected prospective outcomes remain sealed.  
**Competition-outcome boundary:** no audit can guarantee qualification, an award or first place.

## Executive judgment

This project is now substantially stronger as a **measurement-science benchmark** than it was as a forecasting-model project. Its best competition story is not “our XGBoost predicts SEP storms well.” It is: **“before comparing models, make sure the score measures the physical claim you think it measures.”**

The 85-onset matched replay resolves the earlier five-event power problem enough to support the evaluation-effect story, while the project still correctly refuses operational/model-superiority claims. The remaining blockers to a top-tier scientific defense are evidence independence, issue-time feature provenance, temporal-boundary purging and student-authored compliance material—not another model search.

## ISEF-style scorecard

Use this as an internal target, not a promised judge score.

| Criterion | Current audit | Main reason points are not maxed | Highest-value action |
|---|---:|---|---|
| Research question | 9.5 / 10 | Scope must stay narrow and causal | Keep one sentence throughout poster/interview |
| Design & methodology | 12.5 / 15 | Historical feature availability + no universal purge | Present frozen prospective contract as the planned falsification path |
| Execution & data analysis | 17.5 / 20 | Exposed cohort; artifact cannot reconstruct fitting end-to-end | Preserve minimal independent recheck + disclose missing model/threshold-block objects |
| Creativity & potential impact | 17.5 / 20 | Components individually established | Defend novelty as the combined estimand decomposition, not onset itself |
| Presentation / interview readiness | 28 / 35 | Stale submission prose and denominator traps were present | Student rewrite + drill the hard questions + headline two figures |
| **Total** | **85 / 100** |  | **A 90+ defense is plausible after compliance/presentation cleanup, not guaranteed.** |

## SDG alignment

### Primary: SDG 9 — Industry, Innovation and Infrastructure

The project improves the reliability of evaluation methods for space-weather forecasting, which supports resilient technological infrastructure and scientific innovation. This is the cleanest SDG link.

### Secondary: SDG 3 — Good Health and Well-Being

SEP radiation can create exposure risk for astronauts and high-altitude aviation. The link is indirect: this benchmark does not reduce radiation by itself, but it can improve how pre-onset warning claims are evaluated.

Do not force an SDG 13 climate-action framing. Space weather is not terrestrial climate change.

## Novelty gap

### What is not new

- SEP onset and persistence as physical concepts;
- TSS/HSS and event-based forecast validation;
- proton/X-ray features, XGBoost, elastic net or deep SEP models;
- evaluating a persistence baseline.

### What this project contributes

The tested novelty is the **combined measurement framework** that, for unchanged forecast probabilities/alerts:

1. maps positive daily windows to physical SEP episodes;
2. reports standard mapped occurrence;
3. normalizes positive mass so every uniquely mapped physical episode contributes total weight one;
4. separates already-active persistence from genuine new-onset opportunities; and
5. compares these estimands with shared physical-event/quiet-block bootstrap draws.

The novelty claim should be “we did not find this full combined evaluation in the recent SEP work reviewed,” not “nobody has ever thought of onset or event-based evaluation.”

## Recent literature that most threatens novelty

The strongest comparator families are:

1. **Yu et al., SEPNET (2025/2026)** — sophisticated multi-task deep learning and SEPVAL validation; threatens any model-architecture or SOTA claim.
2. **Yu et al., SEP-PRISM Data (2026)** — directly supplies the 14,464-window 24-hour benchmark representation; it is the closest dataset-context paper.
3. **Ali et al. (2024), cycles 22–24 proton/SXR ML** — daily proton/SXR features, persistence comparison and cross-cycle robustness; threatens claims that proton-history evaluation is unexplored.
4. **Rotti et al. (2024), short-term multivariate time-series classifiers** — high-skill SEP classification over short horizons; threatens generic “ML can forecast SEP” novelty.
5. **Papaioannou et al. (2025), ASPECS validation** — rigorous operational validation and standard skill metrics; threatens generic “validation” novelty.

### Defense paragraph skeleton for the student to rewrite

Recent work improves SEP model architecture, input fusion, cross-cycle generalization, curated 24-hour datasets and operational validation. This project asks an orthogonal measurement question: when a fixed-window benchmark gives several positive rows to one physical storm and includes issue times after the operational threshold has already been crossed, what ability does the reported score represent? The contribution is the paired decomposition of the **same frozen predictions** into occurrence, episode-normalized occurrence and causal new-onset estimands, with uncertainty resampled over physical event units rather than treating daily rows as independent.

## Statistical audit

A minimal independent script was run against fixed-model artifact `10137507101` without importing the benchmark runner.

- archive SHA-256 independently matched `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`;
- matched persisted rows: 45,066 across six models = 7,511 rows/model;
- positive resampling units: 85 onset episodes;
- negative resampling units: 1,080 quiet blocks;
- joint-XGBoost mapped/normalized/onset point TSS independently recomputed as 0.726455 / 0.621128 / 0.437234;
- three primary 10,000-draw paired contrasts reproduced to machine precision.

Reproduced bootstrap medians and 95% percentile intervals:

| Contrast | Median | 95% interval |
|---|---:|---:|
| Joint XGB: normalized - mapped | -0.103744 | [-0.146175, -0.063165] |
| Joint XGB: onset - normalized | -0.183333 | [-0.246639, -0.125070] |
| Past-proton proxy: onset - mapped | -0.506167 | [-0.566675, -0.443973] |

All 10,000 stored draws were negative for each of these three contrasts. This is a useful reproducibility diagnostic, **not a classical p-value** and not a family-wise significance guarantee. ANOVA was not added because the primary inferential problem is paired change in a nonlinear skill metric with correlated physical-event units; the frozen physical-unit bootstrap is better aligned with the estimand.

## Methodology weaknesses that must remain visible

1. **Prior exposure:** public source hashes were already development-inspected.
2. **Feature timing:** excluding explicit future columns does not prove every historical predictor was available before issue time.
3. **Boundary leakage control:** no programmed purge exists at every fit/threshold/score boundary.
4. **Retrospective onset state:** historical catalogue boundaries define eligibility; this is not yet a validated issue-time sensor rule.
5. **Artifact completeness:** predictions and thresholds are persisted, but fitted model objects and threshold-block probabilities are not sufficient for a full training/threshold reconstruction from the artifact alone.
6. **Operational performance:** joint-XGBoost onset sensitivity is 48.24% and false-alarm ratio 88.95%; this is not deployment evidence.

These are scoring risks only if hidden. When stated clearly, they strengthen the scientific-thought and integrity story.

## Ethics and ISEF compliance determination

Current methods analyze public astronomy/space-weather datasets and code. No human participant, vertebrate animal, PHBA or hazardous-material procedure is described. Therefore specialized human/animal/biological/hazardous approvals are not currently indicated. If the methodology changes, re-run the Rules Wizard before beginning that new work.

For the current 2027 ISEF form set, plan on the universal project paperwork: Forms 1, 1A + Research Plan/Project Summary, 1B and Student Support Disclosure Form 2A. There is no current Form 8 in the 2027 form list. Form 7 applies only if this is a continuation from a prior competition year; RRI/QS forms apply only if actual institution/supervisor conditions trigger them.

AI/programming assistance must be disclosed accurately. Final ISEF-bound research-plan, abstract, poster and citations must be student-owned and compliant with the current AI-authorship rules. Repository-generated submission text is therefore intentionally relabeled as a drafting/rehearsal scaffold.

## Category recommendation

**Physics and Astronomy (PHYS)** is the strongest category. If a subcategory is requested, lead with the astronomy/space-physics domain. A computational/theoretical-physics framing is a secondary option. Do not choose Systems Software merely because Python/XGBoost are used; judges should evaluate the scientific measurement question.

## Figure audit

### Current TSS line chart

**Keep.** It is the strongest results figure because it makes the same-prediction comparison visible in one glance. Improve the caption/legend, not the data:

- state “matched cohort: 85 onset episodes + 1,080 quiet blocks”;
- state that thresholds/alerts are unchanged;
- label `FAR` elsewhere as false-alarm **ratio** if used;
- if space allows, add the three primary confidence intervals as compact annotations or a small adjacent table rather than cluttering the lines.

### Physical-storm flow diagram

**Keep and use first.** It communicates the mechanism faster than a machine-learning diagram. Suggested wording tweak: change “Fixed joint XGBoost TSS” to “Same fixed XGBoost alerts” so the causal comparison is visually explicit. Keep “previously exposed public data” in the footer.

### Figures to demote

The old 2014–2017 five-onset fixed-model graph should be supplemental/historical, not the headline. The higher-powered 85-event replay supersedes it for judge-facing evidence.

## Code / engineering audit

Strengths:

- source hashes and immutable evidence receipts;
- explicit frozen folds and thresholds;
- shared bootstrap draw tensors;
- separate verifier and source-only compatibility testing;
- protected-evidence registry and fail-closed prospective contract;
- negative experiments preserved rather than deleted.

Fix completed in this audit:

- added `tools/recheck_sep_prism_primary_contrasts_v1.py`, which independently recomputes the primary contrasts from the frozen artifact and can assert the published receipt;
- relabeled stale submission prose as student drafting/rehearsal material;
- updated judge defense to the 85-onset replay;
- corrected the project entry point to the episode-benchmark centerpiece.

## Most important next move

Do **not** run another historical model search. The single highest-value scientific step is the already-preregistered custodian-controlled prospective confirmation after issue-time feature availability is proven. For competition readiness before that result exists, the highest-value student step is a completely self-owned explanation of the benchmark, denominators, bootstrap and limitations.

## Submission red lines

- no guarantee of ISEF/IRIS placement;
- no “all prior SEP work is biased” language;
- no “SEPNET score is wrong” language;
- no “independent prospective validation” label for this replay;
- no operational-readiness claim;
- no p-value invented from the 10,000 bootstrap draws;
- no copying AI-generated submission prose into final ISEF-bound materials.
