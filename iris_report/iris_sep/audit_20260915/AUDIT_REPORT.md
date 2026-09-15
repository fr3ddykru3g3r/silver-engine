# Scientific audit and execution report

## 1. Repository truth

The strongest supported project is an audit of **what SEP forecast scores measure**, using fixed historical forecasts. It is not an independently validated operational onset predictor. A physical radiation episode can contribute several positive daily windows. Daily windows in the audited source are nonoverlapping; “overlapping rolling windows caused the result” is not an established explanation. Repeated representation of a multi-day episode is sufficient.

The defensible question is: **How much do the scores of these fixed forecasts change when positive windows are weighted equally, physical episodes are weighted equally, and already-active episode windows are excluded?** This asks about distinct estimands. It does not assume conventional occurrence verification is invalid.

The default branch is stale relative to the research branches. PR #5 contains the episode benchmark work, but not all subsequent forecasting/controller/source work. The award branch contains additional mathematics and source inventories, while the episode branch contains a later correction that prevents treating reprocessed SGPS data as operationally equivalent. See the reconstruction and provenance tables.

A serious boundary conflict prevents unrestricted re-execution of the frozen archive. The newest prospective contract protects outcomes from 2025-09-10 inclusive, while older monitoring/replay definitions reach that boundary. A timestamp-first, strict-horizon audit was implemented. No excluded outcomes or event identities were interpreted, and no full-cohort numerical reverification is claimed.

## 2. Strongest confirmed discoveries

1. **Evaluation changes have a substantial effect with forecasts held fixed.** In the strictly admitted historical subset, joint-XGBoost TSS is 0.726442 → 0.621115 → 0.437222 for mapped occurrence, episode-normalized occurrence and onset. The differences are not the result of retraining or choosing a new threshold.
2. **The multiplicity mechanism is exact.** The first difference is Cov(n,r)/mean(n) = 0.105327; the episode-average persistence contribution is 0.183894. Their sum is 0.289221, the mapped-to-onset difference, because the negative measure is held constant and each admitted episode has one onset window.
3. **The effect depends on the model.** The past-proton proxy loses about 0.508331 TSS from mapped occurrence to onset. The no-proton comparator has a small pooled multiplicity term, −0.005835. This is a useful contrast, not evidence of exact independence or a universal negative control.
4. **The historical mechanism survives specified temporal-clustering checks.** Paired episode/bootstrap and calendar-bin resampling at 27 days, 90 days, quarter and year all give negative 95% intervals for the two joint-model contrasts. These are fixed-forecast, post-hoc robustness checks; nonstationarity and selection uncertainty remain.
5. **Apparent forecasting advances do not establish a forecasting breakthrough.** The direct Phase-II score has one onset. V2 controller FPR is 23.9%, but FAR is 98.0%, with one true onset and post-hoc development. The separate PR #6 Phase-II experiment misses its improvement criterion.
6. **A source-inventory false negative was corrected.** A partial endpoint month erased a preceding complete run. The corrected diagnostic finds 17 complete science-product months, November 2023–March 2025. This establishes filename coverage only. Sensor reduction, release-time causality and operational equivalence are still unresolved.

## 3. What is genuinely novel

The covariance identity, size-biased sampling and inverse-cluster weighting are known mathematics. Event verification is known in space weather. The potentially original contribution is their specific, reproducible combination with fixed SEP predictions, an onset/persistence decomposition, paired uncertainty and contrasting models on this dataset. Confidence in an application contribution is moderate; priority over every SEP paper is not established. No “first ever” or new-theorem claim is justified.

## 4. What is not novel / must not be claimed

Do not claim a new covariance theorem, universally inflated forecast scores, invalid published SEPNET final scores, operational causality, SOTA, validated controller TSS 0.76, independent test performance, statistically demonstrated onset-model rank reversal, or an effective sample size of 61 inferred from weights alone. Do not call average per-episode window sensitivity “fraction of events detected at least once.”

## 5. Fatal or serious weaknesses

| Priority | Issue | Work performed / remaining requirement |
|---|---|---|
| P0 | Conflicting prospective boundary | Strict timestamp-first horizon guard and tests implemented; custodian must reconcile governance without disclosing outcomes. |
| P0 | Historical development exposure | Explicit evidence classes and unchanged frozen endpoint; cannot be repaired by relabeling the test set. |
| P0 | Student ownership / AI assistance | This package is assistance, not student-authored initial submission material; authentic contribution and records must be demonstrated. |
| P0 | Source availability and sensor equivalence | Failure states retained; no live historical model promoted. |
| P1 | Misidentified novelty | Primary-source comparison and conservative claim hierarchy supplied. |
| P1 | Cluster dependence / overstated uncertainty | Episode and time-bin paired checks executed; model-selection and solar-cycle generalization uncertainty remain. |
| P1 | One-event rescue narrative | Controller and Phase-II metrics independently reconstructed and kept post-hoc/underpowered. |
| P1 | IRIS/SRC eligibility ambiguity | Live requirements checked; archival-data interpretation and current portal dates need official confirmation. |
| P2 | Event-definition sensitivity | Not fully resolved: independent preboundary catalog/flux validation and preregistered alternative definitions remain. Do not select a definition by effect size. |
| P2 | Physical causal explanation | Algebra identifies weighting contributions; it does not prove proton features causally cause score inflation. |
| P3 | Poster/video finish | Reference layout and timing supplied; student must author, rehearse and finalize. |

## 6. Current IRIS competitiveness

A credible methods project with a clear empirical result, but not submission-ready merely because this audit is complete. A deliberately skeptical desk assessment is 8/10 research question, 10/15 design, 13/20 execution, 12/20 creativity/impact: **43/65 for the technical portion**, an illustrative judgment rather than a measured fair score. Poster 10 and interview 25 are unscored because no student-authored poster or live defense was evaluated. Strong interview ownership can matter more than adding another model.

Deduction risks are novelty overclaim, confusing event detection with event-normalized window performance, unclear dataset exposure, weak source causality, and inability to explain AI-assisted work. No estimate of award probability is defensible.

## 7. Winning path

The smallest valuable path is: resolve the protected-boundary/SRC/student-authorship gates; make one fixed-forecast estimand comparison completely traceable; defend the exact decomposition and its limits; demonstrate temporal robustness and preserve negative controls; have the student reproduce and explain it unaided. An independent operational prospective study is a future falsification opportunity, not something to manufacture before a deadline.

For operators, the concrete deliverable is a verification annex: decision-day scores for continuing exposure, episode-average scores for equal physical-event representation, and onset-only scores for new-alert use. Also report false-alert burden and lead times where causally available. No operator adoption or economic benefit has been measured.

## 8. Execution

Implemented protected-horizon archive analysis, independent metric/decomposition checks, general weighting/bounds/AUC/Brier identities, paired uncertainty, time-cluster sensitivity, threshold/fold descriptive checks, controller probability/threshold reconstruction, and a corrected complete-source-run diagnostic. Added scientific invariant tests, machine-readable outputs, provenance/hashes, technical figures, a claim ledger, five-judge defense exercises and a submission reference workbook.

Not completed or falsely represented: independent full protected-boundary replay; a fresh operational forecast model; independent physical-event truth under alternative catalogs; exhaustive full-text review of every paper; expert operator feedback; official SRC decisions; student authorship; signatures; a recorded defense. These require new external evidence or the student's own work, not cosmetic edits.
