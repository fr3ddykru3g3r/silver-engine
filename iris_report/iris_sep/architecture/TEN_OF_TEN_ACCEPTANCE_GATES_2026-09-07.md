# IRIS-SEP — evidence-based 10/10 acceptance gates

Date: 2026-09-07

This file converts the informal goal of making every part of IRIS-SEP "10/10"
into falsifiable engineering and research gates. Passing a software test alone
never upgrades a scientific claim. A category is only marked complete when the
listed evidence exists and is independently auditable.

## 1. Problem relevance

**10/10 gate:** the system addresses a documented operator problem: producing a
24-hour NEW >=10 MeV, >=10 pfu SEP probability while upstream observations may
be late, missing, stale or incompatible. The use case must remain analyst
support; spacecraft autonomous control, economic savings and certification are
outside the demonstrated scope.

Status: **PASS for problem relevance; not a deployment claim.**

## 2. Scientific design

**10/10 gate:** target semantics, chronology, purge, role separation, seed
aggregation, calibration and threshold policies are frozen before decisive
results; negative arms are retained; locked evaluation is not used for model or
policy selection.

Status: **PASS for the current development experiments.**

## 3. Causality and source provenance

**10/10 gate:** every predictor family has a forecast-time observation/publication
record and native/reconstructed/unknown lineage. Any aggregate feature whose
historical construction uses future overlap fitting or noncausal interpolation
must be marked unresolved or rebuilt causally. Finite values may never be
interpreted as proof of native observation.

Status: **NOT YET PASS.** Existing released-table results are retrospective;
strict prospective-input causality remains unresolved for some upstream
preprocessing.

## 4. Clean-input forecasting skill

**10/10 gate:** the frozen candidate shows positive paired evidence against a
credible same-cohort comparator, lower FAR at matched detection, and no material
calibration degradation on an evaluation cohort independent of candidate and
policy selection.

Status: **NOT YET PASS.** Development point estimates are promising; fresh final
superiority is blocked by independence/provenance.

## 5. Missing-data reliability

**10/10 gate:** event-bearing outage experiments demonstrate a predeclared
handling route for each required availability state. NORMAL is forbidden unless
non-degradation is established. DEGRADED must have an explicitly tested fallback
and ABSTAIN must remain available outside support. Quiet-only outage evidence
cannot substitute for event-bearing evidence.

Status: **PARTIAL.** The preregistered event-terminal benchmark is complete and
audited; simple fill methods did not earn NORMAL status. An availability-
conditioned no-fabrication fallback is the next predeclared experiment.

## 6. External comparison

**10/10 gate:** at least one strong published/released architecture and, where
semantics permit, one operational forecast are compared on identical or clearly
qualified dates/targets. Paper-to-paper numbers with different cohorts do not
count as superiority evidence.

Status: **PARTIAL.** Released SEPNET-PRISM architecture comparison exists on the
IRIS cohort. NOAA/SWPC remains contextual until issue-time and target alignment
is fair.

## 7. Reproducibility

**10/10 gate:** clean checkout plus pinned public inputs can reproduce all
headline development receipts; source-only CI passes the published environment
and portability environment; data-bound tests are explicitly separated rather
than silently skipped; every result artifact is hash bound.

Status: **NEAR PASS.** Dual-pandas source CI and immutable receipts exist. Final
packaged-model replay remains required.

## 8. Operator usefulness

**10/10 gate:** one machine-readable output includes probability, permission
(VALID/DEGRADED/ABSTAIN), source/provenance state, reasons, model/policy IDs and
evidence hashes. No missing modality may silently flow through a normal-looking
forecast. Real alternate observed sources outrank reconstruction.

Status: **PARTIAL.** The fail-closed resolver exists; it still needs binding to
the promoted load-only forecasting package and empirical fallback policy.

## 9. Deployable research package

**10/10 gate:** export all 15 specialist models, exact ordered feature lists,
fusion parameters, prevalence, calibration intercept, both frozen thresholds,
dependency versions and hashes. A load-only command must reproduce reference
probabilities within declared tolerance without retraining.

Status: **NOT YET PASS.** Current benchmark runners train in memory.

## 10. Submission and communication

**10/10 gate:** paper tables/figures are generated from verified receipts; the
paper reports negative results, causality limitations and comparator
qualifications; synopsis and 90-second video explain the same frozen scientific
question without award/industry-superiority claims unsupported by evidence.

Status: **NOT YET PASS.** Drafts exist, but must be regenerated after the final
development evidence/package is frozen.

---

## Stop rule

No new neural architecture search is allowed on the current daily aggregate
interface. The next work order is:

1. lock the audited positive-event outage result and conservative operator rule;
2. test availability-conditioned specialist fallback without fabricating missing
   measurements;
3. export and verify the load-only model package;
4. complete provenance/causal-input remediation or clearly bound the retrospective
   claim;
5. finish fair external comparisons;
6. generate submission material directly from receipts.

A category may only be called 10/10 when its gate above is actually satisfied.
