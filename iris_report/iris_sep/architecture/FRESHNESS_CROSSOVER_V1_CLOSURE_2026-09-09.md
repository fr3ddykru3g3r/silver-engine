# IRIS-SEP Freshness Crossover V1 — scientific closure

**Closure date:** 2026-09-09  
**Status:** `CLOSED_NEGATIVE_RESULT_NOT_A_WINNING_CLAIM_ROUTE`

## Decision

The freshness-crossover branch is closed as a discovery/promotion route. It must remain in the project record as a falsifiable negative experiment and must not be rescued by post-result threshold, delay-grid, architecture, cohort, feature, or switching-rule changes.

The tested claim was:

> Under identifiable conditions, using stale additional measurements is worse than using fewer fresh measurements, and a simple rule can exploit that relationship.

The completed frozen experiment did **not** support that claim.

## Immutable execution identity

- Executed commit: `ed3aba1def18bb022382efa4afaaf251c76e6cff`
- GitHub Actions run: `34341335588`
- Evidence artifact: `10100069342`
- Artifact SHA-256: `ba907ffae86301c22b1afb3a41a0bf98c825b562b90b5319ce347a4af1db7868`
- Result record: `architecture/FRESHNESS_CROSSOVER_V1_RESULT_2026-09-09.md`
- Follow-up bootstrap-conformance audit commit: `b3a3de249f9bea5be1ccd7ac24017a161ce44b7c`

## Verified outcome

The archived aggregate arithmetic was independently audited after the run. All 144 delay-table TSS/FAR rows and all six pooled policy rows were consistent with the saved counts. No frozen policy passed the practical gate.

The 2017 clean score cohort contained only three positive opportunities:

| Model | TP | FP | FN | TN | TSS | FAR |
|---|---:|---:|---:|---:|---:|---:|
| Joint | 3 | 152 | 0 | 56 | 0.269231 | 98.06% |
| XRS-only | 2 | 73 | 1 | 135 | 0.315705 | 97.33% |
| Proton-only | 3 | 102 | 0 | 106 | 0.509615 | 97.14% |

Proton-only therefore detected 3/3 sampled positive opportunities but generated 102 false alerts. Under a simple independent-event binomial interpretation, 3/3 corresponds to a very wide two-sided 95% sensitivity interval of roughly 29.2%–100%; temporal/physical dependence can make that intuition even less informative.

## Why the crossover failed

A crossover requires the joint model to begin superior, or at least demonstrably preferable, when both feeds are fresh and then lose that advantage as one feed becomes stale.

That ordering did not occur:

- with XRS delay, fresh proton-only already exceeded joint at delay 0 by about `+0.2404` TSS;
- with proton delay, XRS-only also had a higher point-estimate TSS than joint at delay 0;
- no evaluated delay produced the required model-order sign reversal;
- fixed TTL selection collapsed to `0` minutes and reproduced the always-reduced reference;
- the XRS state-dependent policy increased pooled replay false alerts from `816` to `988` (`+21.08%`);
- all practical policy gates failed.

The experiment therefore cannot isolate a mechanism in which *staleness itself* causes an initially beneficial extra feed to become harmful.

## Statistical and operational limitations that remain part of the result

1. **Sparse positive support.** Calibration had two positives, threshold selection had one, and the 2017 score role had three. Maximum-TSS threshold selection on one positive is a screening device, not a robust skill demonstration.
2. **Prediction-level evidence was not persisted.** The immutable artifact lacks per-issue probabilities, fitted models, and shared bootstrap draw matrices. Aggregate TSS/FAR arithmetic is reproducible, but average precision, paired discordance, paired p-values, and the reported bootstrap intervals cannot be independently regenerated from the artifact alone.
3. **Bootstrap contract only partially implemented.** Methods were paired within each delay contrast, but underlying event/quiet-block samples were redrawn across delays. Existing intervals are pointwise conditional contrasts and cannot establish simultaneous crossover evidence across the delay curve.
4. **Conditional retrospective eligibility.** During simulated proton delay, cohort eligibility still consults contemporaneous proton information for active/inactive status. This is an oracle relative to the artificial outage and means the study estimates performance in a preselected retrospective population, not an implementable availability-aware replay.
5. **Delay changes both age and retained information.** The fixed trailing window loses more of the delayed feed as delay increases, so information age and information quantity are not cleanly separated.
6. **Repeated delay variants are not new events.** Replaying the same three positives over many delays does not multiply independent scientific support.

These limitations must not be repaired silently inside V1. Any correction that changes data, eligibility, estimand, or evidence objects belongs to a separately versioned study.

## Novelty boundary

Daily proton/X-ray forecasting, XGBoost, proton-dominant predictors, multimodal SEP forecasting, and onset prediction already have close prior art. The distinctive element of V1 is the frozen controlled-delay protocol and its negative result, not a new forecasting architecture or demonstrated warning intervention.

## Frozen claim

The strongest claim permitted from this branch is:

> In this retrospective two-satellite experiment, reduced-input models were already competitive with or better than the joint model when both feeds were fresh. The preregistered freshness-induced crossover was not demonstrated, and no frozen switching policy improved the required warning trade-off.

Do **not** generalize this to “X-rays are harmful,” “fewer sensors are better,” “freshness never matters,” or operational superiority.

## Closure rule

- No further model, threshold, delay, feature, or policy tuning may use the exposed V1 score outcomes as a route to a positive freshness claim.
- Any forensic reconstruction of missing per-issue predictions must be clearly labelled as a reconstruction and may not be called the original immutable artifact.
- The negative result remains visible in the notebook, paper evidence dossier, and judging preparation.
- A separate scientific question requires a separate exposure ledger, preregistration, admissible independent cohort, precision/power plan, and evidence package.
