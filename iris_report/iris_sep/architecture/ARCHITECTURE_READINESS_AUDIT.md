# IRIS-SEP architecture readiness audit

Date: 2026-09-04  
Decision: **retain the three-expert backbone; do not begin benchmark training yet**  
Locked-test outcomes accessed: **no**

## Honest verdict

The high-level architecture is scientifically defensible and appropriately
compact, but the current implementation is not yet sufficient evidence for an
IRIS or ISEF-winning project. Architecture cannot guarantee an award. The
project becomes competitive only if the completed, same-cohort experiment
passes the predeclared paired benchmark gate and the student can independently
explain every design choice, limitation, and result.

## Retain

- One primary question: a new operational SEP threshold crossing within 24
  hours.
- Three forecast-time experts: magnetic state, eruption evidence, and
  pre-event particle context.
- Compact temporal convolutions and gated late fusion.
- Explicit observation masks, freshness, and missing-modality training.
- Episode-disjoint chronology, a 24-hour purge, separate calibration and
  threshold roles, and one-time locked evaluation.
- The exact-identity rule that currently keeps AIA/HMI fusion disabled.
- Five seeds, paired episode/block resampling, negative-result reporting, and
  a same-cohort SEPNET-O comparison.

## Correct before training

1. Implement and hash the real data adapter, publication-latency rules,
   feature manifest, cohort manifest, partition manifest, and exclusion log.
2. Enforce episode disjointness, purge boundaries, exact issue identities,
   duplicate rejection, and train-only preprocessing in executable tests—not
   only configuration prose.
3. Make the primary occurrence head the default scientific experiment.
   Secondary heads must be individually enabled and ablated.
4. Add validity masks for missing auxiliary labels. Peak-flux loss must declare
   whether it is conditional on an event; onset loss must correctly represent
   censoring and at-risk intervals.
5. Predeclare class-imbalance handling and multi-task weights using training
   and permitted validation roles only.
6. Preserve feature-level observation masks rather than only an aggregate
   completeness channel when features have different missingness patterns.
7. Specify modality-specific timestamps, cadence conversion, and publication
   latency. A caller-supplied availability flag must not bypass consistency
   checks against the frozen adapter receipt.
8. Define uncertainty from the five-seed ensemble and distinguish epistemic
   spread, calibration uncertainty, and missing-input warnings.
9. Implement the baseline runner, calibration, evaluator, matched-detection
   comparison, and reproducible SEPNET-O adapter on identical identities.
10. Freeze exact tolerances for calibration degradation, lead-time degradation,
    matched detection, seed aggregation, and bootstrap confidence intervals.

## Statistical feasibility gate

Before choosing model capacity, count independent SEP episodes and quiet blocks
in each non-test role. The 650 reported positive rolling windows are not 650
independent events. Run a validation-side power and interval-width analysis.
If separate monitor, calibration, and threshold groups are too sparse, use a
predeclared grouped cross-fitting design within the non-test development era
rather than silently pooling roles or consulting the locked test.

## Judge-facing scope

The public model diagram has three expert blocks and one primary probability.
Secondary forecasts belong in supporting evidence unless they pass their own
validity and ablation gates. The novelty claim is not “a new neural network.”
It is an availability-constrained, physically motivated forecast evaluated
against a reproduced operational comparator under unusually strict leakage and
false-alarm controls.

The final headline is permitted only when the paired benchmark gate passes.
Otherwise the result is reported as negative or inconclusive.

## Competition-readiness gates

- **Architecture readiness:** all ten corrections above implemented and tested.
- **Scientific readiness:** identical-cohort baselines and five-seed validation
  receipts complete without locked-test access.
- **Claim readiness:** one-time locked evaluation passes every declared gate.
- **Presentation readiness:** event, quiet, and failure replays; reliability and
  false-alarm/detection plots; limitations; reproducibility package; and an
  accurate student ownership and assistance record.

Until all four gates pass, describe IRIS-SEP as a promising experimental system,
not a breakthrough or award-winning model.
