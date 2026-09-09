# Freshness crossover V1 post-run audit — 2026-09-09

## Status

`MODEL_RESULTS_FROZEN — UNCERTAINTY_IMPLEMENTATION_CORRECTION_REQUIRED`

The first successful end-to-end run was GitHub Actions run `34341335588` on commit `ed3aba1def18bb022382efa4afaaf251c76e6cff`.

The run produced clean-model, delay and policy point estimates. Those point estimates are now frozen and may not be changed or used to select a new model, cohort, threshold, delay grid, feature, or policy.

## Conformance issue discovered during result audit

The pre-outcome statistical clarification required bootstrap resampling of underlying positive events / quiet blocks **with all delay variants together**, paired across methods and delay conditions.

The executed compatibility wrapper correctly defined positive units as target crossing episodes and negative units as seven-day quiet blocks. It also calculated the `reduced_minus_joint` contrast using the same sampled units within each delay. However, it advanced the random-number generator separately inside each method/delay loop. Consequently, delay-specific intervals were block/event bootstraps, but the same bootstrap replicate was not preserved across every delay and method as required for fully paired cross-delay contrasts.

This is an implementation-conformance defect, not a scientific design change.

## Allowed correction

A correction may only:

1. deterministically regenerate the identical frozen model predictions;
2. generate the bootstrap base-unit sample indices once for each replicate;
3. reuse those exact indices across both delayed families, all frozen delay values, and all methods;
4. recompute uncertainty intervals from those paired replicates.

The correction may not change:

- source files or source semantics;
- clean cohort eligibility;
- labels;
- role boundaries;
- features;
- XGBoost seeds/hyperparameters;
- calibration;
- 2016 thresholds;
- delay grid;
- point-estimate metrics;
- policy selection;
- practical success gate.

The corrected run must preserve and compare the original point estimates byte-for-byte or numerically within floating-point tolerance. Any point-estimate disagreement is a failure and stops the correction.

## Preliminary result boundary

Before the uncertainty correction, the observed point estimates already show no sign-changing freshness crossover under the frozen definition: the fresh reduced-input model has higher TSS than the joint model at delay 0 for both delayed-family experiments. This observation must not be used to alter the study. The corrected uncertainty can quantify the existing frozen comparisons only.

The completed V1 study remains retrospective development evidence with a very small number of positive episodes in the calibration, threshold and score roles. It is not independent final evidence or an operational replay.
