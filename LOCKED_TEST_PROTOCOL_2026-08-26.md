# IRIS locked-test protocol — frozen 2026-08-26

This file is committed before the historical test partition is evaluated by the CDR feature comparators.

## Dataset and independence

- Historical evidence bundle: integrity-locked active-region-connected split.
- Forecast target: M1+ flare within 24 hours.
- Temporal input: 40 SHARP states separated by 36 minutes.
- Physical region groups remain disjoint across train, validation and test.
- Test selection for the feature-comparator experiment is the pre-existing evaluation rule: deterministic temporal cap of 4 endpoints per connected region, maximum 1 positive endpoint per region, seed 2028.
- No test observation, label or metric may be used to change architecture, features, reward, threshold, training duration or sample selection after this commit.

## Frozen models

All four checkpoints are those produced by GitHub Actions run 32957029797. No retraining occurs in the test evaluator.

| Model | Features | Frozen validation threshold | Validation TSS | Validation AUROC |
|---|---:|---:|---:|---:|
| supervised_2 | R_VALUE, AREA_ACR | 0.26 | 0.6268752122 | 0.8649347796 |
| cdr_2 | R_VALUE, AREA_ACR | 0.46 | 0.5423183716 | 0.8355456343 |
| supervised_10 | 10 SHARP features | 0.33 | 0.6009068936 | 0.8650845968 |
| cdr_10 | 10 SHARP features | 0.41 | 0.4629252312 | 0.7520724716 |

The validation evidence already indicates that the implemented CDR reward shaping underperforms its supervised counterpart on TSS/AUROC. The test analysis will report this result unchanged if it persists.

## Frozen primary analyses

For every model report TSS, HSS, recall, FPR, precision, AUROC, AUPRC, Brier score, Brier skill score and 10-bin ECE at its validation-frozen threshold. Uncertainty is a connected-region cluster bootstrap with 5,000 replicates.

The two prespecified paired comparisons are:

1. CDR-2 minus supervised-2.
2. CDR-10 minus supervised-10.

They use the same test sample identities and a paired connected-region bootstrap. Threshold-dependent metrics use each model's own validation-frozen threshold. Positive delta means larger CDR metric; for Brier/ECE a positive delta is worse.

## One-shot rule

After the workflow evaluates the test set, the resulting test metrics are final for these frozen checkpoints. Any subsequent model modification must be explicitly labelled a new experiment and must not be described as the original locked-test result.
