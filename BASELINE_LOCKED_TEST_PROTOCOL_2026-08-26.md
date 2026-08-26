# IRIS real-image CNN locked-test protocol — frozen 2026-08-26

This protocol is committed before the selected real-only CNN is evaluated on the historical test partition.

## Frozen model

- Source checkpoint: `w48_focal/model.pt` from successful benchmark run 32937682857.
- Architecture: FlareCNN, width 48, dropout 0.20.
- Loss used in training: focal BCE, gamma 1.5.
- Learning rate: 3e-4.
- Seed: 2026.
- Validation-selected threshold: **0.40**.
- Frozen validation TSS: **0.5454028788**.
- Frozen validation AUROC: **0.8132559157**.

No retraining or parameter/threshold change occurs during test evaluation.

## Test sampling

The test partition uses the same deterministic connected-region sampling rule used for validation: 10 temporally spread endpoints per connected physical-region group where available, with at most 4 positive endpoints selected first, and seed 2028. This prevents long-lived regions with many hourly observations from dominating row-weighted metrics and keeps the evaluation protocol matched to validation.

## Frozen metrics

Report TSS, HSS, recall, FPR, precision, AUROC, AUPRC, Brier score, Brier skill score and 10-bin ECE at threshold 0.40. Uncertainty is estimated with a 5,000-replicate connected-region cluster bootstrap.

## One-shot rule

The resulting test metrics are final for this frozen checkpoint. Any later architecture, seed, sampling or threshold change is a new experiment and must not replace the original locked-test result.
