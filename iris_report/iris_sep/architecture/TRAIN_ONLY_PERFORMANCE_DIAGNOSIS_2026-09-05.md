# Train-only performance diagnosis — 2026-09-05

This is the first four-fold chronological comparison run entirely inside the
already pinned outer `role=train` data. It is development evidence on the
publisher's legacy operational-window label. It is not a result for the final
NEW-crossing target, SEPVAL, or a locked test.

## What ran

Each fold uses expanding chronological fit data followed by separate
early-stopping, calibration, threshold-selection, and score blocks. Complete
development units stay together and adjacent roles have a strict 24-hour purge.
Preprocessing is fit on each fold's fit block. The tested arms were climatology,
elastic-net logistic regression, fixed XGBoost, the existing compact IRIS model,
and a predeclared compact variant applying `sign(x) * log1p(abs(x))` before
train-fitted normalization. Five fixed seeds were used except climatology.

The four score blocks contain 2,441 distinct windows and 427 development units.
The run read only outer `role=train`; it did not score the already used outer
monitor, calibration, or threshold roles and did not access locked data.

## Result

| Model | Pooled TSS | FAR | Brier | Disposition |
|---|---:|---:|---:|---|
| XGBoost | 0.287 | 0.798 | 0.123 | strongest completed fixed reference |
| Elastic net | 0.276 | 0.798 | 0.137 | no paired advantage over XGBoost |
| Compact IRIS, signed-log | 0.258 | 0.796 | 0.117 | numerically stable; no paired advantage |
| Climatology | 0.000 | 0.857 | 0.134 | skill floor |
| Compact IRIS, original | 0.292 on three folds only | 0.811 | 0.090 | invalid aggregate; failed latest fold |

The original compact model emitted nonfinite logits on the latest fold: only
1,062 of 2,120 calibration/threshold/score logits were finite for seed 7. Its
three-fold aggregate must not be compared with four-fold models. The signed-log
variant completed all folds, demonstrating a numerical mitigation, but its
TSS difference from XGBoost was -0.0289 with paired unit-bootstrap 95% interval
[-0.0928, 0.0285]. Elastic net's difference was -0.0111 with interval
[-0.0451, 0.0225]. Neither selection gate passed.

Performance is unstable by era. XGBoost fold TSS values were 0.013, 0.432,
0.295, and 0.020. Every learned method was poor or harmful on the earliest
score block; all were useful on the middle two; all were weak on the latest.
This variability is more important than the pooled ranking.

## Failure diagnosis

The train-only structural audit found 7,812 windows, 1,382 units, and 1,318
positive windows, but event support varies sharply by quarter. One stopping
block has only three positive rows and one score block has 16. The ten most
missing SHARP aggregates are completely absent from 1986–1999 rows and still
71.89% absent from 1999–2013 rows. A coarse magnetic-availability flag looks
complete because label-like columns are populated while the measurements are
not. The current table has no particle-context branch.

The evidence therefore supports three failure hypotheses: the models learn
instrument/era availability regimes; train-fitted scaling extrapolates badly
under later feature distributions; and thin chronological event support makes
early stopping and threshold selection unstable. Signed-log preprocessing
fixes numeric overflow but not the missing physical information or cohort shift.

## Decision

Do not promote or further tune any tested arm against the used outer monitor.
The next high-value model experiment requires a verified training-only
NEW-crossing cohort with issue-time proton context and XRS history, explicit
publication latency, authoritative episode semantics, and source/product
indicators. Before training it, freeze fold support minima and an abstention or
fallback policy for unsupported source eras. Compare fixed XGBoost, elastic net,
reproduced SEPNET, and the compact model on identical folds. More layers are not
justified by this result.

Authoritative receipts:

- `artifacts/train_inner_diagnostic_v4/preregistration.json`
- `artifacts/train_inner_diagnostic_v4/receipt.json`
- `artifacts/train_inner_diagnostic_v4/analysis_receipt.json`
- `workstreams/luna_training_diagnosis_20260905/training_diagnosis.json`

Failed partial runs `train_inner_diagnostic_v2` and `v3` are retained as audit
history. They are not approved results.
