# Public NEW-crossing missingness result — 2026-09-06

## Decision

**Completed diagnostic; recovery-method signal found, but the fixed reference forecast fails the adequacy gate. Do not use this run as final robustness or superiority evidence.**

This result is preserved because it establishes an executable, hash-bound, train-only NEW-crossing outage benchmark and because its negative reference-model finding materially changes the next experiment.

## Immutable execution

- GitHub Actions run: `33987054599`
- job: `101362475080`
- source head: `413249fad92c07c1ecbb5a29b4436320ec845b52`
- artifact ID: `9975493258`
- artifact ZIP SHA256: `d275ee6a394ccc86b7dde2d9883d0f900423d89421791ab11cfdc00b727767f6`
- result: success
- locked test accessed: **false**
- previously inspected monitor included: **false**

Thirty missingness/model-contract tests passed before the data experiment.

## Package

Target: `new_sep_10mev_10pfu_within_24h`

The package was reconstructed independently from the pinned public SEP-PRISM predictor table and CLEAR operational event catalogue. Issue times already inside an operational >=10 pfu event are excluded; a positive issue is one for which a new operational event starts within the next 24 h.

- rows: **13,308**
- positives: **207**
- causal predictor features: **259**
- observed finite cells: **2,201,367**
- pre-existing unavailable cells: **1,245,405**
- package SHA256: `bd7e4ed4847307024c92d30d1b7ebf1d6195449152c04371f6402cfd107d9b4d`
- metadata SHA256: `76b36cb1d32d84d0b05be3582c4d7590d950bb2c0d46da291268eb0991152eba`
- source-manifest SHA256: `e11fc66133a4a44fb7260274d80b7dac06182fe37f00342448705b0007cd4c40`
- frozen consumer-contract validation: **passed**

Pre-existing unavailable cells were never selected for artificial hiding. This result does **not** claim that every pre-existing unavailable cell has a physical structural cause.

## Critical reference-model adequacy result

The fixed balanced L2-logistic reference used by the generic missingness benchmark is **not adequate as an operational robustness surrogate on this score block**.

On 3,219 score rows containing 21 positives, at its clean frozen threshold:

- TP = 0
- FN = 21
- FP = 403
- TN = 2,795
- POD = **0**
- TSS = **-0.1260**
- Brier = **0.008699**
- ECE = **0.008663**

Therefore the threshold/TSS portion of this diagnostic cannot establish that the promoted IRIS forecast remains operationally useful under missingness. The next transfer must use the promoted cross-fitted specialist evidence stack itself, frozen before outage injection.

## Artificial-outage results

The same three arms were evaluated at predeclared 5%, 20%, and 40% hiding of genuinely observed score-role cells:

1. `MASK_AWARE_NO_FILL`
2. `TRAIN_FIT_MEDIAN`
3. `CAUSAL_FORWARD_FILL`

No model was retrained on the hidden-data score cases.

### 5% hidden

Held-out cells: **32,971**.

| Arm | Delta TSS | Delta Brier | Delta ECE |
|---|---:|---:|---:|
| Mask-aware no-fill | -0.075672 | +0.018545 | +0.019601 |
| Train-fit median | -0.075672 | +0.018545 | +0.019601 |
| Causal forward-fill | **-0.000313** | **+0.000674** | **+0.000741** |

### 20% hidden

Held-out cells: **131,884**.

| Arm | Delta TSS | Delta Brier | Delta ECE |
|---|---:|---:|---:|
| Mask-aware no-fill | -0.085098 | +0.108417 | +0.112302 |
| Train-fit median | -0.085098 | +0.108417 | +0.112302 |
| Causal forward-fill | **-0.001563** | **+0.000006** | **+0.000007** |

### 40% hidden

Held-out cells: **263,768**.

| Arm | Delta TSS | Delta Brier | Delta ECE |
|---|---:|---:|---:|
| Mask-aware no-fill | -0.039489 | +0.196342 | +0.200849 |
| Train-fit median | -0.039489 | +0.196342 | +0.200849 |
| Causal forward-fill | **-0.002502** | **-0.001473** | **-0.001346** |

The point results show that **causal forward-fill preserved the frozen reference probabilities much better than no-fill or train-fit median across all three artificial-outage levels**. This is a development-only recovery-method signal, not a final forecast-robustness claim.

Matched-detection FAR remained extremely poor for the weak reference/candidates (approximately 0.993 at the POD>=0.8 diagnostic operating point), reinforcing the adequacy failure rather than rescuing the result.

## Reconstruction-metric warning

Do **not** interpret the pooled raw-value MAE/RMSE from this run as a physical reconstruction score. The 259 predictors mix incompatible units and scales and include extremely large magnitudes; pooled raw error was therefore dominated by feature scale. A future reconstruction analysis, if needed, must preregister per-feature normalization or dimensionless errors before results are inspected.

## What this result establishes

1. A public, hash-bound, train-only NEW-crossing missingness package can now be built and validated end to end.
2. Synthetic outages can be applied only to real observed cells while pre-existing unavailable cells remain protected.
3. Causal forward-fill is the strongest of the three tested recovery candidates for probability preservation in this weak-reference diagnostic.
4. The generic fixed logistic is not adequate for the final robustness claim.

## What it does not establish

- no locked-test result;
- no final NEW-crossing superiority;
- no operational certification;
- no physics-reconstruction advantage;
- no economic savings;
- no award or breakthrough claim.

## Next experiment

Transfer the exact outage protocol to the already-promoted `IRIS_CROSSFIT_EVIDENCE_STACK_V1` with the model, calibration and thresholds frozen on clean development roles before hiding score-role cells. Because this result has already revealed forward-fill as the strongest arm, the transfer must retain **all three** recovery arms and **all three** outage fractions and explicitly disclose the prior observation. The transfer was preregistered separately before its outcomes were available.
