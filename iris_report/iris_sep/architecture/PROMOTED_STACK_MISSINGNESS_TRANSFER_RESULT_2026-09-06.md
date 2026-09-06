# Promoted-stack missingness transfer result — 2026-09-06

## Decision

**The promoted cross-fitted specialist stack tolerates modest random observed-cell loss in probability space, but severe loss is not safe to treat as a normal forecast. The current evidence supports an explicit DEGRADED/ABSTAIN boundary rather than a universal fill-and-forecast claim.**

This is development-only evidence on an already-inspected score block. It does not establish locked-test robustness or operational certification.

## Immutable execution

- GitHub Actions run: `33987312162`
- job: `101363172765`
- preregistered execution head: `4716fe03fa723005f095251575c28a04742b9435`
- preregistration SHA256: `80aea3a9d5a8e1db8a19b929869045b9a52dccde7f0ddac0d9c25074e23c7cf9`
- artifact ID: `9975602713`
- artifact ZIP SHA256: `a2b6f3d91c6bdfebe0dcf02c86b37f57700143abef97246a858761400989eaa3`
- predictions SHA256: `e86b74dd1d7ce8621066b2b24fca860639a6d5533e4539e0e7e9732fc486756a`
- locked test accessed: **false**
- monitor used: **false**
- all three predeclared outage fractions reported: **yes**
- all three predeclared recovery arms reported: **yes**
- retraining after outage: **no**
- recalibration after outage: **no**
- rethresholding after outage: **no**

Twenty-three promoted-stack and missingness primitive tests passed before the experiment. Both public input hashes were reverified before execution.

## Frozen promoted model

Model: `IRIS_CROSSFIT_EVIDENCE_STACK_V1`

Cross-fitted positive-evidence weights reproduced the promoted architecture:

- solar: **0.126884**
- XRS: **0.244661**
- proton: **0.182343**
- stack intercept: **-4.63530**

Frozen clean thresholds:

- `MAX_TSS`: **0.0190993**
- `POD80_MIN_FAR`: **0.0160292**

Clean score-block performance:

| Policy | TSS | POD | FPR | FAR | Brier | ECE | AUPRC | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAX_TSS | 0.4257 | 0.5714 | 0.1457 | 0.9749 | 0.00635 | 0.00378 | 0.0767 | 0.8688 |
| POD80_MIN_FAR | **0.5120** | 0.7143 | 0.2023 | 0.9773 | 0.00635 | 0.00378 | 0.0767 | 0.8688 |

The `POD80_MIN_FAR` threshold was chosen on the separate threshold role; its achieved POD on this score block is 0.714 because the score distribution differs chronologically.

## Experimental design

Only genuinely observed finite causal feature cells in the score role were eligible for artificial hiding. Pre-existing unavailable cells were never converted into synthetic outages. The model was fitted once on clean fit data; OOF stack weights, calibration intercept and decision thresholds were frozen before perturbation.

Fractions: **5%, 20%, 40%** of eligible observed score-role cells.

Arms:

1. `MASK_AWARE_NO_FILL`
2. `TRAIN_FIT_MEDIAN`
3. `CAUSAL_FORWARD_FILL`

The prior weak-logistic screen had already indicated forward-fill as the strongest probability-preservation arm. This prior observation was explicitly disclosed in the preregistration, and all three arms were retained to avoid post-result cherry-picking.

## Probability preservation

Mean absolute probability drift from the clean promoted forecast:

| Hidden observed cells | No fill | Train median | Causal forward-fill |
|---|---:|---:|---:|
| 5% | 0.001156 | 0.001249 | **0.000429** |
| 20% | 0.002974 | 0.003860 | **0.001297** |
| 40% | 0.004572 | 0.007454 | **0.002217** |

Thus forward-fill produced the smallest probability perturbation at every tested random-missingness level.

Paired Brier-delta bootstrap for causal forward-fill:

| Hidden | Point delta Brier | bootstrap median | 95% interval |
|---|---:|---:|---:|
| 5% | +0.0000089 | +0.0000084 | [-0.0000048, +0.0000268] |
| 20% | +0.0000240 | +0.0000222 | [-0.0000048, +0.0000631] |
| 40% | +0.0000907 | +0.0000880 | **[+0.0000245, +0.0001731]** |

Interpretation: probability quality is effectively unchanged within uncertainty at 5–20% random cell loss, while at 40% the Brier degradation becomes detectably positive, though still small in absolute value.

## Decision-skill preservation

### MAX_TSS policy

Causal forward-fill:

| Hidden | Candidate TSS | Delta TSS | paired median delta | 95% interval |
|---|---:|---:|---:|---:|
| 5% | 0.4292 | +0.0034 | +0.0034 | [-0.1361, +0.1379] |
| 20% | **0.4799** | **+0.0542** | **+0.0517** | **[+0.00093, +0.1670]** |
| 40% | 0.3342 | -0.0915 | -0.0848 | [-0.2381, +0.0091] |

The positive 20% result is interesting but must not be promoted as a general improvement: this is an already-inspected development block, the recovery method had prior evidence, and the effect does not persist across the operational decision policy.

### POD80_MIN_FAR policy

Causal forward-fill:

| Hidden | Candidate TSS | Delta TSS | paired median delta | 95% interval |
|---|---:|---:|---:|---:|
| 5% | 0.4665 | -0.0454 | -0.0425 | [-0.1598, +0.0054] |
| 20% | 0.4161 | -0.0959 | -0.0925 | [-0.2889, +0.0847] |
| 40% | **0.2854** | **-0.2265** | **-0.2214** | **[-0.4240, -0.0487]** |

At 40% random observed-cell loss, the operational-policy TSS degradation is substantial and its paired 95% interval is entirely below zero. This is the clearest boundary discovered by the transfer.

## Comparison with simpler arms

The best recovery method is **not invariant to decision policy and outage severity**.

At 40% under `POD80_MIN_FAR`:

- no fill: delta TSS **-0.0152**, paired interval crosses zero;
- train-fit median: delta TSS **-0.1544**, interval crosses zero;
- causal forward-fill: delta TSS **-0.2265**, interval entirely below zero.

This matters. Forward-fill is the best probability-preservation method overall, but once missingness becomes severe, preserving numerical probabilities is not equivalent to preserving the operational decision boundary. A validity layer should therefore consider missingness severity directly rather than always trusting reconstructed values.

## Scientific interpretation

The evidence supports the following development hypothesis:

> Under modest random transient feature loss, a simple causal carry-forward can keep the promoted forecast probabilities close to their clean values. Under severe loss, reconstruction can produce an apparently well-calibrated-looking probability while materially changing operational decision skill. Therefore forecast validity must be evaluated separately from forecast probability, and sufficiently unsupported inputs should be marked DEGRADED or ABSTAIN rather than silently reconstructed and treated as normal.

This directly strengthens the project’s central separation between **forecast probability** and **permission to expose that probability as valid**.

## Important limitations

1. The score identities were already inspected in earlier architecture work; this is development evidence, not a fresh final test.
2. Random cell dropout is a stress test, not a faithful model of a real sensor outage. Whole-family contiguous outages must be tested separately.
3. Only 21 positive windows occur in the score block, so TSS intervals are wide.
4. No physics reconstruction was evaluated; only no-fill, train median and causal forward-fill were tested.
5. No locked test, SEPVAL result, operational certification, economic claim, company-superiority claim, award claim or breakthrough claim is supported.

## Next decisive robustness experiment

Run a separately preregistered **contiguous modality-outage benchmark** on the frozen promoted stack. Hide complete XRS or proton-family measurements over causal contiguous windows rather than random individual cells; retain all simple recovery arms; freeze model/calibration/thresholds before outage; report coverage, TSS, Brier/ECE and paired episode/bootstrap uncertainty. Use the result to design a missingness-based DEGRADED/ABSTAIN rule, but validate that rule only on evidence not used to choose it.
