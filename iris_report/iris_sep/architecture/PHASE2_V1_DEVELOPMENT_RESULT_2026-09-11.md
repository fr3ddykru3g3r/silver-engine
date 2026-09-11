# IRIS-SEP Phase II V1 development result — 2026-09-11

Status: **CLOSED — UNDERPOWERED DEVELOPMENT SIGNAL; NOT AN OPERATIONAL OR INDEPENDENT WIN**

Study: `IRIS_SEP_ONSET_FORECASTER_PHASE2_V1`

Successful workflow run: `34628549602`

Execution commit: `fe7fe95010c6da1c1595bd5ac56a542cdec4e24c`

Artifact ID: `10275033377`

Artifact ZIP SHA-256: `4fce9da1be1764b2cab4a59e6398bdf6642b8dbe6f7d5cd7246d8b24128e9984`

Protected post-2025 outcomes accessed: **false**

## What the experiment actually found

The preregistered historical Phase II experiment completed successfully after runtime-only compatibility corrections. The score cohort contained **210 rows but only one positive onset episode**. This is the dominant limitation and prevents a reliable superiority claim.

### Reference: occurrence-trained XRS baseline

- TSS: **0.4114832536**
- HSS: **0.0066148758**
- POD: **1.0000**
- FPR: **0.5885167464**
- FAR: **0.9919354839**
- Brier: **0.0045497164**
- AUPRC: **0.5000**
- AUROC: **0.9952153110**
- Confusion: TP=1, FN=0, FP=123, TN=86

### Best Phase II candidate: direct-onset engineered XGBoost

- TSS: **0.4880382775**
- HSS: **0.0089970892**
- POD: **1.0000**
- FPR: **0.5119617225**
- FAR: **0.9907407407**
- Brier: **0.0047860455**
- AUPRC: **0.0500**
- AUROC: **0.9090909091**
- Confusion: TP=1, FN=0, FP=107, TN=102

Nominal TSS change versus the reference: **+0.0765550239**.

The threshold-selected blend selected alpha `0.0`, so it collapsed exactly to the direct-onset engineered model. The direct-onset XRS-only model scored TSS `0.2009569378`; the direct-onset joint proton+XRS model scored TSS `0.0191387560`.

## Physical-unit uncertainty

The development-only bootstrap used 5,000 draws, but the score set contained only **one positive physical episode unit** and 45 quiet-block units.

Champion-minus-reference TSS:

- median delta: **+0.0765550239**
- 95% interval: **[-0.0251256281, +0.1784122610]**
- fraction of draws positive: **0.9176**

The interval crosses zero. With only one positive event, this is not adequate evidence of reliable superiority.

## Engineering/scientific interpretation

Phase II V1 did not produce the hoped-for robust onset forecaster. It produced a nominally better TSS on a severely underpowered score cohort, while FAR remained approximately 99%. Brier score, AUPRC and AUROC also did not improve over the reference. The result therefore must **not** be described as a forecasting breakthrough, state of the art, operational readiness, or independently demonstrated superiority.

The experiment does provide one useful development signal: explicitly engineered causal onset features reduced false-positive rate relative to the occurrence-trained baseline on the frozen 2017 score period. That signal is hypothesis-generating only.

## Freeze decision

Phase II V1 is closed at this result. The 2017 score period has now been inspected and must not be used for further hyperparameter, feature, blend or threshold tuning while being described as held out.

Any further forecasting experiment must be a **new preregistered study** with a genuinely uninspected evaluation design, physical-event grouping, and a minimum positive-event information floor. The protected post-2025 prospective pool remains sealed.

The frozen episode-normalized benchmark remains the stronger competition result and is not modified by this Phase II branch.
