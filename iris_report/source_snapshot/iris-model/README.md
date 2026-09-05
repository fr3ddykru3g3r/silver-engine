# IRIS / ISEF magnetogram model

## Scientific question
Which magnetic-distribution constraints make synthetic LOS magnetograms transfer useful information to a downstream 24 h M1+ flare forecaster, under matched exposure and connected-region-disjoint chronology?

## Data status

The repository contains code for both the earlier conservative Gate-0 slice and
the v2 historical rolling-fold workflow. Always use the checksumed evidence
bundle named by the workflow/protocol being run; do not mix manifests or reuse a
fold artifact from another protocol. The current v2 outer-fold definition is
recorded in `V2_EVALUATION_FREEZE_DECISION_2026-08-26.md`.

The historical bundle contains JSOC `T_REC` values in TAI. Before model loading,
the bundle must be repaired with
`iris-gate0-data/repair_historical_tai_manifest.py`; `data.build_records` fails
closed if the resulting `tai_repair_audit.json` receipt is absent or not `PASS`.
This prevents an hourly metadata join from silently using TAI text as UTC.

The latest BASE generator artifacts did not pass the corrected train-only
generic-fidelity gate. The recovery configuration is documented in
`GENERATOR_RECOVERY_GATE_2026-08-27.md`; no downstream transfer claim is
authorized until that gate passes.

The earlier conservative smoke slice reports only 17/5/6 independent positive
connected regions in train/validation/test. That count is not the v2 historical
rolling-fold count and must not be reused across protocols. It was sufficient
to validate the pipeline, but not to support a high-power final ISEF experiment;
do not interpret the smoke/full one-year forecast as the project result without
the applicable checksumed evidence bundle and manual audits.

## Input representation
Each definitive `hmi.sharp_cea_720s` LOS magnetogram is mapped to a fixed 256 Mm x 256 Mm physical field of view centered on the unsigned-flux centroid, then resampled to 128 x 128. The resulting grid is exactly 2 Mm/pixel. Field values are clipped to +/-3000 G and signed-asinh normalized. This keeps the PIL statistic in physical G/Mm units and avoids variable-patch-size geometry as a model cue.

## Downstream forecaster
`FlareCNN` is a deliberately compact ~2.8M-parameter residual CNN. It receives only the single LOS field channel. The frozen downstream matrix uses the same focal-BCE architecture, AdamW budget, validation-only threshold selection, and connected-region cluster bootstrap for every arm; only `Rw` uses balanced positive weighting. The clean CNN is intentionally used instead of a large foundation model so the generator ablation is the experimental variable. See `V2_DOWNSTREAM_MATRIX_FREEZE_2026-08-27.md`.

## Generator
`ConditionalUNet` is a pixel-space conditional DDPM. It is conditioned on flare label and active-region latitude. Full settings: 128 x 128, base width 48, 400 cosine diffusion steps, EMA, AdamW. Training uses the complete real training partition with a fixed positive sampling fraction so every ablation sees the same exposure and sample order for a given seed.

## Physics-ablation matrix
Use a 2 x 2 factorial rather than a one-direction ladder:

| code | population constraint | PIL constraint |
|---|---:|---:|
| `base` | no | no |
| `hj` | yes | no |
| `pil` | no | yes |
| `hj_pil` | yes | yes |

This cleanly estimates the Hale/Joy population effect, the PIL-gradient effect, and their interaction.

### Population constraint
The differentiable descriptor uses soft positive/negative flux centroids and matches the real training distribution separately by hemisphere using energy distance. We match real training descriptors rather than hard-coding east/west image orientation, so the loss respects the actual SHARP storage convention while capturing polarity ordering and tilt at the population level.

### PIL constraint
A differentiable strong-gradient PIL statistic uses separate smooth positive and negative memberships, opposite-polarity neighborhood contact, and central-difference |grad B| in G/Mm. The batch distribution is matched to real training positives with energy distance after log compression. The implementation explicitly avoids the incorrect `sigmoid(+Bi*Bj/T)` polarity weight.

## Downstream synthetic-data comparison
After generators pass image/physics validation, create exactly the same number of added positive examples for every augmentation arm. The authoritative primary arms are `R`, `Rw`, `D`, `L0`, `L2`, and `L3`, mapped to `real`, `real_weighted`, `duplicate`, `base`, `hj`, and `hj_pil`. Planned auxiliary arms:

1. real-only baseline;
2. real-positive duplication control;
3. unconstrained synthetic (`base`);
4. population-only synthetic (`hj`);
5. PIL-only synthetic (`pil`);
6. population+PIL synthetic (`hj_pil`);
7. optional destructive controls (polarity flip / shuffled patches) if compute permits.

Every arm uses the identical forecaster architecture, optimizer, total number of update steps, synthetic count, seeds and frozen test set. Primary uncertainty is clustered by connected physical region, not image row.

## Metrics
Primary: TSS. Secondary: HSS, AUROC, Brier score, ECE/reliability. Report seed distributions and connected-region bootstrap CIs. Never claim significance from the thousands of hourly rows as if they were independent active regions.

## Current validation
The end-to-end smoke workflow has already compiled all code, downloaded real JSOC FITS, trained the real-only CNN for two epochs, and executed two gradient steps for all four generator ablations. Smoke results are engineering validation only, not scientific results.
