# IRIS / ISEF magnetogram model v1

## Scientific question
Which magnetic-distribution constraints make synthetic LOS magnetograms transfer useful information to a downstream 24 h M1+ flare forecaster, under matched exposure and connected-region-disjoint chronology?

## Frozen real-data slice
The current Gate-0 evidence artifact is the conservative 2025-08-25..2026-08-23 slice. It contains 17,063 primary hourly samples after QUALITY==0, |CMD|<=30 deg, connected-region chronology and 36 h split buffers. Split hash: `be42e4d9d0644cb2e24788aeda7c381f383e41cb2e85d8781a009880cb7750ae`.

Important: only 17/5/6 independent positive connected regions occur in train/validation/test. This is enough to validate the pipeline, but is not yet a high-power final ISEF experiment. Do not interpret the smoke/full one-year forecast as the project result without resolving archival-data eligibility and the manual audits.

## Input representation
Each definitive `hmi.sharp_cea_720s` LOS magnetogram is mapped to a fixed 256 Mm x 256 Mm physical field of view centered on the unsigned-flux centroid, then resampled to 128 x 128. The resulting grid is exactly 2 Mm/pixel. Field values are clipped to +/-3000 G and signed-asinh normalized. This keeps the PIL statistic in physical G/Mm units and avoids variable-patch-size geometry as a model cue.

## Downstream forecaster
`FlareCNN` is a deliberately compact ~2.8M-parameter residual CNN. It receives only the single LOS field channel. It is trained with weighted BCE, AdamW, cosine LR decay, validation-only threshold selection, and test uncertainty via connected-region cluster bootstrap. The clean CNN is intentionally used instead of a large foundation model so the generator ablation is the experimental variable.

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
After generators pass image/physics validation, create exactly the same number of added positive examples for every augmentation arm. Planned arms:

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
