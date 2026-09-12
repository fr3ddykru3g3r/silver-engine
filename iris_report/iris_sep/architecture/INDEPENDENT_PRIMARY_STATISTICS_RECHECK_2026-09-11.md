# Independent primary-statistics recheck — 2026-09-11

**Target artifact:** SEP-PRISM fixed-model replay `10137507101`  
**Published archive SHA-256:** `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`  
**Recheck implementation:** `tools/recheck_sep_prism_primary_contrasts_v1.py`

## Independence boundary

This recheck reads only the persisted matched predictions and shared physical-unit bootstrap tensors in the frozen artifact. It does **not** import the benchmark runner, model-fitting functions, threshold-selection functions or supplied verifier.

It therefore tests whether the stored prediction/evaluation evidence reproduces the reported point TSS values and three primary paired contrasts. It does not reconstruct the original model fitting from raw features, because the replay archive does not contain all objects needed for that stronger claim.

## Inputs verified

- artifact archive SHA-256: exact match;
- `matched_episode_sensitivity_predictions.csv`: 45,066 rows total = 7,511 rows for each of six models;
- `shared_bootstrap_draws.npz`: 10,000 draws;
- positive units: 85 distinct matched onset episodes;
- negative units: 1,080 quiet blocks.

## Independently recomputed point TSS

| Model | Mapped occurrence | Episode-normalized | New onset |
|---|---:|---:|---:|
| Joint XGBoost | 0.7264546541 | 0.6211275486 | 0.4372339912 |
| Past-proton >=10 pfu proxy | 0.6012244093 | 0.4469832002 | 0.0928935643 |

## Independently recomputed primary paired contrasts

| Contrast | Median | 2.5% | 97.5% | Fraction < 0 |
|---|---:|---:|---:|---:|
| Joint XGBoost: episode-normalized minus mapped | -0.1037441709 | -0.1461746092 | -0.0631648922 | 1.0000 |
| Joint XGBoost: onset minus episode-normalized | -0.1833333333 | -0.2466386555 | -0.1250700280 | 1.0000 |
| Past-proton proxy: onset minus mapped | -0.5061665246 | -0.5666753045 | -0.4439734924 | 1.0000 |

The values match the frozen artifact's published primary contrast table to machine precision.

## Statistical interpretation

The all-negative draw fraction is not reported as a classical p-value. The preregistered evidence is the paired percentile interval for each frozen contrast. The recheck also does not introduce ANOVA or a new post-result statistical gate. Its purpose is reproducibility only.

## Reproduction command

```bash
python iris_report/iris_sep/tools/recheck_sep_prism_primary_contrasts_v1.py \
  --artifact-zip sep_prism_fixed_model_replay_v1.zip \
  --assert-published
```

`--assert-published` fails closed if the archive SHA-256 or any of the three frozen primary median/interval values changes beyond numerical tolerance.
