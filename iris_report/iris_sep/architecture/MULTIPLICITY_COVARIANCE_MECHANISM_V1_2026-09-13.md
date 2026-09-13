# Multiplicity-covariance mechanism V1 — 2026-09-13

Status: **VERIFIED EXACTLY ON THE IMMUTABLE FIXED-REPLAY ARTIFACT**

Workflow run: `34741573375`

Execution commit: `c41871d7e44f19c1fd0a6deb5fd9809f110f06a5`

Evidence artifact ID: `10312501241`

Evidence artifact digest: `sha256:4ce3e72799b5c0da13c0916fb498f1539b4478baf2cdac4554c29e28f5884d8f`

Frozen fixed-replay archive used: artifact `10137507101`, SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`.

Protected post-2025 outcomes accessed: **false**.

## Exact identity

For physical positive episode `i`, define:

- `n_i`: number of mapped positive forecast windows representing that episode;
- `r_i`: fraction of those windows on which a fixed alert rule fires.

Then the row-weighted positive detection rate and episode-normalized positive detection rate obey

`POD_row - POD_episode = Cov(n_i, r_i) / E[n_i]`.

If the negative cohort and its weighting are unchanged between the two views, the false-positive rate cancels exactly. Therefore

`TSS_row - TSS_episode = Cov(n_i, r_i) / E[n_i]`.

This is an algebraic identity, not an approximation or fitted relationship.

## Frozen-data verification

The matched cohort contains 85 represented physical positive episodes and 197 mapped positive windows, giving mean positive-window multiplicity `E[n_i] = 2.3176470588`.

| Fixed comparator | Cov(n, r) | Corr(n, r) | Exact mapped-minus-normalized TSS |
|---|---:|---:|---:|
| XGBoost joint | 0.2441110562 | +0.4409371 | **+0.1053271055** |
| Past-proton >=10 proxy | 0.3574766848 | +0.7147619 | **+0.1542412092** |
| XGBoost no-proton | -0.0135238095 | -0.0247576 | **-0.0058351462** |
| Elastic net joint | 0.1391711979 | +0.3617168 | +0.0600484864 |
| Elastic net no-proton | 0.0305219970 | +0.0942782 | +0.0131693896 |
| Fit-prevalence climatology | 0 | undefined | 0 |

The maximum numerical error between the exact covariance identity and the stored artifact TSS differences was `9.54e-17`.

## Scientific interpretation

For the joint XGBoost and the proton-persistence proxy, longer/more-repeated physical episodes were also easier for the fixed alert rule. Ordinary row-weighted scoring therefore gives those easier episodes more positive weight. Episode normalization removes that mechanical weighting advantage.

The no-proton XGBoost provides an important control: its multiplicity/detection covariance is approximately zero and slightly negative, so episode normalization does not automatically lower every model's score. The direction of the score change depends on the empirical relationship between event multiplicity and detection success.

This turns the observed multiplicity effect from a descriptive score change into an explicit mechanism: the inflation term is exactly the covariance between how many times a physical episode is counted and how easy that episode is for the fixed classifier, divided by mean episode multiplicity.

## Claim boundary

This verification uses already-frozen, development-exposed historical predictions. It strengthens the mechanism explanation of the primary benchmark but is not new independent forecast validation, prospective evidence, or a state-of-the-art performance claim. No model was refit, no threshold was changed, and no protected outcomes were inspected.
