# Source-availability evidence — frozen SEP-PRISM interface

**Study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Audit date:** 21 September 2026  
**Evidence rule:** this ledger asks whether the **exact frozen predictor construction** is reproducible at forecast issue time. A source being physically measurable, or having a present-day near-real-time endpoint, is not enough by itself.

## Frozen interface provenance

The audit is tied to the immutable external-model replay artifact produced by GitHub Actions run `34438987251` at repository head `296f302111371e8d421fb5f2a4569ea2c06bd6e1`.

- artifact id: `10137507101`
- artifact digest: `sha256:81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
- artifact `feature_schema.json` SHA-256: `b70c1b9137cfe7493787f8ddcc328ce81e1153314bbcade8d12013c94948fcf1`
- ordered 259-predictor list SHA-256: `cf0fc9e07b1e9b173ad0c330fb021452b5a527ab796dfdbd9282c29a5e3047d4`
- pinned upstream repository revision: `yuyian/SEP-Prediction-V2@e138dcd72c1952a00e11e1a0b025337f9e7c93fb`
- upstream 24-hour table SHA-256: `4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb`

Applying the original replay extraction rule to the pinned 274-column upstream table reproduces the same ordered 259 predictors exactly. The canonical list is stored in `config/frozen_joint_feature_schema_v1.json`.

## Classification rule

A predictor is `VERIFIED` only if evidence supports source identity, physical variable, measurement-time semantics, issue-time/first-seen availability, matching units, matching field definition, known revision/processing policy, and no future/outcome-dependent information.

`SCHEMA_MISMATCH` means an operational/NRT physical source may exist, but the exact retrospective feature construction is not equivalent to the issue-time construction. `RETROSPECTIVE_ONLY` means the frozen value explicitly uses retrospective cataloguing, imputation, reconstruction, or post-event processing. `UNVERIFIED_LATENCY` means the underlying source is plausible for operations, but historical first-seen/revision timing is not proven for the exact field. It does **not** mean that a source was unavailable.

## Family-by-family evidence

| Family | Predictors | Audit status | Why the exact frozen construction fails/pends |
|---|---:|---|---|
| SHARP | 107 | `SCHEMA_MISMATCH` | Upstream acquisition prefers definitive SHARP and the fusion performs nearest-time imputation and SHARP/SMARP reconstruction. Definitive HARP geometry uses complete active-region life history and is produced later; NRT and definitive HARPNUMs differ. |
| SHARP_AR | 107 | `SCHEMA_MISMATCH` | Same product mismatch plus retrospective magnetic fusion; the exact archived region subset must be rebuilt causally. |
| Flare | 11 | `UNVERIFIED_LATENCY` | Archived HEK events are used to derive duration, rise time and strength, but historical first-seen/revision timestamps for every exact field are not yet established. |
| DONKI_CME | 12 | `RETROSPECTIVE_ONLY` | The frozen CDAWDONKI table uses kNN imputation, cross-catalog pairing and stepwise reconstruction of DONKI-like quantities from CDAW. |
| CDAW_CME | 14 | `RETROSPECTIVE_ONLY` | CDAW is a manually curated catalogue and the frozen pipeline uses retrospective imputation/fusion; later historical revisions exist. |
| ProtonFlux | 4 | `SCHEMA_MISMATCH` | GOES proton observations exist operationally, but the frozen history stitches satellites, linearly interpolates gaps and fills gaps from HAPI archives. |
| XRS | 4 | `SCHEMA_MISMATCH` | GOES XRS exists operationally, but the frozen history stitches satellites and linearly interpolates gaps; NOAA documents distinct operational and retrospectively corrected science-quality XRS products. |

## Direct source/code trail

Pinned upstream revision: `https://github.com/yuyian/SEP-Prediction-V2/tree/e138dcd72c1952a00e11e1a0b025337f9e7c93fb`

Relevant code: `Pycode/fetch_solar_data.ipynb`; `Rcode/Dataset_Preprocessing/sharp-smarp_fusion.R`; `Rcode/Dataset_Preprocessing/CDAW_DONKI-fusion.R`; `Rcode/Dataset_Preprocessing/GOES-Sats_Fusion.R`; `Rcode/Dataset_Preprocessing/GOES-HAPI-proton-flux_fusion.R`; and `Rcode/Data_Aggregation.R`.

External technical evidence:

1. Bobra, M. G. et al. (2014), *The Helioseismic and Magnetic Imager (HMI) Vector Magnetic Field Pipeline: SHARPs*. Solar Physics 289, 3549–3578. DOI `10.1007/s11207-014-0529-3`. https://link.springer.com/article/10.1007/s11207-014-0529-3
2. NOAA NCEI, *GOES 1–15 Space Weather Instruments*. https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments
3. NOAA NCEI, *GOES-R Series Level 1b EXIS X-Ray Sensor (XRS) Product*. https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncei.swx%3Aexis-l1b-sfxr-goesr
4. NASA CCMC, *SHINE/ISWAT/ESWW SEP Model Validation Challenge*. https://ccmc.gsfc.nasa.gov/challenges/sep/
5. Yu, Y. et al. (2026), *Realtime forecasting of solar energetic particle event and proton flux using multi-source solar observations and multi-task deep learning*. DOI `10.1038/s41598-026-66110-2`. https://www.nature.com/articles/s41598-026-66110-2

## What this evidence does and does not establish

It establishes that the **exact 259-column retrospective interface is not yet demonstrated to be an issue-time-reproducible operational interface** and that most columns have a direct construction-level reason why the archived value is not the same object as a raw/NRT issue-time value.

It does **not** establish that the original published model is invalid, that the underlying instruments were unavailable, or that a separately rebuilt real-time model would perform poorly.
