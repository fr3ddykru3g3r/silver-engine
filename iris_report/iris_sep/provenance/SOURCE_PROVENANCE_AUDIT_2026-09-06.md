# Source provenance and causality audit — 2026-09-06

## Purpose

This audit answers a narrower question than “is the public table useful?”:

> For each predictor family used by IRIS-SEP, can a finite value in the model-ready daily table be demonstrated to be a native measurement available by the forecast issue time?

Current answer: **not for every family/cell**.

Pinned upstream source: `yuyian/SEP-Prediction-V2@e138dcd72c1952a00e11e1a0b025337f9e7c93fb`.

Until row/cell lineage is reconstructed, IRIS must not equate `finite` with `observed` in experiments whose scientific claim requires native hidden truth.

## Provenance states

- `NATIVE_OBSERVED`: direct measurement with demonstrated observation/publication time <= issue time.
- `ALTERNATE_SOURCE_OBSERVED`: real measurement from another declared source, causally available by issue time.
- `RECONSTRUCTED_CAUSAL`: derived from only information available by issue time under a frozen causal transform.
- `RECONSTRUCTED_NONCAUSAL_OR_RETROSPECTIVE`: transform uses future values, full-period overlap fitting, or another retrospective operation.
- `STRUCTURAL_UNAVAILABLE`: source/feature did not exist or is outside declared source support.
- `UNKNOWN`: current aggregate artifact does not preserve enough lineage to assign one of the above safely.

`UNKNOWN` is intentionally fail-closed. It is not equivalent to missing and not equivalent to observed.

## Family audit

### XRS

Upstream file: `Rcode/Dataset_Preprocessing/XRay-rsb_Impute.R`.

Observed behavior at the inspected stage:

- raw GOES XRSB samples are grouped to five-minute bins;
- sub-five-minute samples within each bin are averaged;
- a complete five-minute timeline is created;
- bins without samples remain `NA` in this script.

This particular script does not itself linearly interpolate XRS gaps. However, this does **not** prove that every finite daily XRS aggregate is native, because the complete downstream path from five-minute XRS to the released daily table has not yet been reconstructed at cell-level provenance resolution.

Current aggregate-table state: `UNKNOWN` unless native lineage is separately demonstrated.

### Historical >10 MeV proton flux

Upstream files:

- `Rcode/Dataset_Preprocessing/GOES-HAPI-proton-flux_fusion.R`
- `Rcode/Dataset_Preprocessing/OMNI-HAPI-proton-flux_fusion.R`

Confirmed retrospective operations:

1. Negative HAPI values are set to missing.
2. HAPI gaps are filled using `na_interpolation(..., option="linear")`. Ordinary linear interpolation can use observations after the missing timestamp.
3. OMNI missing values are also linearly interpolated.
4. An `lm(hapi_flux ~ omni_flux)` mapping is fitted on the complete HAPI/OMNI overlap period.
5. That overlap-fit mapping is used to predict HAPI-equivalent values in the pre-HAPI OMNI-only era and to fill HAPI gaps.
6. A `HAPI_from_OMNI` flag exists in the intermediate OMNI/HAPI product, but the reviewed daily model-ready interface has not been demonstrated to preserve complete row/cell origin flags.

Causality consequence:

- a backcast value produced by a model fit using a future overlap era is not a strictly prospective historical observation;
- a linearly interpolated value cannot be assumed causal without proving the interpolation only used past points;
- therefore finite proton aggregate cells cannot all be labelled native observations.

Current aggregate-table state: `UNKNOWN`, with known existence of `RECONSTRUCTED_NONCAUSAL_OR_RETROSPECTIVE` values upstream.

### Magnetic active-region features (SHARP / SMARP fusion)

Upstream file: `Rcode/Dataset_Preprocessing/sharp-smarp_fusion.R`.

Confirmed retrospective operations:

1. Missing SHARP and SMARP feature values are filled feature-by-feature using data.table rolling joins with `roll = "nearest"` within the active region. “Nearest” is not restricted to an earlier timestamp and can therefore use a future observation relative to the missing row.
2. SHARP and SMARP are aligned in 96-minute windows during their overlap.
3. Stepwise linear models are fitted on the overlap to map SMARP predictors to SHARP shared features.
4. Additional stepwise models predict SHARP-specific features using overlap-derived predictors.
5. These overlap-fit models are then used to construct SHARP-like values in the pre-SHARP SMARP era.
6. The intermediate fused output records `From_SMARP`, but complete preservation of this provenance through the final daily aggregate table has not yet been demonstrated.

Causality consequence:

- nearest-time imputation is not guaranteed past-only;
- pre-SHARP backcasts use models fitted using later overlap data;
- therefore a finite magnetic aggregate cell cannot automatically be described as a native forecast-time observation.

Current aggregate-table state: `UNKNOWN`, with known existence of `RECONSTRUCTED_NONCAUSAL_OR_RETROSPECTIVE` values upstream.

### Flare/CME and other solar-context families

The current development candidate’s `BASE_SOLAR` family can contain multiple upstream sources. Their complete observation time, publication time, reconstruction flag and fit interval have not yet been normalized into one machine-readable lineage table.

Current aggregate-table state: `UNKNOWN` until audited per source family.

## Consequences for existing experiments

### Existing forecast-development results

The already-inspected development results remain valid as **retrospective model performance on the published aggregate dataset**. This audit does not erase or numerically alter those results.

It does restrict the claim:

- chronological model splits alone do not prove that every predictor was causally constructible at the historical issue time;
- the results therefore cannot yet be presented as a clean prospective-replay forecast using only native forecast-time data.

### Random-cell missingness experiment

The previous stress test selected cells based on finiteness. It may still be described as deletion of finite cells in the released aggregate table.

It must **not** be upgraded into a claim that known native sensor observations were hidden unless native provenance is established for those cells.

### Daily modality outage experiment

A daily aggregate-input outage can still test how the promoted stack behaves when entire model-input families become unavailable.

However, hidden-cell reconstruction quality must not be described as reconstruction of native sensor truth merely because the original aggregate value was finite. The experiment must report the provenance limitation and treat native/reconstructed/unknown truth separately when that distinction is material.

### Future raw-sensor outage experiment

A genuine sensor-outage claim requires:

1. high-cadence native source data;
2. explicit observation and publication times;
3. past-only gap simulation;
4. a causal reaggregation pipeline fit only on allowed development periods;
5. row/cell provenance carried into inference.

That is the preferred post-submission scientific path.

## Required implementation rule

Any new provenance-sensitive runner must receive or construct an explicit provenance mask. It may not use `np.isfinite(values)` as the sole definition of `NATIVE_OBSERVED`.

If lineage is unavailable, the provenance state is `UNKNOWN`, and the runner must either:

- exclude the cell from a native-truth reconstruction metric while retaining the forecast case for coverage accounting; or
- report the experiment explicitly as an aggregate-interface perturbation rather than a native-sensor reconstruction test.

## What remains unresolved

- exact downstream transformations from each five-minute/upstream product into every daily model-ready column;
- publication-time latency for all source families;
- whether each intermediate provenance flag survives into the released aggregate table;
- full fit intervals for every harmonization transform;
- causal alternatives for retrospective interpolation/harmonization.

These unresolved items are scientific limitations, not software-test failures.