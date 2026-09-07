# IRIS-SEP event-terminal outage result

Date: 2026-09-07

## Evidence anchor

Workflow run: `34052511341`

Training/result head: `e5ef75532684f8ff29654d242f60470ab8dd45ee`

Artifact digest:
`sha256:2e0c832d949048344649708a03644219fb01454df9e13e9b15b752d24b1a851a`

Predictions SHA-256:
`e7ce61671f625f754d4b1cb57239c6333493498f16c749fe302b83189fac75bb`

Summary SHA-256:
`acd73fb03e4edd6696cd9c64ece7e4a82ad4e9abfd5d3612f2a5209588321f08`

Independent audit status: **PASSED** with `0` mismatches across `54`
arm/policy evaluations and all `9` scenarios.

The preregistration ancestor was
`430283f0dcc702139b38e041acb8186dd064bf80`, before the result-capable runner.
No locked test or development monitor was accessed.

## What this experiment fixes

The earlier deterministic contiguous-outage benchmark happened to place every
outage in quiet score periods. It was useful for false-alert/probability-drift
analysis but could not establish event preservation.

This benchmark deliberately evaluates every eligible positive NEW-SEP score
issue plus a deterministic label-only matched quiet control. Each scenario has
42 terminal rows: 21 positives and 21 quiet controls. A 24/72/168-hour outage
ends at the forecast issue time. Causal forward-fill may use only values from
before the declared outage started.

## Frozen clean reference

On the full score block the promoted `IRIS_CROSSFIT_EVIDENCE_STACK_V1` has:

- MAX_TSS threshold: `0.019099288799212903`
- MAX_TSS score TSS: `0.4257124989`
- MAX_TSS POD: `0.5714285714`
- MAX_TSS FAR: `0.9748953975`
- POD80/min-FAR threshold: `0.01602923550520927`
- POD80-policy score TSS: `0.5119717681`
- POD80-policy POD: `0.7142857143`
- POD80-policy FAR: `0.9773413897`
- AUPRC: `0.0767198429`
- AUROC: `0.8688168200`
- Brier: `0.0063475567`
- ECE: `0.0037800550`

The benchmark-primary decision policy remains **MAX_TSS**. POD80/min-FAR is a
secondary reliability diagnostic and is not substituted as the primary policy
after seeing these results.

## Primary MAX_TSS affected-row result

Values below are TSS difference versus the clean forecast on the same 21 event
+ 21 quiet-control terminal rows.

| Missing family | Duration | Causal forward-fill | Train-fit median | Mask-aware no-fill |
|---|---:|---:|---:|---:|
| Proton | 24 h | -0.0952 | -0.0952 | -0.1429 |
| Proton | 72 h | **-0.2857** | -0.0952 | -0.1429 |
| Proton | 168 h | -0.1905 | -0.0952 | -0.1429 |
| XRS | 24 h | -0.0476 | 0.0000 | **-0.3333** |
| XRS | 72 h | +0.0476 | 0.0000 | **-0.3333** |
| XRS | 168 h | -0.2381 | 0.0000 | **-0.3333** |
| XRS + proton | 24 h | -0.0952 | -0.1905 | **-0.3333** |
| XRS + proton | 72 h | -0.0952 | -0.1905 | **-0.3333** |
| XRS + proton | 168 h | -0.3333 | -0.1905 | **-0.3333** |

Important bootstrap findings:

- proton 72 h causal forward-fill was significantly harmful under MAX_TSS:
  paired 95% interval approximately `[-0.5590, -0.0225]`;
- XRS mask-aware no-fill was significantly harmful at all three durations:
  interval approximately `[-0.6039, -0.0406]`;
- combined XRS+proton mask-aware no-fill had the same significant negative
  interval;
- no simple recovery arm established a clean general non-degradation result
  across all event-bearing outage conditions under the benchmark-primary policy.

## Secondary POD80 observations

Some simple-fill arms look better under the secondary POD80/min-FAR policy. For
example train-fit median during proton-only loss increased terminal-row TSS from
`0.1905` to `0.3810`, with a positive paired interval lower bound around
`+0.0454`. Proton 72 h forward-fill also improved the point estimate.

Those observations are retained, but they **do not override the frozen primary
MAX_TSS policy** and therefore cannot be used to declare a normal validated
recovery path.

## Decision

### NORMAL

No imputation/reconstruction method tested here is admitted to NORMAL operation.
A missing critical modality may not silently produce a normal-looking full-stack
forecast.

### DEGRADED

Simple recovery outputs remain research diagnostics. They may be displayed only
as explicitly DEGRADED if a higher-level operator policy chooses to expose them;
the provenance must state the recovery method and outage state.

### ABSTAIN

ABSTAIN remains the safe default when no separately validated availability-
conditioned fallback or real alternate observed source is available.

### Train-fit median

Do **not** promote train-fit median as the generic outage solution. Although it
performed well for proton-only event-terminal rows under POD80 and did not harm
XRS terminal TSS under MAX_TSS, the earlier quiet contiguous-outage experiment
showed pathological false-alert inflation for long XRS/combined outages. The two
experiments together reject it as a universal operational recovery rule.

### Causal forward-fill

Do **not** promote causal forward-fill to NORMAL. It minimized probability drift
in the quiet contiguous benchmark, but event-bearing performance is unstable and
proton 72 h was significantly harmful under the primary policy.

## Next systems experiment

The next fallback will not invent missing measurements. Instead, IRIS will train
small availability-conditioned evidence stacks using the existing specialist
experts:

- full data: solar + XRS + proton;
- XRS unavailable: solar + proton;
- proton unavailable: solar + XRS;
- XRS and proton unavailable: solar-only fallback.

Each fallback is trained only from fit-era out-of-fold predictions, calibrated
on the calibration role and thresholded on the threshold role before outage
evaluation. Runtime feed loss merely selects an already-trained fallback. This
is an engineering reliability layer, not a new large-model search.
