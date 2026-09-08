# SHARP NRT source-equivalence result — 9 September 2026

## Decision

**FAILED — do not bridge the missing frozen SHARP quantities from a sibling NRT series.**

The bounded `IRIS_SEP_SHARP_NRT_EQUIVALENCE_PROBE_V1` was executed at head `fbe86677df8cc6d51e186ae6dac777fc8fa02798` in GitHub Actions run `34266804259`.

Artifact ID: `10072171022`  
Artifact digest: `sha256:c304e51245567c9e183e1f9d0d8d4101284aad10cdda364cc41a19ee38884596`  
Probe receipt SHA-256: `7a2d2f48059eb941849e76f5b8cfd7e3d7bcf2392625991aa037fe0d30f81015`

No training was performed, no locked test was accessed, and no forecast probability was emitted.

## What passed

On the fixed historical definitive comparison date `2024.05.10`, `hmi.sharp_720s` and `hmi.sharp_cea_720s` matched exactly on the three stored keywords:

| Keyword | finite matched rows | max absolute difference | result |
|---|---:|---:|---|
| `CMASKL` | 1361 | 0.0 | PASS |
| `MEANGBL` | 1042 | 0.0 | PASS |
| `USFLUXL` | 1361 | 0.0 | PASS |

This establishes that the definitive CCD and CEA series stored the same values for these matched keyword records on the fixed historical comparison sample.

## What failed

The live-source availability gate failed.

Both NRT series omit all three required keywords from their series metadata:

- `hmi.sharp_720s_nrt`: missing `CMASKL`, `MEANGBL`, `USFLUXL`;
- `hmi.sharp_cea_720s_nrt`: missing `CMASKL`, `MEANGBL`, `USFLUXL`.

The recent CCD-NRT query returned `1358` rows, but finite values were:

- `CMASKL`: `0`;
- `MEANGBL`: `0`;
- `USFLUXL`: `0`.

The preregistered minimum was 20 finite recent observations for every quantity. Therefore `metadata_gate_passed=false`, `recent_ccd_nrt.passed=false`, and `source_equivalence_gate_passed=false`.

## Scientific interpretation

Historical cross-series agreement cannot establish prospective availability. A quantity that is absent from the operational NRT interface cannot be treated as available merely because the definitive products agree retrospectively.

Therefore the frozen 259-position V3 interface remains:

`FROZEN_MODEL_RETROSPECTIVE_ONLY_FOR_SKILL_UNTIL_SOURCE_EQUIVALENCE_IS_ESTABLISHED`

The failed bridge must not be repaired by zero-fill, similarly named substitutions, retrospective definitive data, or hidden mixed-source reconstruction.

## Required next step

Execute the previously declared fallback: preregister and validate a **separate causal reduced-input study** built only from measurements demonstrably available at forecast time. That study is `NOAA_CAUSAL_REDUCED_INPUT_V1`; it is not V3 and cannot inherit V3's prospective identity or claims.
