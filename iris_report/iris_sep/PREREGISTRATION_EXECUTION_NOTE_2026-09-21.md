# Preregistration execution note

**Study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Frozen preregistration:** `config/operational_reproducibility_preregistration_2026-09-21.json`

The preregistration itself is preserved unchanged after execution. This note records implementation details discovered while applying it.

## Frozen-interface correction

The initial audit utility inferred non-predictors partly by the suffix `_label`. That inference was contradicted by the immutable replay artifact: the frozen joint interface contains present-time label indicators (`SHARP_label`, `SHARP_AR_label`, `Flare_label`, `DONKICME_label`, `CDAWCME_label`, `ProtonFlux_label`, `XRS_label`) as predictors.

Before accepting any audit result, the code was changed to treat `config/frozen_joint_feature_schema_v1.json` as the canonical predictor list. A raw-table fallback now implements the original replay exclusions exactly: `window_begin`, `window_end`, `OSEP_label`, `GSEP_label`, and `Future_*` only.

This is a schema-recovery correction, not an outcome-driven change. No protected SEP outcome was opened or used.

## Phase A execution

Phase A completed on the exact 259 predictors. Machine-readable result: `audit_operational_reproducibility_20260921/interface_audit_v1.json`.

Headline counts: `VERIFIED` 0; `SCHEMA_MISMATCH` 222; `RETROSPECTIVE_ONLY` 26; `UNVERIFIED_LATENCY` 11.

## Phase B gate

The preregistration permits an identical-schema replay only when issue-time availability and schema equivalence are demonstrated. It also forbids silently substituting later definitive values, zero-filled structural absence, or merely similar NRT fields.

Because no exact archived predictor currently satisfies the complete `VERIFIED` rule, Phase B is recorded as `BLOCKED_BY_PREREGISTERED_INTERFACE_GATE`. No causal skill score is generated. No reduced model is retroactively selected.

## Protected data

The protected post-2025-09-10 outcome cohort remains sealed.
