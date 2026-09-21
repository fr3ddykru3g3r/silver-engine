# IRIS-SEP — START HERE

**Active branch:** `codex/iris-sep-operational-reproducibility-20260921`  
**Active study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Frozen:** 21 September 2026  
**Status:** `PHASE_A_COMPLETE — PHASE_B BLOCKED BY PREREGISTERED INTERFACE GATE`

## Research question

> **Can a retrospective 24-hour solar energetic particle forecast be reproduced using only predictor values that were genuinely available in an equivalent form before each forecast issue time?**

## Headline result

The immutable replay artifact contains **259 exact predictors**. The completed interface audit finds:

- **248 / 259 (95.75%)** are directly classified as `SCHEMA_MISMATCH` or `RETROSPECTIVE_ONLY` because their frozen construction differs from an issue-time/NRT construction or explicitly uses retrospective reconstruction;
- **11 / 259 (4.25%)** remain `UNVERIFIED_LATENCY` (flare family);
- **0 / 259** currently meet every criterion for exact `VERIFIED` issue-time equivalence.

Do **not** translate this to “95.75% of measurements were unavailable in real time.” Many underlying sensors/products have operational streams. The result is about exact feature construction and causal reproducibility.

## Read in this order

1. `ACTIVE_PROJECT_2026-09-21.md` — question, result and IB Physics-level explanation.
2. `OPERATIONAL_REPRODUCIBILITY_RESULTS_2026-09-21.md` — complete Phase A result and why Phase B is stopped.
3. `SOURCE_AVAILABILITY_EVIDENCE_2026-09-21.md` — evidence ledger for every source family.
4. `NOVELTY_AUDIT_OPERATIONAL_REPRODUCIBILITY_2026-09-21.md` — hostile prior-art audit and bounded novelty claim.
5. `config/operational_reproducibility_preregistration_2026-09-21.json` — rules frozen before result acceptance.
6. `PREREGISTRATION_EXECUTION_NOTE_2026-09-21.md` — schema bug found/fixed and execution deviations.
7. `config/frozen_joint_feature_schema_v1.json` — canonical ordered 259-predictor interface and cryptographic provenance.
8. `config/operational_source_manifest_v1.json` — source-family evidence/status ledger.
9. `audit_operational_reproducibility_20260921/interface_audit_v1.json` — machine-readable audit summary.
10. `tools/audit_operational_reproducibility_v1.py` — fail-closed audit implementation.
11. `tests/test_audit_operational_reproducibility_v1.py` — regression tests for the exact schema rules.
12. `docs/operational_reproducibility_ib_physics_paper.tex` — internal IB-level paper scaffold; evidence-locked, not submission text.

## Frozen provenance

- GitHub Actions artifact id: `10137507101`
- artifact digest: `sha256:81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
- artifact `feature_schema.json` SHA-256: `b70c1b9137cfe7493787f8ddcc328ce81e1153314bbcade8d12013c94948fcf1`
- ordered 259-feature SHA-256: `cf0fc9e07b1e9b173ad0c330fb021452b5a527ab796dfdbd9282c29a5e3047d4`
- upstream 24-hour table SHA-256: `4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb`
- upstream revision: `yuyian/SEP-Prediction-V2@e138dcd72c1952a00e11e1a0b025337f9e7c93fb`

## Why Phase B is blocked

The preregistration forbids silently replacing a structurally non-equivalent retrospective input with zero, a later definitive value, or a similarly named NRT field and still calling the result the same model.

Since the exact interface does not pass the causal-equivalence gate, an honest 259-feature same-model operational replay cannot be run. A future NRT-only model must be explicitly declared as a **new causal interface**, frozen before outcome evaluation, and judged separately.

Protected post-`2025-09-10T00:00:00Z` outcomes remain sealed.

## Test status

Corrected local audit tests:

```text
4 passed in 0.09s
```

The correction was necessary because the immutable schema proved that present-time fields such as `SHARP_label`, `Flare_label`, `ProtonFlux_label` and `XRS_label` are predictors. The audit now uses the exact frozen list rather than removing all `*_label` columns by name pattern.

## Historical material

Earlier onset/persistence, missing-sensor, direct-onset, blackout-horizon and other branches are preserved for provenance only. They are not the current novelty narrative.
