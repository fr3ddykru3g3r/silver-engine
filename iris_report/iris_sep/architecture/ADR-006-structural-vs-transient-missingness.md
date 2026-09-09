# ADR-006 — Structural versus transient missingness

Status: **Accepted for preregistered train-only missingness research**

Date: 2026-09-05

## Context

The existing train-only structural audit found severe source-era dependence in
SHARP aggregate availability. The ten most-missing SHARP aggregates are absent
from all 1986–1999 rows and remain 71.89% absent in 1999–2013 rows. The current
coarse magnetic-availability flag can therefore look populated while the actual
measurements needed by the model are unavailable.

A generic `missing value -> physics simulation -> filled value` design would
confound two different scientific situations:

1. an instrument/source regime in which the quantity was not authoritatively
   available at all; and
2. a quantity that is normally supported in the regime but is temporarily
   missing, delayed or stale.

The first is structural unavailability. The second is a transient gap.

## Decision

IRIS-SEP will represent every missing candidate feature/modality as one of:

- `STRUCTURAL_UNAVAILABLE`;
- `TRANSIENT_MISSING`;
- `OBSERVED`.

The classification must come from an authoritative source/instrument
availability manifest and forecast-time provenance. Missingness frequency alone
must not be used to infer that a feature was structurally unavailable.

### Structural unavailability

- Never impute it and relabel the output as an observed measurement.
- Handle it with source-era indicators, mask-aware models, era-specific support
  rules, fallback, or abstention.
- A simulated field may be studied as a separate model-derived feature only if
  its own causal provenance and validation are explicit; it does not backfill
  historical sensor truth.

### Transient missingness

- Eligible for deliberately masked reconstruction experiments inside an
  authoritatively supported regime.
- Compare no-reconstruction/mask-aware handling, causal forward fill, train-fit
  statistical recovery, reproducible causal KNN, then one reduced-physics arm.
- Physics must beat the best surviving simpler method on identical hidden issue
  identities and must expose reconstruction uncertainty.

## Consequences for MHD

A full MHD model cannot solve structural SHARP unavailability merely by
producing a plausible solar state. It may become a candidate model-derived
feature or transient-gap reconstruction only after the simpler/reduced-physics
stages leave a measured residual problem and the required boundary conditions
are available before forecast issue time.

This narrows the physics contribution but makes it substantially more
defensible: IRIS is testing whether physics helps where reconstruction is
scientifically meaningful, rather than using simulation to manufacture an
instrument history.

## Implementation

- `src/iris_sep/missingness_audit.py` accepts an authoritative
  `structural_unavailable_mask` and reports structural and transient counts
  separately. Structural cells are excluded from the transient-reconstruction
  denominator and transient outage lengths.
- `config/missingness_recovery_contract_v1.json` forbids structural
  unavailability from being imputed as observed data.
- `tests/test_missingness_audit.py` checks that structural cells cannot also be
  marked observed and that they are excluded from reconstructable-gap counts.

## Claim boundary

This ADR does not establish the authoritative source-era mask, reconstruction
accuracy, physics advantage, SEP forecast improvement, operational readiness,
company superiority, economic savings, breakthrough status or award outcome.
Those remain experimental questions.
