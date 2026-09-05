# ADR-007 — Prefer observed source harmonization before reconstruction

Status: **Accepted as design/novelty correction; no frozen benchmark result changed**

Date: 2026-09-05

## Evidence that changes the design

The current literature does not support presenting forecast-time proton/XRS
context or generic magnetic-field simulation as an IRIS novelty claim.

- SEPNET-Ov2 (Yu et al., 2026, arXiv:2606.14440) combines magnetic predictors
  from SHARP and SMARP with flare/CME information, soft X-ray flux and historical
  >10 MeV proton flux. Its stated reason for SMARP is expanded historical
  magnetic coverage.
- Bobra et al. (2021, arXiv:2108.07918) describe SMARP, derived from SOHO/MDI,
  together with SHARP, derived from SDO/HMI, as active-region maps/keywords
  spanning from 1996 onward.
- The existing IRIS train-only audit found the most-missing SHARP aggregates
  structurally absent in early eras. That does not imply that a simulated SHARP
  measurement should be manufactured for those eras.

## Decision hierarchy

When one magnetic source is unavailable, the project will use this order:

1. **Authoritative observed measurement from the primary source**, if available
   at forecast issue time.
2. **Authoritative observed analogous measurement from a second source**, such
   as a validated SMARP/SHARP harmonized quantity, only after units, definitions,
   scaling, source latency and overlap calibration are audited.
3. **Mask-aware source-era model / fallback / abstention** when the quantity is
   structurally unavailable and no validated observed analogue exists.
4. **Causal statistical reconstruction** only for transient missingness inside a
   supported source regime.
5. **Reduced-physics reconstruction** only for those transient gaps and only if
   it beats the simpler recovery controls on held-out known observations and
   downstream SEP utility.
6. **Assimilation/full MHD** only after a measured residual gap remains and all
   forecast-time boundary conditions and uncertainty requirements are met.

A real observed alternative source is not called imputation. Its source identity
must remain explicit through preprocessing, model input, evaluation receipts and
the validity envelope.

## Novelty consequence

Do not claim novelty from any one of the following by itself:

- using proton history;
- using XRS/soft-X-ray context;
- combining SHARP and SMARP;
- predicting future magnetograms;
- physics-informed coronal magnetic-field modeling.

The intended research contribution is instead the combination of:

1. a frozen daily probability of a **NEW** >10 MeV, >=10 pfu crossing within
   24 hours, with complete episodes kept disjoint;
2. explicit source-era/availability handling and forecast-time provenance;
3. a controlled missingness experiment that separates structural unavailability
   from transient gaps and makes physics compete against simpler recovery;
4. reconstruction uncertainty that can trigger degradation or abstention rather
   than silently converting a model-derived state into an observation; and
5. paired downstream evaluation including matched-detection FAR, calibration,
   coverage and operator-facing validity.

Each element has related prior art; no combination-level novelty claim is made
until a formal literature review supports it and the experiment is complete.

## Comparator consequence

`benchmark_contract_v2.json` remains frozen and is not silently rewritten by
this ADR. SEPNET-Ov2 is nevertheless a required contemporary design reference.
If a faithful same-cohort reproduction becomes feasible before locked evaluation,
it should be documented as an additional contextual comparator or handled by an
explicit pre-test benchmark amendment rather than replacing the preregistered
comparator after outcomes are known.

## Claim boundary

This ADR is a planning correction. It establishes no model improvement,
reconstruction advantage, comparator equivalence, operational readiness,
economic benefit, company superiority, breakthrough status or award outcome.
