# IRIS-SEP authoritative current status

**Status date:** 2026-09-06  
**Work branch:** `codex/iris-sep-continuation-20260905`  
**Purpose:** this is the single authoritative current-status pointer. Historical reports remain preserved for audit continuity but must not be treated as current if they conflict with this page.

## Fixed scientific question

Estimate the probability of a **NEW >10 MeV, >=10 pfu SEP threshold crossing within the next 24 hours**, excluding issue times at which the threshold is already active, and separately decide whether the available inputs justify exposing that probability as `VALID`, `DEGRADED`, or `ABSTAIN`.

The intended demonstrated user is a human analyst deciding whether a forecast merits attention. Spacecraft safe-mode control, financial savings, operational certification, company superiority, award outcome and a physics-simulation claim remain outside the demonstrated scope.

## Current development candidate

`IRIS_CROSSFIT_EVIDENCE_STACK_V1` remains the development candidate. It combines five-seed-median XGBoost specialists for solar, XRS and historical-proton evidence using four expanding chronological out-of-fold fit-era folds, a nonnegative evidence stack, one calibration-role logit intercept and thresholds selected only on the threshold role.

Architecture expansion on the current daily aggregate table is stopped. The rejected residual-anchor and two-state temporal variants remain preserved in the changelog.

Source of truth: `config/current_development_architecture_v1.json`.

## Verified development evidence already inspected

These are development results, not fresh final evidence.

- Cross-fitted stack, frozen POD80/minimum-FAR diagnostic policy:
  - older score TSS: approximately `0.5120`;
  - later 2023–2025 development monitor TSS: approximately `0.2359`;
  - later monitor detection rate: approximately `83.3%`;
  - later monitor false-alarm ratio: approximately `95.5%`.
- Previous late-fusion monitor TSS: approximately `0.1894`.
- The paired monitor advantage of the cross-fitted stack over late fusion is inconclusive because the 95% interval crosses zero.
- Random observed-cell missingness on the promoted stack showed that causal forward-fill preserved probability better than the other simple arms, but at 40% random loss the frozen POD80/minimum-FAR TSS dropped by about `0.227`; probability similarity therefore does not establish safe decision behavior.

Primary historical receipts and decisions are indexed in `CHANGELOG.md`.

## Evaluation policy reconciliation

The frozen benchmark primary operating-threshold policy remains **maximum TSS on the threshold role with the lowest-threshold tie break**, as specified by `config/evaluation_policy_v1.json`.

`POD80_MIN_FAR` is a separately reported matched-detection / reliability-oriented diagnostic policy. Development papers and plots that emphasize POD80 must not relabel it as the frozen benchmark primary policy. Historical POD80 results remain preserved and clearly labelled.

See `architecture/THRESHOLD_POLICY_RECONCILIATION_2026-09-06.md`.

## Current blocking scientific limitation: input provenance and causality

The hash-pinned SEP-Prediction-V2 aggregate table cannot currently be described as a table of purely native forecast-time observations.

The pinned upstream preprocessing includes, among other operations:

- linear interpolation of HAPI and OMNI proton flux;
- an OMNI→HAPI linear mapping fitted over the overlap era and used to backcast the pre-HAPI era and fill HAPI gaps;
- nearest-time SHARP/SMARP feature imputation, which is not constrained to past-only observations;
- SHARP←SMARP regression models fitted over their overlap and used to backcast the pre-SHARP era.

Therefore a finite aggregate cell is **not** sufficient evidence that the value was natively observed or causally available at the forecast issue time. Until row/cell provenance can be reconstructed from the source pipeline, aggregate-table provenance is `UNKNOWN` where a native/reconstructed distinction cannot be demonstrated.

Source of truth: `provenance/SOURCE_PROVENANCE_AUDIT_2026-09-06.md` and `config/source_provenance_contract_v1.json`.

## Missing-data experiments

Three experiment classes must remain separate:

1. **Random observed-cell deletion:** already run on the promoted stack; a development stress test only.
2. **Daily model-input modality outage:** 24/72/168 hours correspond to 1/3/7 consecutive daily issue cycles in the model-ready table. This experiment is being completed under a preregistered contract.
3. **True upstream sensor outage:** requires the high-cadence source stream and causal reaggregation. It cannot be simulated faithfully by deleting daily aggregate rows.

No experiment may call `np.isfinite(value)` equivalent to `native observation`. Finite cells with unresolved provenance are `UNKNOWN` and cannot be used as hidden native truth for a provenance-sensitive reconstruction claim.

## Inspected versus untouched evidence

The score and 2023–2025 monitor blocks used in development have already been inspected and cannot be relabelled as fresh final evidence. Public-development artifacts and comparator outputs used during model development are also not fresh final evidence.

The locked test remains forbidden during development. A future cohort may be called untouched only after a custodian-produced overlap attestation confirms that its identities/outcomes were not exposed during development. The attestation must not reveal protected identities to the development side.

Source of truth: `config/inspected_evidence_registry_v1.json`.

## Reproducibility and delivery status

Implemented:

- target derivation and chronological role construction;
- cross-fitted specialist evidence stack;
- calibration/threshold separation;
- missingness/recovery primitives;
- validity/admission envelope and immutable inference-bundle primitives;
- development result receipts.

Not yet complete:

- a reloadable exported package containing all 15 specialist models, feature order, stack parameters, calibration and thresholds;
- inference that loads that exact promoted package without training;
- end-to-end binding of promoted-stack model files to admission/provenance;
- an independently untouched final superiority evaluation;
- a causal high-cadence XRS/proton pipeline.

## Immediate execution order

1. make timestamps resolution-independent and test both nanosecond and microsecond representations;
2. expand source-only CI to discover all source-only tests on push and pull request;
3. complete source-family provenance/causality audit and replace finite=observed assumptions with explicit provenance where the experiment requires native truth;
4. preregister and execute the daily 1/3/7-cycle outage experiment without moving blocks after outcome inspection;
5. freeze and test always-expose / abstain / degraded policies using development-only information;
6. align external comparators before making any superiority statement;
7. export and reload the promoted model package;
8. generate submission tables, paper and 90-second video from verified receipts.

## Claim boundary

Passing software tests establishes software consistency, not forecast usefulness. Development performance establishes only development evidence. No superiority, operational-readiness, economic-impact, full-physics-simulation or breakthrough claim is permitted unless the corresponding independent evidence exists.