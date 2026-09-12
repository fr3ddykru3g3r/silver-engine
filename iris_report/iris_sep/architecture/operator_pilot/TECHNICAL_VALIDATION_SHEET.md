# SEP Forecast Validity and Evaluation Kit — technical validation sheet

**DRAFT — DO NOT SEND WITHOUT REVIEW**

## Scope

This sheet describes a proposed forecast-validity/evaluation layer. It does **not** certify the historical classifier or claim operational forecast superiority.

## Historical evaluation evidence

- SEP-PRISM model-free table: 14,464 windows, 650 stored positives.
- Unique physical mapping: 614 positive windows -> 257 physical episodes; EMF 2.389.
- Full-table onset-state audit: 418 persistence windows, 228 onset windows.
- Fixed replay: 7,558 score issues.
- Matched inference: 85 distinct onset episodes + 1,080 quiet blocks.
- Shared physical-unit bootstrap: 10,000 draws.
- Joint-XGBoost fixed-alert TSS: `0.726 mapped -> 0.621 episode-normalized -> 0.437 new onset`.
- Onset sensitivity: 48.24%.
- Onset false-alarm ratio: 88.95%; false-positive rate: 4.51%.
- Joint-minus-proton-free onset TSS contrast: `-0.040 [-0.163,+0.083]`.

Interpretation: the evaluation definition materially changes measured skill for the same fixed forecasts. It does not establish a deployable model.

## Validity-state contract

Machine-readable contract: `config/sep_forecast_validity_contract_v1.json`.

`VALID`: required pre-issue input/provenance gates pass and no recovery is declared.

`DEGRADED`: candidate alert may be exposed, but a short causal recovery was applied inside a separately frozen source-specific recovery rule.

`ABSTAIN`: alert is set to null when any fatal validity condition is present.

Fatal conditions include:

- stale required input;
- absent required feed;
- ambiguous >=10 MeV integral proton semantics;
- structural unavailability;
- excessive transient loss;
- failed causal-availability receipt;
- future observation;
- uncertain provenance;
- ledger-integrity failure.

## Provenance record

Each record carries issue time, input ages, source hashes, model/rule hash, threshold hash and previous-ledger hash. The implementation produces a deterministic validity-record hash and explicitly records `protected_outcomes_accessed=false`.

## Missing-data evidence boundary

Development-only random observed-cell masking tested 5%, 20% and 40% loss. Causal forward-fill produced the smallest probability drift across those stress levels. At 5–20%, paired Brier-delta intervals included zero; at 40%, Brier degradation became detectably positive. Under the `POD80_MIN_FAR` policy, 40% causal-forward-fill loss produced TSS delta `-0.2265` with paired 95% interval `[-0.4240,-0.0487]`.

This was not a real-feed outage test and does not define an operational 20%/40% switch. A pilot must freeze source-specific freshness/recovery rules independently.

## Magnetic reconstruction boundary

Physics-based magnetic-map reconstruction has not passed a hidden real-map comparison. It remains experimental. Promotion requires preregistration, a hidden real-map benchmark against persistence, and preserved downstream causal-onset utility.

## Prospective confirmation boundary

Protected post-2025 outcomes remain unavailable to the development side. Confirmation requires independent custody and at least 50 distinct onset episodes plus 500 quiet blocks. Failure to meet the floor returns `INSUFFICIENT_CONFIRMATORY_INFORMATION`; thresholds/models may not be rescued on the same cohort.

## Reproducibility receipts

Historical build baseline: `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`.

Baseline source-only CI: `34573237824` — PASS.

Fixed-model replay artifact: `10137507101`; SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`.
