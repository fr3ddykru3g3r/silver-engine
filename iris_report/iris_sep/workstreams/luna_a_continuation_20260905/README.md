# Luna A continuation: safe-data intake audit (2026-09-05)

This workstream defines the evidence required before IRIS-SEP can ingest a
training-only SEP-PRISM/CLEAR release or participate in a blinded comparison.
It is an intake gate, not a benchmark result. No row-level SEPVAL, CLEAR, or
SEPNET testing data was opened, downloaded, locally filtered, or copied here.

## Contact status

The user reports that the training-only SEP-PRISM / blinded-evaluation request
was sent to the publisher contact already recorded in the request document.
The send timestamp and message identifier are unknown, and no reply or
publisher artifact has been supplied. This status is recorded without
re-sending the request and without storing the email address in this
workstream. No external message was sent during this audit.

## Intake checklist

The following checks must be satisfied by a publisher-side release or
controlled-enclave export. A response that only exposes the existing mixed
archive does not satisfy the gate.

| Gate | Required evidence | Current status | Exact artifact to accept | Blocker if absent |
|---|---|---|---|---|
| Provenance and version | Immutable release/commit, SHA-256, byte size, row count, ordered schema, units, and source lineage for every modality | **BLOCKED** | `source_manifest.json` plus the immutable training-only table/export; hash must be computed over the delivered bytes | Cannot establish what was trained or reproduce the cohort |
| Cohort separation | Publisher-side proof that every training/development row is disjoint from locked evaluation identities and outcomes; include role counts and identity policy without exposing locked identities | **BLOCKED** | `cohort_manifest.json` with train/development roles, episode grouping, purge rule, and separation receipt; opaque evaluator is acceptable | Local filtering of a mixed table would cross the lock boundary |
| Label definition | Exact operational target: new `>10 MeV, >=10 pfu` crossing in the next 24 hours; threshold sampling rule, event grouping, target-window anchor, and treatment of interrupted/gapped intervals | **BLOCKED** | `label_semantics.md` or machine-readable `label_manifest.json`, tied to the source hash | Cannot know whether labels represent new crossings or persistence/already-active conditions |
| New-crossing and already-enhanced handling | Audited exclusion logic for windows already above threshold or already-enhanced at issue time; examples may be synthetic or aggregate, never locked identities | **BLOCKED** | `already_above_threshold_receipt.json` containing rule version, counts by role, and checksum of the applied rule/export | Forecast target can be contaminated by ongoing events |
| Issue-time causality | Last usable timestamp for each feature; no post-issue values; exact window anchor and 24-hour horizon; duplicate/ambiguous issue IDs rejected | **BLOCKED** | `feature_latency_manifest.json` with modality, field/group, source timestamp, publication timestamp, lag rule, and missingness policy | Offline features could leak future information |
| Publication latency | Measured or publisher-declared availability delay for magnetic, flare/X-ray, CME, and proton context streams, including processing time convention | **BLOCKED** | `latency_manifest.json` with units, quantiles or fixed bounds, measurement method, and version | Warning lead time and real availability cannot be audited |
| CLEAR lineage | Explicit CLEAR release/version and mapping from the training export to the selected event catalogue; resolve v1/v2 ambiguity | **BLOCKED** | `clear_lineage_receipt.json` with release identifier, source hashes, mapping version, and attribution | Same threshold name may refer to different event definitions |
| Licensing and attribution | Redistribution/reuse terms for the delivered table, derived metrics, prediction receipts, and each upstream source; exact attribution language | **PARTIAL** | `reuse_terms.json` signed or publisher-authored, plus attribution text | Cannot publish rows, derived files, or comparator claims safely |
| Blinded evaluation | Frozen episode-disjoint cohort or opaque evaluator; one final submission after model/configuration freeze; no test labels or predictions during tuning | **BLOCKED** | `blinded_evaluation_protocol.json` and, preferably, evaluator endpoint/export receipt | Paired superiority claim would be selection-biased |
| Published comparator fidelity | SEPNET-PRISM/O configuration, preprocessing, loss, calibration, threshold and issue-time convention for the same cohort | **BLOCKED** | `comparator_configuration.json` plus versioned source/configuration hashes | Local adapter remains development-only and is not a reproduction |

## Exact artifacts already available

These are metadata/protocol records only and do not clear the intake gate:

- `../luna_a/provenance_manifest.json` — public-source manifest; records the
  lock boundary, source versions, reported table sizes, and unresolved rights
  and lineage items.
- `../luna_a/zenodo_21297635_record.json` — SEP-PRISM Zenodo v1 metadata;
  the approximately 4.9 GB archive was not downloaded.
- `../luna_a/CLEAR_Benchmark_Dataset_V2_0_Documentation.pdf` — CLEAR v2.0
  documentation; core event-list rows remain withheld.
- `../luna_a/SEPVAL2023_RulesofParticipation_v4.pdf` — public SEPVAL protocol;
  its published cardinality differs from the Zenodo/SEPNET prose and must be
  resolved from an authorized frozen file later.
- `../luna_a/RECEIPT.md` — acquisition and rights summary, including hashes.
- `../../receipts/v2_clear_training_access_status_2026-09-05.json` — current
  blocked-access receipt and required publisher-generated export fields.
- `../../config/benchmark_contract_v2.json` — frozen target, causality,
  partition, receipt, and winning-gate requirements.
- `../../config/evaluation_policy_v1.json` — frozen calibration, threshold,
  matched-detection, bootstrap, and non-degradation policy.
- `../../provenance/SEP_PRISM_TRAINING_ONLY_REQUEST.md` — request content and
  requested publisher-side artifacts; it is not evidence of a reply.
- `../../receipts/sepnet_reproduction_status_2026-09-05.json` — confirms that
  published SEPNET-O has not been reproduced and lists the remaining fidelity
  discrepancies.

## Concrete blockers and safe next actions

1. **No verified training-only release.** Wait for an immutable publisher-side
   export or controlled-enclave export. Do not download the 4.9 GB archive, the
   nested 33 MB table, the 803 MB hourly table, or any mixed training/test
   table for local filtering.
2. **New-crossing semantics are not evidenced.** Require the label and
   already-enhanced receipts above before preparing a final cohort.
3. **Feature availability is unquantified.** Require per-modality publication
   latency and last-input timestamps before making a daily operator warning
   or lead-time claim.
4. **CLEAR lineage and licensing are incomplete.** Resolve the version mapping
   and obtain explicit reuse terms before packaging rows or derived benchmark
   files.
5. **Blinded evaluation is unavailable.** Require a frozen episode-disjoint
   evaluator or publisher-side same-cohort scoring arrangement. Until then,
   local V1/V5/V6 results remain development-only and must not be called
   SEPVAL evidence, a final new-crossing score, superiority, or a breakthrough.

When an acceptable release arrives, intake must stop on the first failed
check, hash every accepted artifact, bind the hashes into the benchmark
contract/partition receipts, and notify the primary agent for integration.
The first downstream run should be a schema/causality/role audit; model
training and any locked evaluation remain separate, receipt-driven steps.

## Prohibited actions for this workstream

- Accessing locked identities, outcomes, predictions, or metrics.
- Downloading a mixed table merely to filter it locally.
- Inferring event identity from dates, numeric similarity, or nearest
  timestamps.
- Sending another publisher request or inventing a send timestamp/message ID.
- Treating existing legacy-label development runs as final benchmark evidence.
