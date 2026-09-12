# Published SEPNET target and physical-episode audit result — 2026-09-09

## Disposition

`EXTERNAL_PUBLIC_BENCHMARK_AUDIT_COMPLETE — DEVELOPMENT/METHODOLOGY EVIDENCE`

This result audits the public rolling-window construction and event table in the SEP-Prediction repository. It does **not** reproduce the paper's final SEPVAL score and does not establish that the published model performance is wrong.

Pinned upstream source:

- repository: `yuyian/SEP-Prediction`
- commit: `d0eb54e46b7dd6c760325e123d2ad86f9420fbff`
- rolling table Git blob: `82cf23dc789e70d8185dd6c69bb6b3bd7692f77b`
- event table Git blob: `dc1ea7a18c2b06034a144f656d76f4dcd5e85674`
- construction script Git blob: `8234c4e8e7814b1d0cc477263e82504b1901127a`

Immutable audit workflow:

- workflow run: `34362893938`
- audit commit: `8b7ccf1fa2bc4aae20a3b12e0838270b12f8379b`
- artifact ID: `10108593291`
- artifact digest: `sha256:498104f66efb59ebd85795d628d799006f60266a8e3454bd2df3b127b988673d`

## What was recomputed

The audit uses the public `df_SEP.csv` operational >10 MeV, >=10 pfu start/end fields as physical threshold-episode intervals and classifies each public rolling-table forecast window independently of any model prediction.

For each window it records:

- the public stored `future_Operational_SEP_label`;
- whether an operational >=10 pfu interval actually overlaps the nominal subsequent 24-hour forecast interval;
- whether the issue time is already inside an active operational episode;
- whether a new operational episode starts in the future window;
- unique physical-episode mapping where possible;
- episode-normalized occurrence weight.

No model was fitted or scored in this arm.

## Main result

The public rolling table contains **11,773** fixed 24-hour windows.

Stored public operational target:

- positive: **1,726**
- negative: **10,047**

Independent operational-threshold interval reconstruction:

- windows with an actual >=10 pfu episode overlapping the future interval: **643**
- windows without such an overlap: **11,130**

Target reconciliation:

| Public stored label | No >=10 pfu overlap | >=10 pfu overlap |
|---|---:|---:|
| 0 | 10,047 | 0 |
| 1 | **1,083** | 643 |

Thus **1,083 of 1,726 stored operational-positive windows** do not overlap an operational >=10 pfu episode in the nominal forecast interval under the event-table start/end semantics used in the audit. Across the entire table this is 1,083 / 11,773 = **9.20%** of all rolling rows.

This is a reproducible inconsistency between the public rolling operational target and the simple interpretation “an operational >=10 pfu episode occurs in the subsequent 24 hours.” It is not, by itself, evidence that the final published SEPVAL evaluation uses the same mismatch.

## Onset versus persistence decomposition

Using the same physical intervals, the 11,773 windows decompose into:

- `ELIGIBLE_ONSET_NEGATIVE`: **11,130**
- `ALREADY_ACTIVE_PERSISTENCE`: **411**
- `ELIGIBLE_ONSET_POSITIVE`: **227**
- `DUPLICATE_OR_AMBIGUOUS`: **5**

Therefore most actual threshold-overlap positive windows in this public rolling table are not new-onset opportunities: **411 are already-active persistence windows versus 227 new-onset windows**.

This supports the scientific need to report persistence recognition separately from new-event onset forecasting.

## Physical-episode multiplicity

The public event table contains **265** complete operational >=10 pfu episodes under the audited fields.

For uniquely mapped positive windows:

- positive windows: **610**
- physical episodes represented: **256**
- Episode Multiplicity Factor: **610 / 256 = 2.3828125**
- episodes represented by more than one positive window: **175**
- episodes represented by at least three positive windows: **93**
- maximum positive windows assigned to one episode: **7**
- recorded positive rows mapping ambiguously to multiple episodes: **33**

The episode-normalized positive occurrence mass for the uniquely mapped subset is therefore **256**, compared with 610 unit-weight positive windows in that same mapped subset.

This demonstrates that repeated representation of long physical events is not merely hypothetical in a current public SEP rolling benchmark.

## Why this matters scientifically

Three separate questions must not be collapsed:

1. **Target construction:** does a stored positive label correspond to the operational threshold event claimed by the target definition?
2. **Multiplicity:** when a physical event legitimately appears in multiple forecast windows, should every window carry full positive statistical weight?
3. **Causal task:** is the model predicting a new event, or recognizing persistence of an event already underway?

`IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1` separates these mechanisms rather than treating all positive windows as interchangeable.

## Claim boundary

Authorized statements:

- the pinned public rolling table contains repeated representations of physical operational SEP episodes;
- the pinned public rolling target has a reproducible mismatch with a direct operational >=10 pfu interval-overlap interpretation for 1,083 rows;
- already-active persistence accounts for a large fraction of reconstructed positive threshold-overlap windows;
- episode-normalized and onset-specific evaluation are therefore scientifically motivated.

Not authorized from this audit:

- “SEPNET's published final score is wrong”;
- “all SEP forecasting papers overstate skill”;
- “episode normalization necessarily lowers skill”;
- “the best published model changes under our benchmark”;
- operational-superiority or award claims.

Those require the separate fixed-model benchmark and, for strong external claims, reproduction of the exact published final evaluation cohort.
