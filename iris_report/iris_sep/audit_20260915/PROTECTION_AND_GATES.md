# Protection, evidence classes and remaining gates

## Boundary decision

Authority conflict: `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json` protects from **2025-09-10T00:00:00Z inclusive**, while older “starts_after” wording and replay/monitor definitions are not aligned. An issue before the boundary can still have an outcome horizon crossing it.

This audit admits a record only when issue time + 24 hours is **strictly earlier** than the protected boundary. Timestamp parsing happens before outcome/probability/event-field interpretation. Excluded labels, event IDs, predictions and counts are not emitted. Full source/archive bytes may be hashed without interpreting their outcomes. Originals remain immutable. This is a new diagnostic subset, not a replacement primary endpoint. Do not infer excluded outcomes by differencing tables.

A freeze created after a time interval does not make forecasts for that interval prospective. Prospective status requires predictions recorded before their outcomes, a causally available source contract, and an independent custodian where required. The repository's protected study requires a custodian and sufficient onset/quiet-block support; no progress toward its event floor is reported here.

## Evidence labels

- Historical frozen: recorded outputs tied to immutable artifacts; prior development exposure remains.
- Independently recomputed historical diagnostic: strict preboundary archive subset, no refit, new uncertainty draws.
- Post-hoc retrospective: controllers, new robustness checks and weighting extensions.
- Prospective: reserved for genuinely preissued forecasts with sealed outcomes and independent release. No prospective performance claim is made.

## Required human/external gates

| Gate | Who / concrete completion evidence |
|---|---|
| Resolve boundary wording and any earlier access | Independent custodian: written timeline and authorized scope, without revealing protected results to developers. |
| Confirm IRIS archival-data interpretation | SRC/organizer: written clarification of the 12-month-data wording for new analysis of historical public satellite data. |
| Confirm portal cycle/deadline | Student/guide: current registration portal receipt; public pages show mixed year/deadline language. Plan against October 3, 2026 until clarified. |
| Verify eligibility | Student/guide: grade, age, enrollment and prior IRIS Class-12 registration checked against official rules. |
| Establish ownership | Student: dated contribution log, actual code/proof exercise, source-reading notes and unaided defense; AI and mentor roles disclosed. |
| Forms and approvals | Actual signatories/SRC: use current official forms; no generated signature, backdating or retrospective invented approval. |
| Physical generalization | Expert-reviewed, predeclared preboundary alternative-catalog/flux protocol with source provenance; no outcome-driven definition selection. |
| Operational equivalence | Source provider documentation and both-yaw sensor mapping plus issuance/release latency evidence; current science coverage alone fails. |

No messages to organizers, reviewers or mentors have been sent. No branch has been merged and no fresh model has been trained. On September 20, four fully superseded remote branches were retired with exact-SHA archive tags; see REPOSITORY_CLEANUP_20260920.md. Scientific conclusions and protected boundaries are unchanged.
