# IRIS-SEP authoritative current status

Status: 8 September 2026. Reviewed base: `27692b5fac31d2e0094a66e0e2f3caa09a401a2a`.

**Decision: freeze the existing V3 development candidate and finish causal end-to-end evidence. No predictive superiority or award outcome is established.**

Read [the current senior review](architecture/SENIOR_REVIEW_2026-09-08.md) for verified numbers, fixes, remaining risks and the dated delivery plan. Historical reports remain evidence of their original checkpoints.

## What exists

- Full-data cross-fitted specialist stack and V3 distilled remaining-sensor fallback.
- Exported load-only model packages; earlier independent-implementation replay evidence is recorded in `architecture/BLACK_BOX_VALIDATION_2026-09-07.md`.
- Source registry, causal aggregation, provenance and sealed-evaluation primitives with source tests. Their existence does not prove that a fresh causal forecast has run.
- Completed development-only outage/fallback experiments. Their score/monitor periods have been inspected and cannot become fresh final evidence.

## What the results support

V3 MAX_TSS score results recomputed in this review: NO_XRS TSS 0.43049 (13 detections / 603 false positives); NO_PROTON TSS 0.51300 (16 detections / 796 false positives). There are 21 positives in 3,219 score rows. Top-5% retrospective capture is unchanged from V1 (8 and 6 positives respectively). NO_PROTON removes 18 false positives; this is a small development gain.

Full-data superiority over late fusion remains inconclusive. No large operational improvement is demonstrated. The primary policy remains MAX_TSS; POD80_MIN_FAR is separately reported.

## Blocking evidence

- Released aggregate inputs have unresolved causal/native lineage and retrospective preprocessing.
- The latest fresh-source audit emitted no forecast. NRT SHARP lacks three expected frozen fields; proton outcomes contain real gaps.
- Matching a source registry and recomputing hashes do not independently prove retrieval time or pre-outcome forecast publication.
- There is no verified untouched final evaluation. The author has not supplied a training-only release or blinded test arrangement as last reported by the user.

## Evaluation correction

Use V2 outcome receipts: incomplete 24-hour outcomes remain unresolved, never quiet negatives. Both model and comparator evaluation validate receipt integrity. Forecast counts and self-reported seals alone cannot establish independence. Preserve old receipts and regenerate labels from complete raw outcome records before scoring with the corrected code.

The next milestone is one genuine acquisition-to-mature-outcome replay, followed by a preregistered, externally witnessed comparison. No new architecture search is justified before that milestone.
