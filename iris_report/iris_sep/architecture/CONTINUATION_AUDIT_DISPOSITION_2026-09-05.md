# Primary integration of continuation audits

The user reports the publisher training-only request was sent. No reply,
message ID, or sent timestamp is available. No resend or other message occurred.

## Engineering disposition

The new `src/iris_sep/pilot_replay.py` boundary verifies receipt bytes against a
trusted expected digest and binds model, schema, calibration, thresholds,
freshness policy and critical modalities. It recomputes observation age and
rejects publication after issue time. Required uncertainty fields are finite.
Missing/mutated evidence, missing inputs, stale feeds, schema mismatch and
future publication produce ABSTAIN with no probability or advisory state.
The existing low-level builder is retained; its syntactic hash check alone is
not sufficient for pilot admission. The new boundary accepts synthetic receipts
only; it is an offline replay tool, not a deployed forecast endpoint.

The 11 receipt-driven replays and SVG are explicitly synthetic. VALID means
fixture contract acceptance, never scientific validation. No event/quiet-period
outcome claims, benchmark plots, or final-results report were generated.

## Red-team findings still gating independent evaluation

Luna E's report is retained unchanged. E1/E3/E5/E6 require an integrated sealed
cohort/prediction boundary with coverage, role, purge, manifest and bootstrap
membership audits. Array-based metric utilities do not themselves enforce
access control. Do not connect them to a locked data source during development.
E2 requires authoritative episode and data-quality semantics; the existing
threshold-run groups remain development heuristics. A physical episode can
contain several issue labels, so unit-label homogeneity must not be imposed
blindly as a scientific definition. E4/E8/E9 require source revision, ordered
feature-schema and latency manifests on real intake. Replay timestamp checks
are partial engineering mitigation, not source authenticity or latency proof.
E7 is mitigated for fixtures by digest/binding and uncertainty checks; an
approved immutable evidence registry is still needed for real pilot admission.

## Research disposition

The Luna B matrix is hypothesis guidance, not a cross-paper leaderboard.
Proton context, then XRS, then geometry are conditional ablations; their value
for a NEW crossing can differ from value for an already-enhanced legacy target.
No optimization was run. The known V1 table lacks proton/XRS history; inventing
those inputs or retuning the used monitor would not test these hypotheses.
Only role=train may supply future chronological episode-disjoint inner folds.
Exact data/episode availability, comparator reproduction, fold support and
configuration hashes must be frozen before executing the proposed batch.
Retain negative runs and select the simplest supported candidate. Published
random-split scores are not comparable with the proposed frozen cohort.

## Scope and storage

ADR-005 reconciles the historical HMI-only checklist with the existing primary
contract. No frozen evaluation rule was changed. All historical source and
failed experiments are preserved. Old versions referenced by pinned receipts
cannot be deleted without breaking current reproducibility. Large artifacts
remain outside ordinary Git; deleting their only copy is not part of source
synchronization. Remote root `iris-model/` is not overwritten by the newer local
source snapshot, which differs in 14 common files. Separate namespaces preserve
both versions and existing history.
