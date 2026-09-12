# SEP forecast validity/evaluation workflow pilot

**DRAFT — DO NOT SEND OR START WITHOUT REVIEW, APPROVAL AND EXECUTION FREEZE**

## Pilot objective

Determine whether an SEP forecast-validity/evaluation record captures the information a real reviewer needs to decide whether a forecast is interpretable, auditable and acceptable to expose.

This pilot does **not** promise improved SEP forecast accuracy, operational savings or deployment readiness.

## Phase 1 — workflow review

Conduct one structured 30-minute interview with an appropriate external reviewer. No sensitive operational data are required. Record only the requirements the reviewer agrees may be retained.

Required topics:

- operational decision affected by an SEP alert;
- useful warning-time range;
- acceptable alert/review burden;
- stale/missing-feed policy;
- provenance and audit requirements;
- preferred behavior when the system cannot verify a required input.

**Phase-1 acceptance criterion:** at least one concrete requirement is documented and incorporated into a newly versioned protocol. Do not rewrite the already-inspected historical claim to match that feedback.

## Phase 2 — approved predictor-side shadow record

Only after applicable fair/SRC approval, independent-custodian assignment and execution freeze:

- capture predictor/source evidence at the frozen pre-issue clock;
- retain raw public responses and retrieval timestamps;
- compute source hashes and input ages;
- create the candidate prediction/rule output before the target window;
- assign `VALID`, `DEGRADED` or `ABSTAIN` under the frozen validity contract;
- append the record to the independently verifiable hash chain.

No development-side protected outcomes are accessed in this phase.

## Phase 3 — independent aggregate evaluation

The independent custodian later applies the frozen outcome/evaluation contract. The development side receives only permitted aggregate results and receipts, not protected row-level labels or event identities.

The confirmatory information floor remains at least 50 distinct onset episodes and 500 quiet blocks. If it is not met, report `INSUFFICIENT_CONFIRMATORY_INFORMATION`.

## Pilot artifacts

Each predictor record should expose:

- issue time;
- candidate alert or null on abstention;
- validity state and reason codes;
- input ages;
- source hashes;
- model/rule and threshold hashes;
- previous-ledger hash and current record hash.

## Missingness rule

Do not use a universal percentage threshold from the development stress test. Short causal recovery is allowed only within a separately frozen source-specific freshness/recovery rule and must be labelled `DEGRADED`. Structural absence, staleness, excessive transient loss, ambiguous semantics, uncertain provenance or causal-receipt failure produces `ABSTAIN`.

## Physics reconstruction

Magnetic-map reconstruction is excluded from the initial pilot. It can enter a future protocol only after a separately preregistered hidden real-map benchmark beats persistence and preserves downstream causal-onset utility.

## Stop conditions

Stop or abstain if:

- approval/consent boundary is unclear;
- a required source cannot be verified;
- the record would be created after its target window starts;
- ledger integrity fails;
- development-side protected outcome access would be required;
- a requested change would amount to post-hoc rescue of the same prospective cohort.

## Impact claim boundary

A completed workflow interview is evidence of a reviewed requirement, not evidence of operational benefit. A valid prospective result is evidence only for the frozen evaluated claims. No company-benefit or deployment claim is permitted without corresponding external or prospective evidence.
