# SEP Forecast Validity and Evaluation Kit — operator brief

**DRAFT — DO NOT SEND WITHOUT STUDENT/MENTOR REVIEW**

## The problem

A forecast can be numerically correct-looking while still be hard to interpret operationally. Two issues are especially important for solar energetic particle (SEP) alerts:

1. a daily 24-hour score can count the same long physical radiation storm multiple times or reward recognition of a storm already active at issue time;
2. a forecast can be exposed even when its required inputs are stale, structurally unavailable, ambiguously identified or causally reconstructed beyond a validated envelope.

IRIS-SEP separates these questions from the forecasting model itself.

## What the kit does

The proposed pilot is an **audit and validity layer**, not a replacement operational forecaster.

For each pre-issue forecast record it exposes:

- issue time;
- candidate alert;
- `VALID`, `DEGRADED` or `ABSTAIN` state;
- machine-readable reason codes;
- input ages;
- source hashes/provenance;
- model/rule and threshold hashes;
- append-only ledger linkage.

The evaluation side then reports forecast performance in three complementary ways:

- mapped 24-hour occurrence;
- physical-episode-normalized occurrence;
- genuine causal new onset.

This makes it possible to ask whether a score represents repeated daily decisions, performance across distinct physical storms, or advance warning before a new storm begins.

## What the historical research found

The strongest historical replay uses 7,558 chronological score issues and a matched inferential population of 85 onset episodes plus 1,080 quiet blocks. With the **same fixed joint-XGBoost alerts**, TSS changes from `0.726` under mapped occurrence to `0.621` after episode normalization and `0.437` for causal new onset.

This is methodological evidence, not an operational-performance claim. Under new-onset scoring, the same research comparator has 48.24% sensitivity and an 88.95% false-alarm ratio. It is therefore **not proposed as a deployable forecasting product**.

A proton-free XGBoost comparator has a higher onset point estimate, but the frozen joint-minus-proton-free TSS contrast is `-0.040 [-0.163,+0.083]`; the evidence does not establish reliable model superiority.

## Input validity policy

The current pilot interface fails closed:

- short causal recovery inside a separately frozen source-specific envelope -> `DEGRADED`;
- stale input, missing required feed, ambiguous proton channel, structural absence, excessive transient loss, failed causal-availability receipt, future observation, uncertain provenance or ledger-integrity failure -> `ABSTAIN`.

Development-only random-missingness tests found causal forward-fill preserved probability space relatively well at modest 5–20% random cell loss, while a 40% stress condition materially degraded one operational-policy TSS. Those percentages are **not** proposed as universal deployment thresholds; a real pilot must freeze source-specific age/recovery rules before execution.

## Pilot question

The first external goal is deliberately modest:

> Does the validity/evaluation record expose information that an operator or forecaster would need to trust, reject or audit an SEP alert?

We would first request a 30-minute workflow interview rather than claim operational benefit. The pilot is successful only if a reviewer identifies a concrete workflow requirement and that requirement is incorporated into a versioned prospective protocol.

## What we need from a reviewer

We want to understand:

- what decision an SEP alert changes;
- minimum useful warning time;
- acceptable false-alarm/review burden;
- required behavior during delayed, stale or absent feeds;
- which data/provenance fields must be retained for audit;
- when an explicit `ABSTAIN` is preferable to a best-effort forecast.

## Evidence boundary

The historical data were development-exposed. Protected post-2025 outcomes remain sealed from the development side. Genuine prospective predictions, if approved, must be recorded before their target windows and evaluated later by an independent custodian under a frozen protocol.
