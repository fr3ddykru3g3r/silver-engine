# IRIS-SEP two-week finish and real-world pilot plan — 2026-09-11

## Purpose

Run two parallel tracks without reopening the retrospective model-development loop:

1. **Competition completion:** freeze the scientific result and complete student-owned paper, poster, video, forms and oral defense within 14 days.
2. **Real-world impact:** package the work as an **SEP Forecast Validity and Evaluation Kit**, secure independent custody and prepare approved prospective predictor collection.

The scientific centerpiece remains the evaluation question: **does a 24-hour SEP score measure advance warning, repeated representation of one physical event, or recognition of a storm that is already active?**

No further retrospective threshold/model tuning is permitted without a new preregistration.

## Frozen historical baseline

Submission baseline:

- commit: `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`
- source-only CI run: `34573237824`
- fixed replay issues: 7,558
- matched inferential cohort: 85 onset episodes + 1,080 quiet blocks
- shared bootstrap draws: 10,000
- joint XGBoost mapped -> episode-normalized -> onset TSS: `0.726 -> 0.621 -> 0.437`
- joint XGBoost onset sensitivity: 48.24%
- joint XGBoost onset false-alarm ratio: 88.95%
- joint-minus-proton-free onset TSS contrast: approximately `-0.041`, interval crosses zero

The historical model remains a research comparator, not a deployable forecasting product.

## Track A — competition completion

### Days 1–3 — freeze the submission record

Required outputs:

- immutable submission evidence bundle index;
- paper draft and synopsis pinned to the historical baseline;
- two headline figures pinned to the frozen replay;
- workflow receipts, artifact IDs/hashes and source-only CI receipt;
- negative-result ledger including the failed freshness experiment;
- protected-data statement;
- denominator audit proving the distinction between **228 full-table onset windows** and **85 matched onset episodes**;
- explicit statement that any later scientific analysis requires a new preregistration.

Acceptance gate:

- every displayed metric is traceable to a frozen receipt;
- no protected post-2025 outcome access;
- no language implying untouched, prospective, operationally validated or state-of-the-art historical evidence.

### Days 4–7 — student-owned final writing

Students must rewrite/check in their own words:

- abstract;
- paper/research-plan prose;
- poster text;
- 85–90 second spoken script;
- acknowledgements and support disclosures.

Each student must pass a mastery checklist covering:

- operational target definition (>10 MeV, >=10 pfu);
- mapped occurrence, episode-normalized occurrence and causal new onset;
- Episode Multiplicity Factor;
- TSS;
- false-alarm ratio versus false-positive rate;
- positive and negative bootstrap units;
- prior exposure of the historical data;
- causal feature-availability limitation;
- missing-data boundary;
- protected-cohort custody.

Citation gate:

- every retained citation must be checked against the original paper/publisher record;
- no reference remains solely because generated text cited it.

Forms gate:

- use truthful project dates, roles, AI/programming support, mentor involvement and prospective-extension status;
- do not imply a prospective phase started before approval.

### Days 8–11 — produce the presentation

Headline visuals:

1. graphical abstract showing repeated positive windows and already-active persistence;
2. fixed-alert comparison titled **“Measured SEP forecast skill changes when repeated episodes and persistence are removed.”**

Required production checks:

- print-ready poster PDF;
- print proof at intended dimensions;
- text legible at judging distance;
- 85–90 second video recorded with natural delivery;
- central claim visible without overclaiming the forecasting model.

### Days 12–14 — defense and submission gate

Run at least three skeptical mock judging sessions.

Required defense topics:

- denominator differences;
- row dependence and physical-unit bootstrap;
- FAR versus FPR;
- missing-data policy;
- historical prior exposure;
- novelty boundary;
- protected prospective cohort;
- why model ranking differences do not prove proton features are harmful;
- why the comparator is not operationally ready.

Final submission is permitted only after:

- forms pass;
- citation audit passes;
- student-authorship/ownership review passes;
- print proof passes;
- video timing passes;
- evidence archive passes;
- every headline claim is defensible from a receipt.

## Track B — company-facing reliability pilot

### Product definition

Working product name: **SEP Forecast Validity and Evaluation Kit**.

It is not marketed as a new operational forecaster. Its job is to make forecast inputs, validity state and evaluation semantics auditable for operators.

Every forecast record should expose at minimum:

```json
{
  "issue_time": "UTC timestamp",
  "alert": true,
  "validity_state": "VALID | DEGRADED | ABSTAIN",
  "reason_codes": [],
  "input_age_seconds": {},
  "source_hashes": {},
  "model_hash": "sha256",
  "threshold_hash": "sha256",
  "ledger_previous_hash": "sha256"
}
```

Required reason-code families:

- `STALE_INPUT`
- `ABSENT_REQUIRED_FEED`
- `AMBIGUOUS_PROTON_CHANNEL`
- `STRUCTURAL_UNAVAILABILITY`
- `EXCESSIVE_TRANSIENT_LOSS`
- `FAILED_CAUSAL_AVAILABILITY_RECEIPT`
- `LEDGER_INTEGRITY_FAILURE`
- `PROVENANCE_UNCERTAIN`

### Missing-data policy

- causal forward-fill is allowed only for short, explicitly transient gaps inside a frozen age limit;
- future observations are forbidden;
- structural absence is never forward-filled;
- modest transient recovery inside the validated envelope receives `DEGRADED`;
- stale, severe, ambiguous, structurally unavailable or provenance-uncertain inputs receive `ABSTAIN`;
- physics-based magnetic-map reconstruction remains experimental until a preregistered hidden real-map comparison shows improvement over persistence and preserves downstream onset-forecast performance.

Current evidence boundary:

- random transient loss in the roughly 5–20% range has shown reasonable probability preservation in development tests;
- at 40% loss, operational-policy skill degrades materially;
- this does **not** establish real magnetic-map reconstruction validity.

### Prospective execution

Before collection:

- obtain applicable fair/SRC approval for the extension;
- appoint an independent adult custodian;
- complete the execution manifest with source, rule, model, threshold, schema and code hashes;
- freeze the 00:00 UTC issue clock and pre-issue collection rules.

During collection:

- raw public responses retained;
- retrieval timestamps retained;
- source hashes retained;
- validity decisions recorded before outcome windows;
- predictions written before target windows;
- append-only ledger chaining enforced;
- protected labels, episode identities and running event counts withheld from the development side.

Confirmatory gate:

- >=50 distinct onset episodes;
- >=500 quiet blocks;
- otherwise disposition is `INSUFFICIENT_CONFIRMATORY_INFORMATION`.

No threshold rescue, model rescue or selective time-period reporting is allowed after protected outcomes are opened.

### Stakeholder recruitment

Prepare for review, but do not send without student/mentor approval:

- two-page operator brief;
- one-page technical validation sheet;
- five-minute demonstration;
- 30-minute workflow-interview request;
- narrowly framed pilot proposal.

Target stakeholder classes:

- satellite operators;
- space-weather service providers;
- aviation-radiation teams;
- university space-weather groups;
- national forecasting organizations.

Workflow-interview questions should ask:

- what operational decision changes after an SEP alert;
- acceptable false-alarm burden;
- required warning time;
- behavior under feed loss/staleness;
- auditability requirements;
- whether `VALID / DEGRADED / ABSTAIN` states are useful;
- which reason codes need to be machine-readable.

First measurable impact milestone:

**one documented external workflow review and one concrete requirement from that review incorporated into the frozen prospective protocol.**

No company-benefit or deployment-impact claim is permitted before such evidence exists.

## Cross-track verification and acceptance criteria

- baseline commit and replay artifacts remain byte-identifiable;
- historical metrics reproduce;
- protected post-2025 outcome access remains forbidden;
- paper, poster, video, synopsis and PR use identical denominators and claim boundaries;
- prospective records are created before their target windows;
- append-only ledger passes independent verification;
- missing-data tests cover future-value rejection, staleness, structural absence, ambiguity, tampering, forward-fill limits, degradation and abstention;
- physics reconstruction receives no promotion without a preregistered real-map persistence comparison;
- submission passes ownership, citation, forms, print, timing and oral-defense gates;
- impact claims require recorded stakeholder feedback or valid prospective evidence.

## Assumptions

- submission deadline is within two weeks;
- no confirmed company partner currently exists;
- an independent adult custodian can be recruited through the mentor, school or research network;
- prospective collection begins only after applicable approval and execution freeze;
- the 50-onset information floor will not be reached inside the submission sprint;
- the prospective system is therefore presented at submission as a frozen validation path, not completed confirmatory evidence.
