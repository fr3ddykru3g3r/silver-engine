# IRIS-SEP authoritative current status

**Status date:** 2026-09-08  
**Decision update:** 2026-09-09  
**Work branch:** `codex/iris-sep-continuation-20260905`  
**Purpose:** single authoritative human-readable status pointer. Historical reports remain preserved for audit continuity. Machine-readable historical delivery state remains `config/eight_issue_status_2026-09-08.json`; exposure control is now extended by `config/inspected_evidence_registry_v2.json`.

## Project in one sentence

Estimate the probability of a **NEW >10 MeV, >=10 pfu solar energetic-particle threshold crossing within the next 24 hours** and separately decide whether the evidence justifies exposing that probability normally, as `DEGRADED`, or not at all (`ABSTAIN`).

The intended demonstrated user is a human analyst. Spacecraft autonomous control, financial savings, operational certification, company superiority, guaranteed award outcome, and a full-physics/MHD solar simulation are outside the demonstrated scope.

## Current scientific disposition

IRIS-SEP is a **RETROSPECTIVE reliability research program with a prospective fail-closed validation framework**. It does not currently contain a demonstrated first-place-standard forecasting discovery.

Two conclusions now control all further work:

1. the frozen V3 model remains retrospective-only for forecast skill because its exact live causal interface is unavailable;
2. the separate freshness-crossover study is now closed as a negative result and is not a route for further post-result rescue tuning.

## Frozen development model and live-interface blocker

The frozen missing-feed candidate remains `IRIS_AVAILABILITY_DISTILLED_EVIDENCE_STACK_V3`.

Runtime semantics remain:

| State | Evidence | Permission |
|---|---|---|
| `FULL` | solar + XRS + proton | normal only if provenance/interface gate passes |
| `NO_XRS` | solar + proton | `DEGRADED` |
| `NO_PROTON` | solar + XRS | `DEGRADED` |
| `NO_XRS_OR_PROTON` | solar only | `ABSTAIN` |

The V3 package contains 15 serialized XGBoost specialists and has deterministic package/replay evidence. Package integrity does **not** establish independent prospective forecasting skill.

The exact 259-position V3 interface still cannot be reproduced from `hmi.sharp_cea_720s_nrt` because three required frozen SHARP quantities are unavailable:

- `CMASKL`;
- `MEANGBL`;
- `USFLUXL`.

They affect **18 frozen feature-vector positions**.

Frozen disposition remains:

`FROZEN_MODEL_RETROSPECTIVE_ONLY_FOR_SKILL_UNTIL_SOURCE_EQUIVALENCE_IS_ESTABLISHED`

Zero-fill, similarly named replacements, retrospective backcasts, silent definitive-to-NRT substitution, or calling a reduced feature interface the same V3 model remain forbidden.

The prospective preflight therefore emitted **no forecast probability**. That is correct fail-closed behavior, not a skill result.

## Development performance remains development-only

Previously inspected evidence includes:

- older diagnostic-policy score TSS about `0.5120`;
- 2023–2025 development-monitor TSS about `0.2359`;
- development-monitor POD about `83.3%`;
- development-monitor FAR about `95.5%`;
- late-fusion monitor TSS about `0.1894`, with paired advantage inconclusive;
- `NO_XRS` V3 point estimate about `0.368087 -> 0.430492`, with uncertainty crossing zero and slightly worse Brier;
- `NO_PROTON` V3 about `0.507371 -> 0.512999`, retaining `16/21` detections and reducing false positives `814 -> 796`.

The high false-alarm burden must remain visible.

The rejected decision filter and monotone veto remain rejected. The monotone veto reduced false positives by about 25–29% but lost detections. Plain V3 remains frozen; no more post-hoc alert-filter tuning is allowed on exposed score evidence.

## Freshness crossover V1 — CLOSED NEGATIVE RESULT

Execution identity:

- study: `IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1`;
- executed commit: `ed3aba1def18bb022382efa4afaaf251c76e6cff`;
- run: `34341335588`;
- artifact: `10100069342`;
- artifact SHA-256: `ba907ffae86301c22b1afb3a41a0bf98c825b562b90b5319ce347a4af1db7868`.

The study used a documented two-satellite retrospective definition: GOES-13 NASA/SPDF OMNI five-minute `Av >10 MeV` proton measurements and GOES-15 NOAA/NCEI operational XRS. Historical SWPC-primary-stream equivalence was not established.

Role support was sparse:

| Role | Issues | Positives |
|---|---:|---:|
| Fit 2011–2014 | 968 | 30 |
| Calibration 2015 | 225 | 2 |
| Threshold selection 2016 | 250 | 1 |
| Retrospective score 2017 | 211 | 3 |

2017 clean controls were:

| Model | TP | FP | FN | TN | TSS | FAR |
|---|---:|---:|---:|---:|---:|---:|
| Joint | 3 | 152 | 0 | 56 | 0.269231 | 98.06% |
| XRS-only | 2 | 73 | 1 | 135 | 0.315705 | 97.33% |
| Proton-only | 3 | 102 | 0 | 106 | 0.509615 | 97.14% |

The proposed freshness-induced model-order crossover was **not demonstrated**. Proton-only already exceeded the joint model at zero XRS delay; XRS-only also had a higher point-estimate TSS than joint at zero proton delay. No evaluated delay produced the required sign reversal.

No switching policy passed the practical gate. Fixed TTL selection collapsed to zero minutes and produced no false-alert improvement over the reduced reference. The XRS state-dependent policy increased pooled replay false-alert executions from 816 to 988 (`+21.08%`).

The result is closed under `architecture/FRESHNESS_CROSSOVER_V1_CLOSURE_2026-09-09.md`.

### Freshness evidence limitations

The closure record preserves all major limitations:

- only three 2017 positive opportunities;
- proton-only 3/3 detection accompanied by 102 false alerts;
- per-issue predictions and fitted models were not packaged, so ranking metrics and paired uncertainty cannot be independently regenerated from aggregate counts alone;
- the executed bootstrap paired methods within each delay but redrew underlying units across delays, so it does not provide simultaneous curve inference;
- simulated proton-delay eligibility still used contemporaneous proton information, making the experiment a conditional retrospective population rather than an implementable availability-aware replay;
- increasing delay changed both information age and retained window coverage;
- repeated delay variants do not create new independent solar events.

The strongest permitted claim is that this frozen experiment **failed to demonstrate** the proposed freshness crossover or a beneficial switching intervention. It does not establish that XRS is harmful, that fewer sensors are generally better, or that freshness never matters.

## Novelty boundary

Broad novelty claims are closed. Daily GOES proton/X-ray features, XGBoost/SVM SEP forecasting, proton-dominant predictors, calibrated multimodal forecasting, onset forecasting, and 24-hour multi-source SEP prediction all have close precedent.

The distinctive contribution of freshness V1 is the controlled-delay protocol plus its negative result—not a field-leading architecture or successful warning policy.

## Exposure state and future data protection

`config/inspected_evidence_registry_v2.json` is now the active exposure guard.

Important consequences:

- all eligible historical development rows before the fixed monitor were already used for development;
- the 2023-07-31 through 2025-09-10 monitor is already inspected development evidence;
- freshness V1 exposed its 2011–2017 roles and outcomes;
- no convenient 2018–2022 slice may be relabelled as untouched merely because a new question is proposed;
- observations after 2025-09-10 are reserved as a **protected candidate**, with identities, labels, event counts, and scores not authorized for development-side inspection.

A genuinely untouched claim requires an independent custodian to define/hash the cohort and attest non-exposure against the complete registry before outcomes are released.

## Onset-versus-continuation candidate — NOT AUTHORIZED TO RUN YET

A separate question has been considered: whether headline SEP forecasting skill can be decomposed into warning of a **new onset** versus recognition of **continuation/persistence** of an event already in progress.

This distinction is operationally meaningful but not itself novel; NOAA/SWPC already distinguishes expected proton-event onset from expected persistence.

Feasibility is recorded in `architecture/ONSET_CONTINUATION_FEASIBILITY_2026-09-09.md`.

The current decision is:

`DO_NOT_RUN_AS_AN_INDEPENDENT_CLAIM_STUDY_YET`

Reasons:

1. historical periods are development-exposed;
2. the only plausible post-monitor period must remain protected until a contract and custodian-controlled cohort exist;
3. exposed-data planning indicates sparse positive support relative to the precision needed for a strong detection-harm claim;
4. the novelty claim must be narrower than the onset/persistence distinction itself;
5. a new experiment must persist issue-level predictions/models, an exclusion ledger, and shared resampling objects from the start.

Do not inspect post-2025 outcomes merely to learn whether the proposed study has enough positives.

## Immediate research priority

The immediate priority is **evidence completion and eligibility**, not another model-development cycle:

1. preserve the immutable negative freshness result and its audit;
2. recover any genuinely original per-issue prediction/model evidence only if it already exists; otherwise label any reconstruction explicitly and never tune on 2017;
3. maintain the complete exposure ledger;
4. obtain a written IRIS ruling on eligibility of newly executed computational research using older archival observations;
5. complete authentic sponsor/student/parent and AI-assistance documentation as required;
6. have the student authors independently understand, interpret, and write the competition material under the applicable AI rules;
7. reserve future/prospective data rather than consuming it in another underpowered exploratory score run.

If no admissible independent cohort can meet a preregistered precision requirement in time, stop rather than manufacture a positive discovery claim.

## What is established

- deterministic V3 package/replay integrity;
- explicit `FULL`/`DEGRADED`/`ABSTAIN` semantics;
- fail-closed refusal to expose a skill forecast when the exact V3 live interface is invalid;
- negative missingness/outage and alert-filter evidence retained rather than hidden;
- a completed controlled-delay freshness experiment with verified aggregate arithmetic;
- failure of the freshness-crossover and switching-policy hypotheses on that experiment;
- a stronger exposure guard protecting candidate future evidence.

## What is not established

- independent prospective forecasting skill;
- low false-alarm operation;
- operational or company superiority;
- causal benefit of stale-to-fresh switching;
- a novel onset-versus-continuation discovery;
- sufficient independent support for a new prospective claim;
- full-physics simulation, economic impact, certification, or award outcome.

## Claim boundary

Passing software tests establishes software consistency, not forecasting usefulness. Development results remain development evidence. Negative results remain part of the record. Protected future data must remain protected. No superiority, breakthrough, operational-readiness, economic-impact, full-physics-simulation, or competition-outcome claim is permitted without corresponding independent evidence.
