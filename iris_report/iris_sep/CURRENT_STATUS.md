# IRIS-SEP authoritative current status

**Status date:** 2026-09-08  
**Work branch:** `codex/iris-sep-continuation-20260905`  
**Purpose:** this is the single authoritative human-readable status pointer. Historical reports remain preserved for audit continuity but must not be treated as current if they conflict with this page. The machine-readable source of truth is `config/eight_issue_status_2026-09-08.json`.

## Project in one sentence

Estimate the probability of a **NEW >10 MeV, >=10 pfu solar energetic-particle threshold crossing within the next 24 hours** and separately decide whether the available evidence justifies exposing that probability normally, as `DEGRADED`, or not at all (`ABSTAIN`).

The intended demonstrated user is a human analyst. Spacecraft autonomous control, financial savings, operational certification, company superiority, guaranteed award outcome, and a full-physics/MHD solar simulation are outside the demonstrated scope.

## Current scientific framing

The project is now best described as a **retrospective reliability study with a prospective fail-closed validation framework**.

The central reliability question is:

> A forecasting system can continue producing a plausible-looking probability after an input feed disappears. When is that number still trustworthy enough to expose, and when should the system refuse to guess?

The strongest concise lesson supported by the development work is:

> **A stable-looking probability does not imply a reliable warning.**

## Frozen development model

### Full-data architecture

`IRIS_CROSSFIT_EVIDENCE_STACK_V1`

- five-seed-median XGBoost specialists for solar/context, XRS, and historical-proton evidence;
- chronological expanding out-of-fold fit-era evidence construction;
- non-negative evidence stack;
- calibration separated from fitting;
- operating-threshold selection separated from fitting and calibration;
- frozen benchmark primary threshold policy: `MAX_TSS`;
- `POD80_MIN_FAR` retained only as a diagnostic policy.

### Missing-feed runtime

`IRIS_AVAILABILITY_DISTILLED_EVIDENCE_STACK_V3` remains the frozen missing-feed candidate.

| State | Available evidence | Runtime path | Permission |
|---|---|---|---|
| `FULL` | solar + XRS + proton | full frozen stack | normal only if provenance/interface gate passes |
| `NO_XRS` | solar + proton | pre-trained reduced-input fallback | `DEGRADED` |
| `NO_PROTON` | solar + XRS | pre-trained reduced-input fallback | `DEGRADED` |
| `NO_XRS_OR_PROTON` | solar only | diagnostic only | `ABSTAIN` |

No missing feed is reconstructed, fabricated, or retrained at runtime by V3.

## Package and replay status — complete

The frozen V3 package is now exported, reloadable, and independently replayed.

Verified package facts:

- 15 serialized XGBoost specialists;
- exact ZIP SHA-256: `69be4a5d79c17e452d2fe6115c6447995f22297002a6b25e2dc692c760495a75`;
- manifest SHA-256: `88c1a73f73f9c3b48c7b5b3c59cdc42dd3d80421ec7b47829f8ab4036e8790a7`;
- self-replay maximum absolute probability difference: `0.0`;
- separate black-box replay maximum absolute probability difference: approximately `2.22e-16`;
- ordered feature schema is position-bound;
- runtime retraining/recalibration/rethresholding is forbidden;
- the solar-only `ABSTAIN` state cannot emit an alert.

Original package workflow run: `34141515134`, artifact `10026120558`.

The exact raw package ZIP is also durably archived outside short-lived GitHub Actions storage in Google Drive, file ID `1u7GdhAXusiDNojJ-ARrKRCmqogR5Oi8z`, while preserving the same ZIP digest.

See `architecture/BLACK_BOX_VALIDATION_2026-09-07.md` and `config/eight_issue_status_2026-09-08.json`.

## Development evidence already inspected

These results are development evidence and cannot be relabelled as fresh final evidence.

Previously inspected full-stack context includes:

- older score TSS approximately `0.5120` under the diagnostic POD80/minimum-FAR policy;
- later 2023–2025 development monitor TSS approximately `0.2359`;
- later monitor detection/POD approximately `83.3%`;
- later monitor FAR approximately `95.5%`;
- previous late-fusion monitor TSS approximately `0.1894`;
- paired monitor advantage over late fusion remains inconclusive because the paired uncertainty interval crosses zero.

The high FAR must remain visible in any scientific presentation.

### V3 missing-feed point estimates

Development-only evidence includes:

- `NO_XRS`: TSS point estimate about `0.368087 -> 0.430492`, but the stratified unit-bootstrap interval crosses zero and Brier score is slightly worse;
- `NO_PROTON`: TSS about `0.507371 -> 0.512999`, with `16/21` detections retained and false positives reduced from `814` to `796`.

These are promising/inconclusive point estimates, not established superiority.

## Missingness/outage conclusion

Three experiment classes remain explicitly separate:

1. random observed-cell deletion;
2. daily model-input modality outage over 1/3/7 daily cycles;
3. true upstream sensor outage at native cadence.

The third cannot be reproduced faithfully by deleting rows from the daily aggregate table.

The random-missingness work showed that causal forward-fill could preserve probabilities comparatively well while decision quality still degraded. At 40% random observed-cell loss, frozen diagnostic-policy TSS fell by roughly `0.227`.

Therefore probability similarity is not evidence of safe warning behavior.

## Alert-filter experiments — closed, rejected

Two post-V3 attempts to obtain a large false-alarm reduction were tested on an already-inspected development score block.

### First decision filter

Run `34206898291` was rejected because the meta-model could create alerts outside the original frozen V3 alert set. It was therefore not a pure suppression layer.

### Monotone veto

Run `34216432375` fixed that structural problem: the layer could only suppress an existing V3 alert.

Development score results:

- `NO_XRS`: `13 TP / 603 FP -> 12 TP / 431 FP`, a `28.5%` FP reduction but one lost detection;
- `NO_PROTON`: `16 TP / 796 FP -> 13 TP / 594 FP`, a `25.4%` FP reduction but three lost detections.

The veto was rejected because the false-positive improvement came with detection loss.

**Final decision:** keep plain V3 and stop further post-hoc alert-filter tuning on the inspected score block. Further tuning would increase overfitting risk.

## Historical aggregate provenance limitation

The retrospective aggregate table remains useful for development, but its preprocessing includes interpolation, overlap-era mapping/backcasting, nearest-time SHARP/SMARP matching, and fitted SHARP<-SMARP transformations.

Therefore a finite historical aggregate cell is not automatically evidence that the value was natively available at forecast issue time.

This means retrospective skill cannot automatically be promoted to prospective causal skill.

## Prospective source preflight — implemented and executed

The all-source preflight now attempts and receipts seven registered source families:

1. NOAA/SWPC primary proton;
2. NOAA/SWPC primary XRS;
3. NOAA/SWPC GOES instrument-source routing;
4. JSOC HMI SHARP CEA-NRT;
5. LMSAL HEK GOES flares;
6. NASA CCMC DONKI CME;
7. NASA GSFC CDAW CME.

Final hardened run: `34217607075`  
Artifact: `10052465094`  
Artifact ZIP SHA-256: `95ae852fb005674fcac555bc7c741bd1becab04ac2829823819ea6c0575ed5e3`

### Acquisition result

Six of seven source families authenticated successfully.

CDAW remained unavailable because the external provider reset the HTTPS connection after five bounded retries. No silent catalogue substitution was made.

Authentication receipt SHA-256: `b6fd5156a91dfb54c13d1255e059ab82f3e245f5c84751580605150429a6932b`.

## Current primary scientific blocker — exact frozen live interface

Even if CDAW were available, the frozen V3 skill forecast remains scientifically inadmissible prospectively because the exact 259-position feature interface cannot currently be reproduced from `hmi.sharp_cea_720s_nrt`.

The frozen vector contains:

- 251 solar/context positions;
- 4 XRS positions;
- 4 proton positions.

CEA-NRT does not expose three required frozen SHARP keywords:

- `CMASKL`;
- `MEANGBL`;
- `USFLUXL`.

Those missing quantities affect 18 frozen feature-vector positions.

Frozen disposition:

`FROZEN_MODEL_RETROSPECTIVE_ONLY_FOR_SKILL_UNTIL_SOURCE_EQUIVALENCE_IS_ESTABLISHED`

Forbidden workarounds include zero-fill, similarly named substitution, retrospective backcast, silent definitive-to-NRT replacement, or calling a reduced interface the same frozen model.

The prospective preflight therefore emitted **no forecast probability**.

That no-forecast outcome is the correct fail-closed behavior; it is not a skill result.

See `architecture/prospective_causal_interface_disposition_2026-09-08.json`.

## Prospective evaluation framework — implemented, awaiting an admissible live case

### Forecast seal and external witness

Implemented:

- `IRIS_SEP_SEALED_FORECAST_V2`;
- semantic probability/horizon validation;
- maximum seal delay of 300 seconds;
- GitHub-server witness for pre-outcome digest existence.

A witness proves only that the digest existed before the outcome, not that the source data or forecast was scientifically valid.

### Fair comparator

Implemented:

- `IRIS_SEP_SEALED_COMPARISON_V1`;
- no caller-supplied duplicate IRIS probability;
- one forecast / one comparison / one label linkage;
- no post-outcome comparator selection;
- frozen fit-role-prevalence climatology comparator.

No live comparison result exists because no admissible prospective V3 skill forecast has been emitted.

### Correct 24-hour outcome evaluator

Implemented:

- `IRIS_SEP_SEALED_NEW_CROSSING_LABELS_V2`;
- exact issue+24h endpoint requirement;
- maximum 300-second observation gap;
- finite, non-negative flux requirement;
- duplicate timestamps rejected;
- immature/gappy windows -> `UNRESOLVED`;
- already-active issue time -> `INELIGIBLE` for a new-crossing label.

No live V3 outcome has been labelled because there is no scientifically admissible prospective V3 forecast to label.

## Eight-item delivery state

The current machine contract is `config/eight_issue_status_2026-09-08.json`.

| # | Item | Status |
|---|---|---|
| 1 | immutable V3 package | `PASS` |
| 2 | fresh trusted source acquisition | `CLOSED_WITH_EXTERNAL_BLOCKER` — CDAW unavailable after bounded retries |
| 3 | exact causal 259-feature interface | `CLOSED_WITH_SCIENTIFIC_BLOCKER` — 3 missing CEA-NRT fields / 18 affected positions |
| 4 | load-only inference and provenance | `PASS` |
| 5 | forecast seal and external witness | engineering complete; no admissible live forecast to witness |
| 6 | fair comparator | engineering complete; no admissible live case |
| 7 | correct 24h outcome evaluator | engineering complete; no admissible live forecast to label |
| 8 | evidence and paper-generation support | `PASS_INTERNAL_EVIDENCE_PACKAGE` |

No item is being hidden behind an `IN_PROGRESS` label.

## Evidence-writing package

Created and bound into item 8:

- `architecture/IRIS_EVIDENCE_DOSSIER_2026-09-08.md` — receipt-backed internal scientific evidence source;
- `architecture/STUDENT_PAPER_WRITING_TEMPLATE_2026-09-08.md` — structured template for the student authors to write the final fair material themselves.

The evidence dossier includes positive, negative, inconclusive, and blocked results. The template intentionally does not provide submission-ready prose.

## What is established

- deterministic V3 packaging and load-only replay;
- exact ordered-schema binding;
- explicit full/degraded/abstain runtime semantics;
- missing-feed fallbacks that do not fabricate a runtime feed;
- solar-only `ABSTAIN` cannot emit an alert;
- development missingness/outage evidence showing that probability similarity does not guarantee decision safety;
- trusted prospective acquisition receipt machinery;
- six-of-seven final source-family authentication in the hardened preflight;
- exact prospective feature-interface blocker identified;
- fail-closed refusal to emit a scientifically inadmissible prospective skill forecast;
- prospective forecast sealing, witnessing, comparator, and 24h outcome machinery.

## What is not established

- independent prospective V3 forecast skill;
- superiority over a same-date operational forecast;
- a causally equivalent live source for the 18 unavailable frozen feature positions;
- full authentication of all seven registered source families in the final preflight;
- low false-alarm operation;
- operational certification;
- economic impact;
- full-physics simulation;
- award outcome.

## Next scientific move

Do **not** tune V3 further on the inspected retrospective score/monitor evidence.

The next legitimate skill experiment should use a new feature interface designed from quantities demonstrably available at forecast time, freeze that interface before evaluating outcomes, and reserve a genuinely untouched cohort for final assessment. It should be treated as a new prospective-compatible experiment rather than a silent modification of the frozen 259-feature model.

Until then, V3 remains a retrospective reliability candidate with a validated fail-closed prospective boundary.

## Claim boundary

Passing software tests establishes software consistency, not forecast usefulness. Development performance establishes development evidence only. The current prospective preflight establishes source/interface behavior and a correct refusal to forecast, not independent skill. No superiority, operational-readiness, economic-impact, full-physics-simulation, breakthrough, or competition-outcome claim is permitted without corresponding independent evidence.
