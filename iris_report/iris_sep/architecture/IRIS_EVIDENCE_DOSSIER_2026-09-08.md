# IRIS-SEP evidence dossier — 2026-09-08

**Purpose:** internal, receipt-backed source of truth for the student team. This is **not** a submission paper, abstract, poster, or award claim. Final competition prose must be written by the student authors in their own words and must disclose assistance as required by the applicable fair rules.

**Branch:** `codex/iris-sep-continuation-20260905`  
**Project state:** open, unmerged development continuation  
**Authoritative machine status:** `config/eight_issue_status_2026-09-08.json`

---

## 1. The project in one sentence

Can a daily solar energetic-particle forecasting system estimate the probability of a **NEW >10 MeV, >=10 pfu threshold crossing within 24 hours** while explicitly degrading or abstaining when the available measurements do not justify a normal forecast?

### Plain-language version for a non-specialist judge

A solar-radiation warning system can still output a plausible-looking number after one of its input feeds disappears. This project asks a different question from ordinary forecast ranking: **when is that number still trustworthy enough to show, and when should the system refuse to guess?**

### What the project is not

- not a full magnetohydrodynamic or solar-flare simulation;
- not a claim of operational certification;
- not a claim of economic savings;
- not a claim of superiority to an operational agency forecast;
- not a claim that the system will win IRIS or ISEF;
- not a claim of independent prospective skill yet.

---

## 2. Fixed target and decision semantics

**Forecast target:** probability that a **new** >10 MeV proton event crosses **10 pfu within the next 24 hours**.

Issue times at which the threshold is already active are excluded from the new-crossing target.

The probability and the permission to expose it are separate outputs:

- `VALID` / normal path: evidence sufficient for the full-data forecast path;
- `DEGRADED`: a pre-trained reduced-input fallback is permitted, but it is explicitly not presented as normal/full evidence;
- `ABSTAIN`: no alert is emitted even if an internal diagnostic probability exists.

The intended demonstrated user is a **human analyst**, not an autonomous spacecraft controller.

---

## 3. Frozen model family

### 3.1 Full-data development architecture

`IRIS_CROSSFIT_EVIDENCE_STACK_V1`

Core design:

- XGBoost specialist families for solar-context, X-ray, and historical-proton evidence;
- five random-seed specialists per family, aggregated by seed median;
- chronological expanding out-of-fold evidence construction in the fit era;
- non-negative evidence stack;
- calibration-role intercept separated from fitting;
- threshold-role operating threshold separated from fitting and calibration;
- frozen benchmark primary threshold policy: `MAX_TSS`;
- `POD80_MIN_FAR` retained only as a diagnostic/matched-detection policy.

### 3.2 Missing-feed runtime

`IRIS_AVAILABILITY_DISTILLED_EVIDENCE_STACK_V3`

Runtime states:

| Availability state | Inputs used | Permission |
|---|---|---|
| `FULL` | solar + XRS + proton | normal/full path subject to provenance gate |
| `NO_XRS` | solar + proton distilled fallback | `DEGRADED` |
| `NO_PROTON` | solar + XRS distilled fallback | `DEGRADED` |
| `NO_XRS_OR_PROTON` | solar-only diagnostic | `ABSTAIN` |

No missing runtime feed is reconstructed or fabricated by V3.

---

## 4. Reproducibility evidence

### 4.1 Immutable package

The V3 package contains **15 serialized XGBoost specialists** plus feature order, stack parameters, calibration, thresholds, dependency information, and SHA-256 bindings.

Authoritative package evidence from `config/eight_issue_status_2026-09-08.json`:

- build workflow run: `34141515134`;
- original GitHub artifact: `10026120558`;
- exact package ZIP SHA-256: `69be4a5d79c17e452d2fe6115c6447995f22297002a6b25e2dc692c760495a75`;
- manifest SHA-256: `88c1a73f73f9c3b48c7b5b3c59cdc42dd3d80421ec7b47829f8ab4036e8790a7`;
- specialist count: `15`;
- self-replay maximum absolute probability difference: `0.0`;
- durable raw-package archive: Google Drive file ID `1u7GdhAXusiDNojJ-ARrKRCmqogR5Oi8z`.

The durable archive removes dependence on the original short-lived Actions artifact while preserving the exact ZIP digest.

### 4.2 Independent black-box replay

`architecture/BLACK_BOX_VALIDATION_2026-09-07.md` records a separate implementation pass that did not import the IRIS runtime and used a different XGBoost version. It:

- verified all 15 model hashes;
- verified ordered state schemas;
- reproduced probabilities to maximum absolute difference approximately `2.22e-16`;
- verified that the solar-only `ABSTAIN` state cannot emit alerts even when its numerical probability crosses a threshold.

**Interpretation:** this establishes package integrity and deterministic replay, not prospective skill.

---

## 5. Development performance evidence already inspected

These results are **development evidence only** and must never be relabelled as untouched final evidence.

### 5.1 Full stack context

Previously inspected development evidence includes:

- older score TSS approximately `0.5120` under the diagnostic POD80/minimum-FAR policy;
- later 2023–2025 monitor TSS approximately `0.2359`;
- later monitor detection/POD approximately `83.3%`;
- later monitor false-alarm ratio approximately `95.5%`;
- previous late-fusion monitor TSS approximately `0.1894`;
- paired monitor advantage over late fusion is inconclusive because the paired uncertainty interval crosses zero.

The **95.5% FAR must be shown, not hidden**. It is a major limitation and one reason the project focuses on trust/abstention rather than claiming a production-ready warning system.

### 5.2 V3 missing-feed point estimates

From the current PR evidence summary:

- `NO_XRS`: development TSS point estimate improves from about `0.368087` to `0.430492`, but the stratified unit-bootstrap interval crosses zero and Brier score is slightly worse;
- `NO_PROTON`: TSS changes from about `0.507371` to `0.512999`; detections remain `16/21` while false positives fall from `814` to `796`.

These are **promising point estimates**, not established superiority.

### 5.3 Probability similarity is not decision safety

`architecture/PROMOTED_STACK_MISSINGNESS_TRANSFER_RESULT_2026-09-06.md` and the event-bearing outage work show that preserving a similar-looking probability under missingness does not guarantee safe classification behavior.

A key observed stress-test result was that causal forward-fill often preserved probability better than simpler alternatives under random deletion, yet at 40% random observed-cell loss the frozen POD80/minimum-FAR TSS fell by roughly `0.227`.

This supports the central reliability idea:

> **A stable-looking probability does not imply a reliable warning.**

---

## 6. Event-bearing outage and fallback evidence

The project separates three different concepts that must not be mixed:

1. random observed-cell deletion;
2. daily model-input modality outage over 1/3/7 daily cycles;
3. true upstream sensor outage at native cadence.

The third requires a causal high-cadence source pipeline and is not equivalent to deleting daily aggregate rows.

The availability-conditioned/distilled fallback was designed to avoid runtime reconstruction:

- a missing XRS feed selects a pre-trained solar+proton fallback;
- a missing proton feed selects a pre-trained solar+XRS fallback;
- loss of both feeds yields a solar-only diagnostic but `ABSTAIN` permission;
- no runtime retraining occurs.

The strongest defensible conclusion is architectural rather than promotional: **the runtime can represent reduced evidence explicitly and can refuse to turn an under-supported numerical probability into an alert.**

---

## 7. Rejected false-alarm filters — negative evidence that must remain visible

Two post-V3 alert-filter experiments were explored on an already-inspected development score block. Neither is promoted.

### 7.1 Rejected decision filter

Run: `34206898291`

Failure mechanism: the meta-model could create alerts outside the original frozen V3 alert set. That violated the intended semantics of a false-alarm suppression layer.

Observed examples included:

- `NO_XRS`: V3 `13 TP / 603 FP` became `15 TP / 674 FP`;
- `NO_PROTON`: V3 `16 TP / 796 FP` became `16 TP / 698 FP`.

Although one state gained detections, the layer was not monotone and could create new false alerts. It was rejected structurally rather than tuned around the score cohort.

### 7.2 Rejected monotone veto

Run: `34216432375`

Structural correction: the filter could **only suppress an alert already emitted by frozen V3** and could never create a new alert.

Development score result:

| State | Frozen V3 | Veto | FP change | TP change |
|---|---:|---:|---:|---:|
| `NO_XRS` | 13 TP / 603 FP | 12 TP / 431 FP | −172 (−28.5%) | −1 |
| `NO_PROTON` | 16 TP / 796 FP | 13 TP / 594 FP | −202 (−25.4%) | −3 |

The veto achieved large false-positive reductions but lost detections. It therefore failed the desired no-detection-loss promotion gate.

### 7.3 Final alert-filter decision

**Keep plain V3. Stop post-hoc alert-filter tuning on the inspected score set.**

Reason: additional threshold/filter searches after observing score outcomes would increase overfitting risk. The rejected runs are retained as negative evidence.

---

## 8. Prospective causal-source preflight

Workflow: `IRIS-SEP prospective source preflight`  
Final hardened run: `34217607075`  
Artifact ID: `10052465094`  
Artifact ZIP SHA-256: `95ae852fb005674fcac555bc7c741bd1becab04ac2829823819ea6c0575ed5e3`

The preflight attempts all seven registered source families, creates acquisition receipts, authenticates them against the trusted-source registry, checks the frozen causal-interface disposition, and **never emits a forecast merely because acquisition partly succeeded**.

### 8.1 Source acquisition result

Six of seven registered source families authenticated successfully:

1. `NOAA_SWPC_PRIMARY_PROTON_7D`;
2. `NOAA_SWPC_PRIMARY_XRS_7D`;
3. `NOAA_SWPC_GOES_INSTRUMENT_SOURCES`;
4. `JSOC_HMI_SHARP_NRT`;
5. `LMSAL_HEK_GOES_FLARES`;
6. `NASA_CCMC_DONKI_CME`.

`NASA_GSFC_CDAW_CME` failed because the provider connection was reset after bounded retries. No replacement catalogue was silently substituted.

Authentication receipt SHA-256: `b6fd5156a91dfb54c13d1255e059ab82f3e245f5c84751580605150429a6932b`.

### 8.2 Stronger blocker: the exact frozen feature interface is not available from CEA-NRT

The frozen package expects 259 ordered feature positions:

- 251 solar/context positions;
- 4 XRS positions;
- 4 proton positions.

JSOC `hmi.sharp_cea_720s_nrt` does not expose three required SHARP keywords used by the frozen interface:

- `CMASKL`;
- `MEANGBL`;
- `USFLUXL`.

Those missing quantities affect **18 frozen vector positions**.

The frozen scientific disposition is therefore:

`FROZEN_MODEL_RETROSPECTIVE_ONLY_FOR_SKILL_UNTIL_SOURCE_EQUIVALENCE_IS_ESTABLISHED`

Forbidden workarounds include zero-fill, similarly named replacement, retrospective backcast, silent definitive-to-NRT substitution, or calling a reduced feature interface the same frozen model.

### 8.3 Correct prospective behavior

The preflight emitted **no forecast probability**.

That is a successful fail-closed result. Six authenticated source families do not override an invalid exact model interface.

---

## 9. Why the historical aggregate table cannot prove live causality

The development aggregate table remains useful for retrospective modeling, but its preprocessing includes operations such as interpolation, overlap-era mapping/backcasting, nearest-time SHARP/SMARP matching, and fitted SHARP←SMARP transformations.

Therefore:

- a finite aggregate cell is not automatically a native forecast-time observation;
- retrospective development skill is not automatically prospective causal skill;
- the provenance gate must remain separate from the prediction model.

This is a central limitation, not a footnote.

---

## 10. Prospective evidence framework already implemented

Even though item 3 blocks a valid live V3 skill forecast, the supporting prospective framework exists.

### Forecast seal / witness

- seal format: `IRIS_SEP_SEALED_FORECAST_V2`;
- maximum seal delay: 300 seconds;
- semantic probability/horizon checks are enforced;
- a GitHub-server witness can establish pre-outcome digest existence;
- witness existence is **not** treated as proof of input authenticity or forecast skill.

### Comparator framework

- comparison format: `IRIS_SEP_SEALED_COMPARISON_V1`;
- caller cannot provide an independent duplicate of the IRIS probability;
- one forecast / one comparison / one label linkage is enforced;
- post-outcome comparator selection is forbidden;
- built-in frozen comparator: fit-role prevalence climatology.

### 24-hour outcome evaluator

- label format: `IRIS_SEP_SEALED_NEW_CROSSING_LABELS_V2`;
- maximum data gap: 300 seconds;
- exact issue+24h endpoint required;
- finite, non-negative flux required;
- duplicate timestamps rejected;
- immature or inadequately observed windows remain `UNRESOLVED`;
- already-active issue times are `INELIGIBLE` for a new-crossing label.

These are engineering validation components. A valid live case is still required before they can generate independent prospective skill evidence.

---

## 11. The strongest defensible scientific story today

### Problem

Forecast systems are usually judged by how accurate their probabilities are when inputs are present. Real monitoring systems also face missing or degraded measurements.

### Research gap addressed by this project

A model may continue outputting a plausible probability when evidence disappears. The project tests and engineers a stricter principle: **forecast probability and permission to trust/expose that probability should be separate decisions.**

### Contribution supported by current evidence

1. A frozen multi-source SEP development model and deterministic package were built.
2. Missing-feed states are explicit rather than hidden behind runtime imputation.
3. Reduced-input models are pre-trained; no runtime retraining or fabricated sensor values are needed.
4. The system can `DEGRADED`-label reduced-evidence forecasts and `ABSTAIN` when evidence becomes too weak.
5. Prospective source acquisition is cryptographically receipted and fail-closed.
6. When the exact frozen interface could not be reproduced from CEA-NRT, the system **refused to issue a skill forecast** rather than silently substituting unavailable quantities.

### Result that should not be claimed

The project has **not** yet established that V3 is independently superior in an untouched prospective cohort.

---

## 12. Judge-facing figures to build from receipts

These are figure specifications, not fabricated results.

### Figure A — one-picture system concept

Flow:

`SUN / SPACE-WEATHER MEASUREMENTS -> INPUT AVAILABILITY CHECK -> FULL V3 OR REDUCED FALLBACK -> VALID / DEGRADED / ABSTAIN -> HUMAN ANALYST`

Show `ABSTAIN -> no alert` explicitly.

### Figure B — why stable probability is not enough

Use the existing missingness/outage receipts to contrast probability similarity against the change in TSS/decision behavior. Caption must state that these are development stress tests.

### Figure C — missing-feed state machine

Four states: `FULL`, `NO_XRS`, `NO_PROTON`, `NO_XRS_OR_PROTON`, with the exact allowed inputs and permissions.

### Figure D — negative alert-filter experiment

Two rows (`NO_XRS`, `NO_PROTON`) showing frozen V3 TP/FP versus monotone-veto TP/FP. The lesson is the trade-off: ~25–29% FP reduction was obtainable only with detection loss on this development cohort.

### Figure E — prospective preflight fail-closed result

Seven source boxes: six green/authenticated and CDAW external failure. Then an independent interface gate showing missing `CMASKL`, `MEANGBL`, `USFLUXL` -> 18 unavailable frozen positions -> **NO SKILL FORECAST EMITTED**.

This is likely the clearest figure for scientific integrity.

---

## 13. Claim ledger

| Claim | Current status | Safe wording |
|---|---|---|
| Exact V3 package can be replayed | supported | deterministic load-only replay verified |
| Missing feeds can be handled without runtime reconstruction | supported for V3 design | state-specific pre-trained fallbacks are used |
| Solar-only state can be forced to abstain | supported | numerical threshold crossing cannot create an alert in `ABSTAIN` |
| V3 is better than previous fallbacks | development point estimates only | promising/inconclusive development evidence |
| V3 beats operational forecasts | unsupported | do not claim |
| Prospective V3 skill established | unsupported | do not claim |
| All seven live source families authenticated | false | six of seven authenticated in final preflight |
| Frozen 259-feature interface is available live | false | three required CEA-NRT keywords are unavailable; 18 positions affected |
| System appropriately refused an invalid live forecast | supported | preflight emitted no probability when exact interface was inadmissible |
| ~25–29% FP reduction possible with veto | observed development result | achieved only with TP loss; veto rejected |
| Full-physics solar simulation | false | do not claim |
| Award/winning outcome | unsupported | do not claim |

---

## 14. Questions the student team must be able to answer without prompts

1. What exactly is an SEP and how is it different from a solar flare?
2. What does `>10 MeV, >=10 pfu` mean physically and operationally?
3. Why is the target a **new** crossing rather than simply “proton flux above threshold”?
4. Why is the forecast horizon 24 hours?
5. What are TSS, POD, FAR, Brier score, and calibration error?
6. Why can a high POD coexist with an unusably high FAR?
7. Why are fit, calibration, threshold, score, monitor, and final evaluation roles separated?
8. Why does random row/cell deletion differ from a real sensor outage?
9. Why is forward-fill not automatically causal proof?
10. Why is `ABSTAIN` different from predicting “no event”?
11. Why was the first alert filter structurally invalid?
12. Why was the monotone veto rejected even though it cut false positives strongly?
13. What do `CMASKL`, `MEANGBL`, and `USFLUXL` represent in the frozen feature interface, and why is silently replacing them unacceptable?
14. Why does six-of-seven successful source acquisition still not authorize a forecast?
15. What does the black-box replay prove, and what does it not prove?
16. Why is the historical aggregate table not sufficient to establish strict forecast-time causality?
17. What evidence would be needed before claiming prospective superiority?
18. What parts of the work were completed with external/AI assistance, and what parts did the students independently design, execute, check, and understand?

The team should be able to derive or explain these from first principles rather than memorizing slogans.

---

## 15. Evidence index

Primary internal sources for final student-authored materials:

- `config/eight_issue_status_2026-09-08.json` — current eight-item delivery state;
- `architecture/BLACK_BOX_VALIDATION_2026-09-07.md` — independent package replay;
- `architecture/PROMOTED_STACK_MISSINGNESS_TRANSFER_RESULT_2026-09-06.md` — random missingness transfer evidence;
- `architecture/EVENT_TERMINAL_OUTAGE_RESULT_2026-09-07.md` — event-bearing outage evidence;
- `architecture/THRESHOLD_POLICY_RECONCILIATION_2026-09-06.md` — primary vs diagnostic threshold policies;
- `architecture/prospective_causal_interface_disposition_2026-09-08.json` — frozen live-interface blocker;
- `.github/workflows/iris-sep-prospective-source-preflight.yml` — source-acquisition witness workflow;
- workflow run `34217607075`, artifact `10052465094` — final all-source preflight;
- package build run `34141515134`, artifact `10026120558` — immutable V3 package;
- rejected filter runs `34206898291` and `34216432375` — negative alert-filter evidence.

---

## 16. Scientific bottom line

The most defensible present conclusion is:

**The project demonstrates a reproducible reliability-aware SEP forecasting architecture that separates numerical probability from permission to expose an alert, avoids runtime fabrication of missing sensor feeds, and fails closed when the exact frozen feature interface cannot be reproduced prospectively. Development performance remains imperfect and independent prospective skill has not yet been established.**

That limitation should remain visible. The scientific integrity of refusing an inadmissible live forecast is part of the result, not something to hide.
