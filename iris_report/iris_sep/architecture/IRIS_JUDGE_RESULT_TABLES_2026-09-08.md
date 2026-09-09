# IRIS-SEP judge-facing result tables — 2026-09-08

**Use:** compact verified tables for internal figure/poster planning. These are not a substitute for student-authored competition prose. Development results must remain labelled as development evidence.

## Table 1 — reliability states

| State | Evidence available | Runtime path | Permission | Runtime fabrication? |
|---|---|---|---|---|
| `FULL` | solar + XRS + proton | full frozen stack | normal only if provenance/interface gate passes | no |
| `NO_XRS` | solar + proton | pre-trained reduced-input fallback | `DEGRADED` | no |
| `NO_PROTON` | solar + XRS | pre-trained reduced-input fallback | `DEGRADED` | no |
| `NO_XRS_OR_PROTON` | solar only | diagnostic-only path | `ABSTAIN` / no alert | no |

## Table 2 — key development evidence

| Evidence | Result | Interpretation boundary |
|---|---:|---|
| Later 2023–2025 monitor TSS | ~0.2359 | inspected development evidence |
| Later monitor POD | ~83.3% | high detection does not imply low alert burden |
| Later monitor FAR | ~95.5% | major limitation; must remain visible |
| Previous late-fusion monitor TSS | ~0.1894 | paired advantage remains inconclusive |
| `NO_XRS` V3 TSS point estimate | 0.368087 -> 0.430492 | uncertainty crosses zero; Brier slightly worse |
| `NO_PROTON` V3 TSS point estimate | 0.507371 -> 0.512999 | small development gain |
| `NO_PROTON` detections | 16/21 retained | development evidence |
| `NO_PROTON` false positives | 814 -> 796 | modest development reduction |
| 40% random observed-cell loss | diagnostic-policy TSS drop ~0.227 | probability preservation did not ensure safe decisions |

## Table 3 — rejected monotone-veto experiment

Run: `34216432375`  
Status: **rejected; plain V3 remains frozen**

| State | Frozen V3 TP | Frozen V3 FP | Veto TP | Veto FP | FP reduction | TP change |
|---|---:|---:|---:|---:|---:|---:|
| `NO_XRS` | 13 | 603 | 12 | 431 | 28.5% | −1 |
| `NO_PROTON` | 16 | 796 | 13 | 594 | 25.4% | −3 |

Reason for rejection: large false-positive reductions were accompanied by lost detections. Further post-hoc tuning on the already-inspected score block was stopped to reduce overfitting risk.

## Table 4 — immutable package / replay evidence

| Check | Result |
|---|---|
| Serialized specialists | 15 |
| Package ZIP SHA-256 | `69be4a5d79c17e452d2fe6115c6447995f22297002a6b25e2dc692c760495a75` |
| Manifest SHA-256 | `88c1a73f73f9c3b48c7b5b3c59cdc42dd3d80421ec7b47829f8ab4036e8790a7` |
| Self-replay max absolute difference | 0.0 |
| Independent black-box replay max absolute difference | ~2.22e-16 |
| Ordered feature schema | bound by position |
| Runtime retraining | forbidden |
| Solar-only `ABSTAIN` alert emission | forbidden |

## Table 5 — prospective preflight

Final run: `34217607075`  
Artifact: `10052465094`  
Artifact ZIP SHA-256: `95ae852fb005674fcac555bc7c741bd1becab04ac2829823819ea6c0575ed5e3`

| Source family | Final status |
|---|---|
| NOAA primary proton | authenticated |
| NOAA primary XRS | authenticated |
| NOAA GOES instrument routing | authenticated |
| JSOC HMI SHARP CEA-NRT | authenticated source, but interface incomplete |
| LMSAL HEK GOES flares | authenticated |
| NASA CCMC DONKI CME | authenticated |
| NASA GSFC CDAW CME | external connection reset after five bounded retries |

Summary: **6/7 source families authenticated.** Full source authentication remained false.

## Table 6 — exact prospective feature-interface blocker

| Item | Frozen requirement / result |
|---|---|
| Total feature-vector positions | 259 |
| Solar/context positions | 251 |
| XRS positions | 4 |
| Proton positions | 4 |
| CEA-NRT missing frozen quantities | `CMASKL`, `MEANGBL`, `USFLUXL` |
| Frozen positions affected | 18 |
| Allowed to zero-fill/substitute/backcast? | no |
| Prospective frozen-model skill forecast admissible? | no |
| Forecast probability emitted by preflight? | no |

Frozen disposition: `FROZEN_MODEL_RETROSPECTIVE_ONLY_FOR_SKILL_UNTIL_SOURCE_EQUIVALENCE_IS_ESTABLISHED`.

## Table 7 — current eight-item delivery state

| # | Item | Current status |
|---|---|---|
| 1 | immutable V3 package | `PASS` |
| 2 | fresh trusted source acquisition | `CLOSED_WITH_EXTERNAL_BLOCKER` |
| 3 | exact causal 259-feature interface | `CLOSED_WITH_SCIENTIFIC_BLOCKER` |
| 4 | load-only inference and provenance | `PASS` |
| 5 | forecast seal and external witness | engineering complete; no admissible skill forecast to witness |
| 6 | fair same-cohort comparator | engineering complete; no admissible live case |
| 7 | correct 24h outcome | evaluator complete; no admissible live forecast to label |
| 8 | internal evidence / student writing support | `PASS_INTERNAL_EVIDENCE_PACKAGE` |

## Visual assets

- `../figures/FIGURE_1_RELIABILITY_STATE_MACHINE_2026-09-08.svg`
- `../figures/FIGURE_2_ALERT_VETO_TRADEOFF_2026-09-08.svg`
- `../figures/FIGURE_3_PROSPECTIVE_PREFLIGHT_FAIL_CLOSED_2026-09-08.svg`

## Claim boundary

These tables support a **retrospective reliability study with a prospective fail-closed validation framework**. They do not establish independent prospective V3 skill, operational superiority, low-FAR operation, economic benefit, operational certification, full-physics solar simulation, or any competition outcome.
