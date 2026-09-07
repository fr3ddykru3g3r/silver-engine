# Event-terminal outage evidence — 2026-09-07

## Decision

The preregistered event-terminal aggregate-interface outage experiment completed and passed an independent audit. Unlike the earlier label-blind daily-outage benchmark, every scenario here deliberately includes **all 21 eligible NEW-crossing positive score issue rows** plus 21 deterministic matched quiet controls. It therefore supplies event-preservation evidence for the three simple recovery arms.

This is development-only evidence on an already-inspected score cohort. It does not establish raw-sensor outage robustness, operational certification, or fresh final superiority.

## Immutable execution

- GitHub Actions run: `34052511341`
- source head: `e5ef75532684f8ff29654d242f60470ab8dd45ee`
- artifact ID: `9995086137`
- artifact ZIP SHA256: `2e0c832d949048344649708a03644219fb01454df9e13e9b15b752d24b1a851a`
- prediction CSV SHA256: `e7ce61671f625f754d4b1cb57239c6333493498f16c749fe302b83189fac75bb`
- summary SHA256: `acd73fb03e4edd6696cd9c64ece7e4a82ad4e9abfd5d3612f2a5209588321f08`
- preregistration ancestor: `430283f0dcc702139b38e041acb8186dd064bf80`
- independent audit: `PASSED`, 0 mismatches across 54 arm × threshold evaluations
- locked test accessed: **false**
- monitor used: **false**

Each modality-duration scenario contains 42 terminal decisions: 21 NEW-crossing positives and 21 deterministic quiet controls. The outage ends at the issue time, so causal recovery may use only information preceding the artificial outage.

## Benchmark-primary results: MAX_TSS

Each row is evaluated on the same 21 event terminals + 21 quiet controls. `TP/21` is event detection and `quiet FP/21` is the false-alert count on matched quiet controls.

| Missing family | Gap | Recovery | TP/21 | POD | quiet FP/21 | affected TSS |
|---|---:|---|---:|---:|---:|---:|
| Proton | 24 h | causal forward-fill | 12 | 0.571 | 7 | 0.238 |
| Proton | 72 h | causal forward-fill | 11 | 0.524 | 10 | 0.048 |
| Proton | 168 h | causal forward-fill | 11 | 0.524 | 8 | 0.143 |
| XRS | 24 h | causal forward-fill | 11 | 0.524 | 5 | 0.286 |
| XRS | 72 h | causal forward-fill | 14 | 0.667 | 6 | 0.381 |
| XRS | 168 h | causal forward-fill | 10 | 0.476 | 8 | 0.095 |
| XRS + proton | 24 h | causal forward-fill | 10 | 0.476 | 5 | 0.238 |
| XRS + proton | 72 h | causal forward-fill | 13 | 0.619 | 8 | 0.238 |
| XRS + proton | 168 h | causal forward-fill | 8 | 0.381 | 8 | 0.000 |
| Proton | any tested | mask-aware no-fill | 6 | 0.286 | 2 | 0.190 |
| XRS | any tested | mask-aware no-fill | 0 | 0.000 | 0 | 0.000 |
| XRS + proton | any tested | mask-aware no-fill | 0 | 0.000 | 0 | 0.000 |
| Proton | any tested | train-fit median | 16 | 0.762 | 11 | 0.238 |
| XRS | any tested | train-fit median | 16 | 0.762 | 9 | 0.333 |
| XRS + proton | any tested | train-fit median | 21 | 1.000 | 18 | 0.143 |

The apparent constancy of some no-fill/median results across durations follows from the aggregate interface and recovery definition; it is not evidence that duration is irrelevant at the raw-sensor level.

## What the event-bearing experiment changes

### 1. `MASK_AWARE_NO_FILL` is not an acceptable general recovery strategy

For XRS-only and combined XRS+proton loss, no-fill detects **0/21** event terminals under MAX_TSS. Its zero quiet false alerts therefore reflect suppression, not safe forecasting. It is retained as a diagnostic/fail-closed input state, not promoted as an event-preserving recovery method.

### 2. `TRAIN_FIT_MEDIAN` is not promoted despite high event detection

Median fill detects 16/21 proton or XRS events and 21/21 combined-loss events in this event-terminal stress test. However, it produces 9–18 false alerts on only 21 matched quiet controls. More importantly, the independently audited label-blind quiet-window experiment already showed severe false-alert inflation: for a 168 h combined XRS+proton outage, MAX_TSS false alerts increased from 6 in the clean reference to **27** under train-median. Because the method's apparent event sensitivity comes with unstable false-alert behavior, it is rejected as an operator recovery default.

### 3. Causal forward-fill is the only simple imputation arm that remains admissible for bounded use

Forward-fill had the smallest probability drift in all nine earlier label-blind quiet-window scenarios and maintains nonzero event detection here. But it is not universally safe:

- **Proton loss:** all 24/72/168 h point estimates clear the preregistered simple detection gate (affected TSS > 0 and POD >= 0.50). The 72 h affected-row paired TSS difference versus clean is significantly negative (95% interval approximately -0.559 to -0.022), so this is not evidence of equivalence to clean data.
- **XRS loss:** 24 h and 72 h clear the point-estimate detection gate; 168 h fails POD >= 0.50 (10/21 detections).
- **Combined XRS+proton loss:** the results are non-monotonic and 24 h and 168 h fail the detection gate. A 72 h point estimate cannot justify a bizarre rule that allows a longer gap after a shorter gap fails. Therefore **combined-feed loss is not granted a simple-imputation DEGRADED permission**.

These are development admission decisions, not operational certification.

## Secondary POD80/min-FAR evidence

The secondary reliability policy changes some point estimates but does not overturn the conservative decision. Forward-fill event detection under POD80 is:

- proton: 14/21 (24 h), 17/21 (72 h), 14/21 (168 h);
- XRS: 15/21, 15/21, 12/21;
- combined: 12/21, 14/21, 10/21.

Train-median again raises event detection but with large matched-control alert counts, while no-fill remains especially poor for XRS/combined loss. MAX_TSS remains benchmark-primary by contract.

## Recovery decision after this experiment

The simple recovery hierarchy is now:

1. **Observed primary source** — use normally if all provenance/admission checks pass.
2. **Verified alternate observed source** — preferred over synthetic recovery after source-specific harmonization is validated.
3. **Availability-conditioned specialist fallback** — preferred next if its separately preregistered experiment passes; this avoids fabricating missing measurements altogether.
4. **Causal forward-fill** — bounded development fallback only where a frozen evidence rule passes.
5. **Mask-aware degraded forecast** — diagnostic/fail-closed route, not assumed event-preserving.
6. **ABSTAIN** — mandatory when the availability state/gap is not supported.

`TRAIN_FIT_MEDIAN` is **rejected as the operator default**. Reduced-physics reconstruction remains **unpromoted** until it beats the simpler causal control on hidden truth and downstream forecast preservation.

## Conservative provisional boundaries pending specialist-fallback result

- Proton-only outage: forward-fill may remain a **DEGRADED development candidate** through the tested 168 h aggregate horizon, but must carry an explicit warning that 72 h showed significant affected-row skill loss versus clean.
- XRS-only outage: forward-fill may remain a **DEGRADED development candidate through 72 h**; at 168 h it fails the primary detection gate and requires ABSTAIN unless a verified specialist fallback passes.
- XRS + proton outage: **ABSTAIN** under simple imputation. Do not exploit the non-monotonic 72 h point estimate.
- No missing-data state is promoted to `NORMAL` from this development study. Fresh independent evaluation is required.

If the separately preregistered availability-conditioned specialist fallback passes its gates, it supersedes forward-fill for the relevant missing-family state because it uses only actually available expert inputs and performs no runtime imputation.

## Claim boundary

Supported claim: deliberately removing XRS/proton information can materially change event detection and false-alert behavior; no-fill and train-median are unsafe as universal defaults; causal forward-fill is the most stable simple fill but requires bounded, modality-specific admission, and unsupported states should abstain.

Not supported: perfect recovery of missing measurements, raw five-minute sensor-outage robustness, physics-reconstruction superiority, operational certification, company superiority, economic savings, or final locked-test skill.
