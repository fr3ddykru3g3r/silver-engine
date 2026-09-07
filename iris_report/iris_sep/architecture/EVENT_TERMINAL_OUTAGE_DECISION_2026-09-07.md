# IRIS-SEP event-bearing outage decision — 2026-09-07

## Evidence anchor

This decision uses the preregistered development-only event-terminal outage run:

- GitHub Actions run: `34052511341`
- source head: `e5ef75532684f8ff29654d242f60470ab8dd45ee`
- artifact: `iris-sep-event-terminal-outage-e5ef75532684f8ff29654d242f60470ab8dd45ee`
- artifact SHA-256: `2e0c832d949048344649708a03644219fb01454df9e13e9b15b752d24b1a851a`
- predictions SHA-256: `e7ce61671f625f754d4b1cb57239c6333493498f16c749fe302b83189fac75bb`
- summary SHA-256: `acd73fb03e4edd6696cd9c64ece7e4a82ad4e9abfd5d3612f2a5209588321f08`
- independent audit: `PASSED`, zero mismatches across 54 arm/threshold evaluations
- locked test accessed: **false**
- monitor used: **false**

Each scenario contains 21 positive NEW-SEP issue rows and 21 deterministic quiet controls. The artificial outage terminates at the issue time. Causal forward-fill may use only observations from before the outage began.

## Clean reference

On the full development score role, the frozen promoted stack has:

- MAX_TSS policy: TSS `0.4257`, POD `0.5714`, FAR `0.9749`
- POD80/min-FAR reliability policy: TSS `0.5120`, POD `0.7143`, FAR `0.9773`
- AUROC `0.8688`, AUPRC `0.0767`, Brier `0.00635`, ECE `0.00378`

The very high FAR is an unresolved forecasting limitation caused by the extreme rarity of NEW-SEP initiation. Missing-data handling does not erase this limitation.

## Positive-event outage results

The table below uses the separately declared POD80/min-FAR reliability policy on the 42 affected event+quiet rows. The reference row for every scenario is the same clean forecast on those 42 identities: POD `0.7143`, FAR `0.4231`, TSS `0.1905`.

| Outage | Recovery | POD | FAR | TSS | ΔTSS vs clean | 95% paired ΔTSS interval |
|---|---|---:|---:|---:|---:|---:|
| Proton 24 h | causal forward-fill | 0.6667 | 0.4400 | 0.1429 | -0.0476 | [-0.2174, 0.1111] |
| Proton 72 h | causal forward-fill | 0.8095 | 0.3704 | 0.3333 | +0.1429 | [0.0000, 0.3103] |
| Proton 168 h | causal forward-fill | 0.6667 | 0.3913 | 0.2381 | +0.0476 | [-0.1528, 0.2665] |
| XRS 24 h | causal forward-fill | 0.7143 | 0.3750 | 0.2857 | +0.0952 | [-0.1650, 0.3593] |
| XRS 72 h | causal forward-fill | 0.7143 | 0.3478 | 0.3333 | +0.1429 | [-0.1524, 0.4583] |
| XRS 168 h | causal forward-fill | 0.5714 | 0.4545 | 0.0952 | -0.0952 | [-0.4189, 0.2200] |
| XRS+proton 24 h | causal forward-fill | 0.5714 | 0.4000 | 0.1905 | 0.0000 | [-0.2526, 0.2488] |
| XRS+proton 72 h | causal forward-fill | 0.6667 | 0.3913 | 0.2381 | +0.0476 | [-0.2933, 0.3810] |
| XRS+proton 168 h | causal forward-fill | 0.4762 | 0.4444 | 0.0952 | -0.0952 | [-0.4662, 0.2955] |

These intervals are broad because there are only 21 positive event identities in the score block. Point estimates are therefore used only to set a conservative research support boundary, not to claim operational superiority.

## Decision across both outage experiments

The prior label-blind contiguous-outage experiment and this event-bearing experiment must be interpreted together.

### CAUSAL_FORWARD_FILL — **ADMIT AS DEGRADED RESEARCH FALLBACK THROUGH 72 HOURS**

Reason:

1. In the prior quiet-window benchmark it produced the smallest probability distortion in all nine modality-duration scenarios.
2. In this event-bearing benchmark, all three modality cases at 72 h have non-negative TSS point differences versus the corresponding clean affected-row reference.
3. XRS and combined 168 h outages reduce event-bearing TSS; therefore support is not extended to 168 h.
4. Recovery is never relabelled as observed and never receives `VALID` status.

This is a conservative *development support boundary*, not an operationally certified maximum outage duration.

### TRAIN_FIT_MEDIAN — **REJECT FOR OPERATOR AUTO-RECOVERY**

Reason:

- Although some event-bearing proton scenarios show favorable point estimates, the prior contiguous quiet-window experiment showed pathological false-alert inflation, including 27 false alerts versus 6 clean-reference alerts for a 168 h combined outage at MAX_TSS.
- A method whose apparent value flips between event-stratified and ordinary outage contexts is not stable enough for automatic recovery.

Train-median remains a benchmark control only.

### MASK_AWARE_NO_FILL — **DO NOT EXPOSE AS A NORMAL FORECAST FOR CRITICAL XRS/PROTON OUTAGES**

Reason:

- On event-bearing XRS outages it reduces POD from `0.7143` to `0.1429` under the POD80-derived frozen threshold.
- On combined XRS+proton outages it produces zero event detections at that operating point.

A mask-aware path may remain useful internally, but if no validated reconstruction/alternate observation exists the operator-facing system should fail closed rather than call the output normal.

### PHYSICS RECONSTRUCTION — **HOLD**

Physics does not enter the promoted operator path merely because it is more sophisticated. It must beat causal forward-fill on the same hidden observations and preserve downstream NEW-SEP forecasting better without calibration degradation. Until then, it remains experimental.

## Frozen operator rule from current evidence

For XRS and proton modalities:

1. use a fresh approved primary observation when available (`VALID`);
2. otherwise use a verified harmonized alternate real observation (`VALID`);
3. for a transient outage of **<=72 h**, causal forward-fill may be admitted only with causal provenance and the correct evidence receipt (`DEGRADED`);
4. do not use train-median as automatic recovery;
5. for a critical transient outage **>72 h**, or if recovery provenance/evidence fails, **ABSTAIN** unless a later preregistered reconstruction method earns a wider support envelope;
6. structural historical unavailability is never treated as a transient outage or reconstructed observation.

## Claim boundary

The defensible result is:

> In a preregistered development stress test containing actual NEW-SEP positive issue rows, causal forward-fill was the only simple recovery method that remained sufficiently consistent across both ordinary outage and event-bearing evaluations to justify a conservative DEGRADED research fallback, currently bounded at 72 h. Beyond that boundary IRIS-SEP fails closed.

Do **not** claim that 72 h is an operationally safe universal limit, that reconstruction improves every event, or that this test proves spacecraft-operations benefit. Those require independent prospective evaluation.