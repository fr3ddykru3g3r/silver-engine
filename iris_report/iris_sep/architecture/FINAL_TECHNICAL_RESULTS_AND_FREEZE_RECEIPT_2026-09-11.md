# IRIS-SEP final technical results and freeze receipt — 2026-09-11

## Disposition

`RETROSPECTIVE_TECHNICAL_STACK_FROZEN — REPRODUCIBILITY_VERIFIED — PROSPECTIVE_OUTCOMES_SEALED`

This receipt records the final technical state after the two-week execution plan was converted into code, validity contracts, tests, evidence packaging and an external-pilot scaffold. It does not convert historical evidence into prospective evidence and does not authorize protected-outcome access.

## Frozen historical baseline

- submission baseline commit: `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`;
- baseline source-only CI: run `34573237824`, PASS;
- latest implementation head verified before this documentation-only receipt: `6d746035b36589e4245337531fd8efd779cf9ad2`;
- latest source-only CI at that implementation head: run `34602124615`, SUCCESS;
- pandas 2.3.2: 398 passed, 10 explicitly registered data-bound tests deselected, 14 warnings;
- pandas 3.0.1: 398 passed, 10 explicitly registered data-bound tests deselected, 14 warnings;
- JSON configuration parse: PASS;
- explicit data-dependent-test registry validation: PASS.

The baseline commit remains the byte-identifiable scientific submission checkpoint even though later commits add execution-plan, validity-interface and packaging infrastructure.

## Immutable scientific artifacts

### SEP-PRISM model-free confirmation

- workflow run `34438057070`;
- artifact `10136909164`;
- independently re-hashed archive SHA-256: `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`.

### SEP-PRISM fixed-model replay

- workflow run `34438987251`;
- artifact `10137507101`;
- independently re-hashed archive SHA-256: `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`;
- bundled independent verifier: PASS;
- evidence files verified: 12;
- 7,558 unique score issues;
- 85 matched onset-episode units;
- 1,080 quiet-block units;
- recomputed gate: `STRONG_MODEL_CONFIRMATION`.

A portable evidence archive containing both immutable scientific artifact ZIPs, latest CI diagnostics, matched point metrics, bootstrap summaries, shared physical-unit draw tensors and checksums was generated during the final technical audit. It must be retained outside ephemeral GitHub Actions storage by the student team; this repository receipt does not claim that external archival step has already been performed.

## Model-free physical-event audit

- daily 24-hour windows: **14,464**;
- positives: **650**;
- stored-versus-reconstructed target mismatches: **0**;
- uniquely mapped positive windows: **614**;
- represented physical episodes: **257**;
- Episode Multiplicity Factor: **2.389105**;
- multi-episode-overlap positive windows excluded from EMF: **36**;
- already-active persistence windows: **418**;
- new-onset windows: **228**;
- ambiguous positive windows: **4**.

The 228 full-table onset windows are not interchangeable with the 85 distinct matched onset episodes used in the replay inference.

## Fixed replay — matched-population TSS

| Model | Mapped occurrence | Episode-normalized occurrence | New onset |
|---|---:|---:|---:|
| XGBoost joint | 0.726455 | 0.621128 | 0.437234 |
| XGBoost no-proton | 0.508922 | 0.514758 | 0.477391 |
| Past-proton >=10 proxy | 0.601224 | 0.446983 | 0.092894 |
| Elastic net joint | -0.122768 | -0.182816 | -0.266990 |
| Elastic net no-proton | -0.318921 | -0.332090 | -0.333015 |
| Fit-prevalence climatology | 0.000000 | 0.000000 | 0.000000 |

The main fixed joint-XGBoost comparator therefore changes `0.726455 -> 0.621128 -> 0.437234` while the prediction/alert record is held fixed and the evaluation question changes.

## Frozen paired bootstrap contrasts

Shared physical-unit bootstrap: 10,000 draws.

| Contrast | Point change | Bootstrap median | 95% percentile interval |
|---|---:|---:|---:|
| Joint XGB: episode-normalized minus mapped | -0.105327 | -0.103744 | [-0.146175, -0.063165] |
| Joint XGB: onset minus episode-normalized | -0.183894 | -0.183333 | [-0.246639, -0.125070] |
| Past-proton proxy: onset minus mapped | -0.508331 | -0.506167 | [-0.566675, -0.443973] |
| Joint XGB minus no-proton XGB, onset | -0.040157 | -0.041300 | [-0.162967, +0.082920] |
| Joint elastic minus no-proton elastic, onset | +0.066025 | +0.066698 | [+0.015149, +0.111101] |

All 10,000 stored paired draws are negative for each of the three preregistered directional evaluation-effect contrasts. This is a bootstrap diagnostic, not a classical p-value. The XGBoost joint-versus-no-proton interval crosses zero, so reliable XGBoost model superiority or a harmful-proton-feature conclusion is not supported. The elastic-net difference is positive, but both elastic-net onset TSS values are negative, so it is not an operational-success result.

## Joint-XGBoost causal-new-onset operating characteristics

Matched cohort:

- TP = 41;
- FN = 44;
- FP = 330;
- TN = 6,984;
- sensitivity/POD = **48.24%**;
- false-positive rate = **4.51%**;
- false-alarm ratio = **88.95%**;
- TSS = **0.437234**;
- HSS = **0.164201**;
- Brier score = **0.011391**.

The 88.95% value is the false-alarm ratio `FP/(TP+FP)`, not the quiet-day false-positive rate. The alert burden prevents an operational-readiness claim.

## Fold structure and frozen XGBoost thresholds

| Fold | Fit rows | Threshold rows | Score rows | Score positives | Joint XGB threshold | No-proton XGB threshold |
|---|---:|---:|---:|---:|---:|---:|
| OOF_2005_2010 | 5,079 | 1,827 | 2,191 | 46 | 0.04012613 | 0.00629751 |
| OOF_2011_2017 | 6,906 | 2,191 | 2,557 | 115 | 0.02526334 | 0.03127309 |
| OOF_2018_2025 | 9,097 | 2,557 | 2,810 | 83 | 0.03569588 | 0.01356591 |

The full six-comparator replay contains **45,348** prediction rows. The matched-sensitivity persisted table contains **45,066** rows after the frozen matching/eligibility construction.

## Forecast-validity and missing-data technical layer

The predictor-side validity interface is implemented as `VALID | DEGRADED | ABSTAIN` and fails closed. Exact frozen reason codes include:

- `STALE_INPUT`;
- `REQUIRED_FEED_ABSENT`;
- `AMBIGUOUS_PROTON_CHANNEL`;
- `STRUCTURAL_UNAVAILABILITY`;
- `EXCESSIVE_TRANSIENT_LOSS`;
- `CAUSAL_AVAILABILITY_RECEIPT_FAILED`;
- `LEDGER_INTEGRITY_FAILURE`;
- `FUTURE_OBSERVATION`;
- `PROVENANCE_UNCERTAIN`;
- `TRANSIENT_FORWARD_FILL_APPLIED` for a bounded degraded record.

`ABSTAIN` exposes no actionable alert (`alert=null`). Structural unavailability is never reconstructed and relabelled as observed data. Causal forward-fill can produce `DEGRADED` only inside separately frozen source-specific limits.

Development missingness stress evidence supports the limited observation that causal forward-fill preserved probability space reasonably under roughly 5–20% random transient loss, whereas a 40% random-loss stress condition materially degraded the tested operational-policy skill. These percentages are not universal operational thresholds. Real magnetic-map reconstruction remains experimental and cannot be promoted without a preregistered hidden real-map comparison against persistence plus preserved downstream new-onset utility.

The current source-only suite covers clean validity, future observations, staleness, absent feeds, structural absence, ambiguous proton semantics, failed causal receipts, uncertain provenance, ledger-integrity failure, bounded forward-fill, excessive transient loss and deterministic compound faults.

## Prospective confirmation state

`IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1 — FROZEN_NOT_EXECUTED`

- protected pool begins `2025-09-10T00:00:00Z`;
- development-side protected outcome access remains forbidden;
- missing/late/unverifiable required inputs cause `ABSTAIN`;
- every prospectively evaluated learned feature requires issue-time causal availability evidence;
- required comparator is the frozen pre-issue past-proton-active rule;
- bootstrap is frozen at 10,000 shared physical-unit draws, seed `20260911`;
- information floor is at least 50 distinct onset episodes plus 500 quiet blocks;
- failure to reach that floor returns `INSUFFICIENT_CONFIRMATORY_INFORMATION`;
- final protected evaluation is custodian-controlled and aggregate-only.

No protected post-2025 outcomes were accessed to produce this receipt.

## Final technical conclusion

The retrospective scientific claim and its computational evidence are complete and frozen. The supported conclusion is that **the same fixed 24-hour SEP forecast record can receive materially different measured skill when repeated representation of physical episodes and already-active persistence are separated from genuine new-onset warning**.

The project does not support an operational deployment claim, untouched prospective validation, universal model/feature superiority, a claim that all earlier SEP studies are biased, or economic/company-benefit claims.

Remaining work that cannot be completed by additional development-side coding is intentionally external or future-facing: student-owned final competition prose and forms, durable external artifact preservation, independent-custodian appointment/approval, genuinely pre-issue prospective evidence accrual, and documented stakeholder workflow feedback.
