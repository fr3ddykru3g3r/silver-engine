# IRIS-SEP black-box validation — 2026-09-07

## Scope

This is an **independent implementation / black-box validation pass**, not an external third-party certification and not an untouched final evaluation. The checks below were performed from exported GitHub Actions artifacts and public upstream documentation without importing the IRIS-SEP runtime implementation for the prediction reconstruction.

The purpose is to separate three questions that must not be collapsed into one score:

1. **Does the packaged software reproduce what it claims?**
2. **Do the development performance numbers recompute from the exported predictions?**
3. **Has the scientific claim been independently established on causal, untouched data?**

The answer is respectively: **yes**, **yes with qualifications**, and **not yet**.

## Frozen artifacts checked

- Full cross-fitted stack run: `33985243824`
  - GitHub artifact SHA-256: `ed04d5b34df094e2eb7f7db3f07b426619494b38987f1993c0faddca90d58f99`
- Distilled fallback V3 replay run: `34138370215`
  - GitHub artifact SHA-256: `0d6c38cc34d1d55604bade0611f37d5635620ad6112d35a9f193ef51b39b2390`
- Distilled V3 load-only package run: `34141515134`
  - GitHub artifact SHA-256: `69be4a5d79c17e452d2fe6115c6447995f22297002a6b25e2dc692c760495a75`

All three downloaded ZIP digests independently recomputed to the GitHub-reported artifact digests.

## 1. Package integrity and portability

The V3 package contains 15 serialized XGBoost Booster JSON files plus its manifest, package receipt, replay input, exported reference predictions and load-only CLI output.

Independent checks:

- all 15 model-file SHA-256 values recomputed correctly;
- manifest, replay input, reference prediction and CLI-output hashes recomputed correctly;
- all four feature-vector schema hashes recomputed from family order, within-family feature order and exact vector index;
- every schema row has contiguous indices and exactly matches the family feature lists;
- the package declares 4,297 cross-fitted OOF rows and 87 OOF positives;
- locked test access is declared false in the bound receipt.

The independent reconstruction ran with XGBoost `3.1.3`, whereas the package was produced with XGBoost `3.0.4`. The reconstruction loaded the native Booster JSON files and reimplemented the package mathematics directly:

1. five-seed median specialist prediction;
2. finite-fraction XRS/proton reliability;
3. prevalence-centred clipped logit evidence;
4. state-specific teacher/student evidence stack;
5. logit-intercept calibration.

Maximum absolute probability disagreement versus the package reference was `2.22e-16` across all states. This is effectively floating-point equality and is stronger portability evidence than same-environment self-replay.

**Disposition: PASS for package integrity and load-only reproducibility.**

## 2. Fail-safe availability behaviour

The package separates numerical threshold crossing from permission to expose an alert.

For `NO_XRS_OR_PROTON`, independent recomputation found:

- `MAX_TSS`: 5,830 numerical threshold crossings;
- `POD80_MIN_FAR`: 8,873 numerical threshold crossings;
- permitted alerts under the bound `ABSTAIN` policy: 0;
- actual CLI alerts: 0 under both policies.

This demonstrates that a solar-only probability cannot silently turn into a normal alert simply because it crosses a threshold.

**Disposition: PASS for the tested abstention safety contract.**

## 3. Chronology and role isolation

The full-stack prediction artifact contains the following ordered roles:

| Role | Rows | Positives | Date range |
|---|---:|---:|---|
| fit | 6,659 | 144 | 1986-02-04 to 2005-01-16 |
| calibration | 2,553 | 21 | 2005-01-23 to 2012-03-13 |
| threshold | 877 | 21 | 2012-03-16 to 2014-09-11 |
| score | 3,219 | 21 | 2014-09-13 to 2023-07-29 |
| monitor | 737 | 24 | 2023-07-31 to 2025-09-10 |

Independent checks found:

- strict chronological ordering between roles;
- zero `unit_id` overlap between any pair of roles;
- zero duplicate issue times;
- each unit has one label only.

**Disposition: PASS for the exported development partition mechanics.** This does not make the score or monitor fresh final evidence; both were already inspected during development.

## 4. Full-data predictor: empirical red team

Under the frozen primary `MAX_TSS` threshold policy, independent metric recomputation matches the full-stack artifact.

| Model | Score TSS | Monitor TSS | Score Brier |
|---|---:|---:|---:|
| Late fusion | 0.460511 | 0.178004 | 0.00665712 |
| IRIS cross-fitted evidence stack | 0.425712 | 0.220664 | 0.00634756 |

Therefore the cross-fitted stack is **not established as superior to late fusion**:

- score TSS difference, IRIS minus late fusion: approximately `-0.03480`;
- independent paired unit bootstrap 95% interval: approximately `[-0.1338, +0.0182]`;
- monitor TSS difference: approximately `+0.04266`;
- paired unit bootstrap 95% interval: approximately `[-0.0488, +0.1720]`.

The intervals cross zero. The cross-fitted stack does have better Brier score on the score role, so it remains a defensible development candidate, but the evidence does not support a superiority claim.

**Disposition: INCONCLUSIVE for comparative forecast-skill superiority.**

## 5. Distilled V3 missing-feed states

All values below were independently recomputed from the hash-bound V3 replay predictions using thresholds selected on the separate threshold role and evaluated on the 3,219-row score role.

### NO_XRS

- V1 TSS: `0.36808719735549006`
- V3 TSS: `0.4304922719556866`
- delta: `+0.06240507460019656`
- V1 Brier: `0.006422954687965309`
- V3 Brier: `0.006434312514196317`
- top-5% review capture: `8/21` for both
- independent stratified unit-bootstrap 95% interval for TSS delta: approximately `[-0.0356, +0.2057]`

The point estimate improves, but the interval crosses zero and Brier score is slightly worse.

**Disposition: PROMISING DEVELOPMENT RESULT; improvement not independently established.**

### NO_PROTON

- V1 TSS: `0.5073706781023855`
- V3 TSS: `0.5129991959260252`
- delta: `+0.00562851782363971`
- V1 Brier: `0.006415824074880846`
- V3 Brier: `0.006417693397774553`
- detections: `16/21` for both;
- false positives: `814 -> 796`;
- top-5% review capture: `6/21` for both;
- independent stratified unit-bootstrap 95% interval for TSS delta on this development cohort: approximately `[+0.0031, +0.0087]`.

The improvement is small and is driven by fewer false positives without losing detections. It remains development evidence because this score cohort has already informed architecture selection.

**Disposition: SMALL, INTERNALLY STABLE DEVELOPMENT IMPROVEMENT; not final independent evidence.**

## 6. Brier-skill reference qualification

The stored Brier scores are correct. The stored Brier Skill Score uses the **fit-role prevalence** (`0.021624868598888722`) as its frozen climatology reference, not the score-role prevalence (`0.006523765144454799`).

That is not future leakage because the fit prevalence predates scoring, but reports must name the reference explicitly. Against score-cohort climatology, the apparent Brier skill is much smaller. Brier Skill Score must therefore be reported as **skill versus frozen fit-role climatology**, not as an unlabeled generic BSS.

## 7. Provenance/security red team

The ordered feature-vector binding is position-sensitive and closes the same-length/reordered-vector loophole. The prospective bundle also recomputes both the vector schema and causal lineage gate during replay rather than trusting a stored `VALID` flag.

One important integrity boundary remains unresolved:

- provenance records require a non-empty `source_revision`, but the current contract does not yet bind that revision to an independently trusted acquisition artifact or signed/hashed source registry;
- adding a caller-supplied digest without a trusted registry would only create security theatre;
- a future causal-source pipeline should create immutable acquisition receipts and bind a trusted source-registry hash into prospective inference.

This is separate from the already-documented scientific limitation that the released aggregate table contains retrospective interpolation/backcasting and unresolved cell-level lineage.

**Disposition: STRONG FAIL-CLOSED SOFTWARE CONTRACT, but source identity and prospective causal lineage are NOT YET ESTABLISHED.**

## 8. Corrected component assessment

These scores deliberately separate engineering quality from scientific evidence.

| Component | Independent assessment | Reason |
|---|---:|---|
| Ordered schema / package integrity | 9.5/10 | adversarial binding + independent hash/order reconstruction |
| Runtime reproducibility / portability | 9.5/10 | independent implementation, different XGBoost version, `2.22e-16` max difference |
| Fail-safe availability behaviour | 9.5/10 | thousands of solar-only threshold crossings, zero permitted alerts |
| Runtime observability | 8.5/10 | useful five-seed disagreement diagnostics; not yet calibrated into trust policy |
| Chronology / role isolation | 9/10 | strict exported chronology and zero unit overlap |
| Full predictor comparative skill evidence | 6.5/10 | late-fusion comparison remains inconclusive |
| V3 `NO_XRS` skill evidence | 6.5/10 | positive point estimate; bootstrap interval crosses zero; Brier slightly worse |
| V3 `NO_PROTON` skill evidence | 7.5/10 | small, stable development FP reduction; no detection/review gain |
| Source provenance / causality evidence | 6/10 | good fail-closed contract, unresolved causal lineage/source authentication |
| Untouched final evaluation | 5/10 | no final independent cohort result yet |

A fair summary is approximately **9/10 engineering maturity for the promoted runtime path, but only ~6.5–7/10 scientific evidence maturity**. Those numbers should not be averaged into a single flattering system score.

## Final validation disposition

The validation supports the following claims:

- the load-only package is internally consistent, portable across the tested XGBoost versions and exactly reproduces its exported probabilities;
- feature ordering is cryptographically bound;
- the tested abstention policy is enforced by executable runtime behaviour;
- development metrics and chronology recompute from exported predictions.

The validation does **not** support:

- superiority of the full predictor over late fusion;
- a statistically established `NO_XRS` V3 improvement;
- prospective causality of the released aggregate feature table;
- authenticated source identity for prospective lineage records;
- NORMAL trust for missing-feed forecasts;
- operational certification or final independent forecast performance.

The next scientific priority is not more architecture capacity. It is a causally reconstructed source pipeline plus an untouched evaluation whose identities/outcomes were not used during development.