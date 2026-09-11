# IRIS-SEP authoritative current status

**Status date:** 2026-09-11  
**Work branch:** `codex/iris-sep-episode-benchmark-v1-20260909`  
**Umbrella continuation:** `codex/iris-sep-continuation-20260905`  
**Purpose:** scientific source of truth. Historical inspected results remain development evidence and are never relabelled as untouched final evidence.

## Project in one sentence

**Are 24-hour SEP forecasting scores measuring prediction of a new radiation storm, or partly rewarding recognition of a storm that is already active and repeated counting of the same physical event?**

The centerpiece is the episode-normalized causal evaluation benchmark, not a claim of a new state-of-the-art forecasting architecture.

## Current disposition

`SEP_PRISM_FIXED_MODEL_REPLAY_V1 — STRONG_MODEL_CONFIRMATION_ON_PREVIOUSLY_EXPOSED_PUBLIC_DATA`

The evaluation effect reproduces in a preregistered, higher-powered chronological replay on public historical data. The source hashes were already recorded as development-inspected, so this is strong methodological confirmation rather than untouched final evidence. The protected post-2025 cohort remains sealed.

### SEP-PRISM model-free confirmation

The pinned 1986–2025 table contains 14,464 daily windows and 650 stored positives, with zero stored-versus-reconstructed target mismatches. The onset-state audit finds 418 already-active persistence windows, 228 genuine new-onset windows and four ambiguous positives. The Episode Multiplicity Factor is **614 uniquely mapped positive windows / 257 represented physical episodes = 2.389**, because 36 multi-episode-overlap positives are excluded from that calculation. Persistence exceeds onset in 1986–2010, 2011–2017 and 2018–2025; their respective EMFs are 2.46, 2.52 and 1.95.

### SEP-PRISM frozen fixed-model replay

The replay covers 7,558 unique daily score issues across three chronological out-of-fold periods from 2005–2025. Six fixed comparators were evaluated under unchanged alerts. The matched inferential cohort contains 85 onset episodes and 1,080 Monday-anchored quiet blocks, with 10,000 shared physical-unit bootstrap draws.

| Frozen directional TSS contrast | Point change | Paired 95% interval |
|---|---:|---:|
| Joint XGBoost: episode-normalized minus mapped occurrence | -0.105 | [-0.146, -0.063] |
| Joint XGBoost: onset minus episode-normalized occurrence | -0.184 | [-0.247, -0.125] |
| Past-proton proxy: onset minus mapped occurrence | -0.508 | [-0.567, -0.444] |

All three intervals lie below zero, satisfying the frozen `STRONG_MODEL_CONFIRMATION` rule. On the same matched population, joint-XGBoost TSS changes **0.726 -> 0.621 -> 0.437** across mapped occurrence, episode normalization and onset. Its onset confusion table is TP=41, FN=44, FP=330 and TN=6,984 (sensitivity 48.24%, FAR 88.95%), so the result is evidence about evaluation sensitivity, not operational readiness.

The joint-XGBoost minus no-proton onset contrast is -0.040 with interval [-0.163, +0.083], which does not establish a reliable rank reversal. Elastic-net's corresponding contrast is positive, but both elastic-net onset TSS values are negative. No broad proton-feature or model-superiority claim is supported.

### External public-data arm

A pinned audit of `yuyian/SEP-Prediction` found 11,773 24-hour windows, 1,726 stored operational positives, 411 already-active persistence windows versus 227 new-onset windows, and 610 uniquely mapped positive windows representing 256 physical episodes (Episode Multiplicity Factor **2.3828125**). Under the audited event-table semantics, 1,083 stored positives do not overlap a reconstructed >=10 pfu operational episode in the nominal future interval. This motivates the benchmark; it does not prove the paper's final SEPVAL score is wrong.

### Earlier internal development arm

The exposed 2014–2017 expanding-OOF cohort contained 936 scored issues but only five distinct new-onset episodes. It remains supporting development evidence, not the main power claim. Its strongest mapped-episode effects were joint-XGBoost multiplicity shift -0.133 [-0.225,-0.041], joint-XGBoost persistence shift -0.400 [-0.800,-0.125], and current-proton-active persistence shift -0.567 [-0.867,-0.267].

## Frozen prospective confirmation — new on 2026-09-11

`IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1 — FROZEN_NOT_EXECUTED`

The next confirmation is now preregistered **without inspecting, counting or scoring protected post-2025 outcomes**. It is intentionally harder than the historical replay:

- protected pool begins `2025-09-10T00:00:00Z`;
- development-side label, event-count and episode-identity access remain forbidden;
- every predictor must have an issue-time feature-availability receipt before use;
- missing or late required input causes `ABSTAIN`, not retrospective substitution;
- the historical SEP-PRISM 259-variable feature set is blocked as a prospective set until every field is causally verified;
- the required comparator is a frozen pre-issue past-proton-active rule; learned operational models are optional and can enter only if serialized, hashed and frozen before outcome access;
- inference uses the same three evaluation views: mapped occurrence, episode-normalized occurrence and causal new onset;
- bootstrap is frozen at 10,000 shared physical-unit draws, seed `20260911`;
- confirmatory information floor is **>=50 distinct onset episodes and >=500 quiet blocks**, chosen before protected counts were inspected;
- if that floor is not met, the only allowed scientific disposition is `INSUFFICIENT_CONFIRMATORY_INFORMATION`;
- the custodian runner emits aggregate results and receipts only, never row-level protected labels or episode identities to the development side.

Primary required prospective contrast:

`past_proton_active_proxy: NEW_ONSET_CAUSAL TSS - MAPPED_OCCURRENCE TSS`

A prospective confirmation requires the information floor and a paired 95% percentile interval strictly below zero. No protected execution is authorized yet because issue-time latency/availability receipts and a final `EXECUTION_FROZEN` manifest are still required.

Read:

- `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json`
- `config/prospective_operational_execution_manifest_template_v1.json`
- `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md`
- `architecture/PROSPECTIVE_CUSTODIAN_HANDOFF_2026-09-11.md`
- `tools/run_custodian_prospective_episode_evaluation_v1.py`

## Authoritative evidence receipts

SEP-PRISM model-free confirmation:

- workflow run `34438057070`
- artifact `10136909164`
- archive SHA-256 `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`
- row-level target reconstruction: PASS

SEP-PRISM fixed-model replay:

- workflow run `34438987251`
- source commit `296f302111371e8d421fb5f2a4569ea2c06bd6e1`
- artifact `10137507101`
- archive SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
- supplied verifier: PASS
- separate independent reconstruction: PASS

Audit-corrected earlier development workflow:

- run `34371418428`
- commit `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact `10112207048`
- artifact SHA-256 `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`

Prospective-contract source-only verification at preregistration head `749967ccd99a9d4187ea2a82add79353698c708f`:

- dedicated prospective contract workflow: PASS on Python 3.11 and 3.12;
- full IRIS-SEP source-only verification run `34570209759`: PASS.

No protected data are accessed by the prospective source-only workflow.

## Post-result correction boundary

Independent audit of the earlier development package found an OOF threshold-summary bug and an estimand-labeling ambiguity. The correction changed **only** the result/evidence layer: no refit, no threshold reselection, no feature change, no hyperparameter search, and no protected-outcome access. Legacy uncorrected tables remain preserved.

## Protected evidence rule

`config/inspected_evidence_registry_v2.json` remains authoritative.

Post-`2025-09-10T00:00:00Z` candidate outcomes remain protected. Development-side work must not query, count, inspect, score or stratify them, including for power rescue.

The historical replay is **development/methodology evidence**, not independent final evidence. The prospective contract does not change that classification until a valid custodian-controlled execution actually occurs.

## Historical V3 reliability engineering — preserved, not the centerpiece

The frozen V3 work remains valid historical evidence.

**Status date:** 2026-09-08

The V3 model remains **RETROSPECTIVE** for skill because the exact prospective interface cannot be reproduced. `CMASKL`, `MEANGBL`, and `USFLUXL` are unavailable in the required near-real-time interface, affecting **18 frozen feature-vector positions**. The prospective fail-closed run emitted **no forecast probability**.

This historical fact must not be mixed with the new episode-normalized model-score result.

The V3 `NO_XRS_OR_PROTON` state remains `ABSTAIN`, with no alert. Prior post-hoc alert filters remain rejected because false-alert reductions came with lost detections.

## Scientific claim boundary

Supported:

- repeated physical-event representation exists in current public SEP rolling benchmarks;
- already-active persistence can contribute strongly to occurrence-style skill while being a different task from new-onset forecasting;
- in the higher-powered historical replay, all three frozen directional evaluation-effect intervals were below zero;
- physical-episode weighting and onset/persistence separation materially change measured skill for some fixed models;
- an aggregate-only prospective confirmation protocol is frozen without protected-outcome inspection.

Not supported:

- all prior SEP papers are wrong or biased;
- SEPNET's final published SEPVAL score is wrong;
- independent prospective superiority;
- prospective confirmation before the frozen custodian study is validly executed;
- operational readiness/certification;
- state-of-the-art model performance;
- economic savings;
- award outcome.

## 9/10 competition target

The project now has a credible high-level competition structure:

- **Scientific thought:** falsifiable mechanism + causal estimands + paired physical-unit uncertainty + a predeclared prospective falsification path.
- **Creativity:** benchmark contribution rather than another classifier.
- **Thoroughness:** public audits, 85-episode replay, attrition/evidence ledgers, fixed comparators, preserved negative/legacy evidence, shared bootstrap tensors and a protected-outcome governance contract.
- **Skill:** pinned environments, immutable receipts, independent reconstruction, double hash verification, fail-closed feature availability and aggregate-only custodian tooling.
- **Clarity:** one question — “forecasting a new storm, or recognizing one already happening?”

Remaining weaknesses are evidence independence, issue-time availability for operational predictors, full training-object replay, and prospective validation. Do not disguise them.

## Immediate next actions

1. Keep PR #5 synchronized with this status and the green prospective-contract CI receipt.
2. Preserve all SEP-PRISM archive digests, independent receipts and exact matched-population denominators.
3. Keep the protected final cohort sealed; do not inspect it for event counts or power.
4. Complete issue-time first-seen/latency receipts for the minimum prospective input set.
5. Freeze a completed `EXECUTION_FROZEN` manifest only after every evaluated feature passes the availability gate.
6. Hand the sealed evaluation to an independent custodian; development-side execution is not authorized.
7. Do not tune or rescue the historical replay after score inspection.
