# SOL continuation prompt

Continue IRIS-SEP from the **verified SEP-PRISM episode benchmark + frozen prospective confirmation contract**, not from the older freshness or V3 modeling loops.

## Branches and PR

Primary branch: `codex/iris-sep-episode-benchmark-v1-20260909`  
Umbrella continuation: `codex/iris-sep-continuation-20260905`  
Dedicated PR: **#5**, open against the umbrella branch. Do not merge automatically.

Always inspect the exact branch head and PR state before writing.

## Central research question

> Are 24-hour SEP forecasting systems forecasting a new radiation storm, or do ordinary scores partly reward recognition of already-active storms and repeated counting of one physical SEP episode?

The contribution is an evaluation benchmark, not a new state-of-the-art forecasting architecture.

## Read first

1. `CURRENT_STATUS.md`
2. `architecture/EPISODE_NORMALIZED_CAUSAL_BENCHMARK_RESULT_2026-09-10.md`
3. `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md`
4. `architecture/PROSPECTIVE_CUSTODIAN_HANDOFF_2026-09-11.md`
5. `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json`
6. `config/inspected_evidence_registry_v2.json`

## Higher-powered SEP-PRISM result

Model-free confirmation:

- 14,464 daily windows, 650 stored positives, zero target-reconstruction mismatches;
- 614 uniquely mapped positive windows / 257 represented physical episodes = EMF 2.389;
- 418 persistence, 228 onset, four onset-state ambiguities;
- effect persists outside the original internal period.

Receipt: run `34438057070`, artifact `10136909164`, SHA-256 `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`.

Frozen fixed-model replay:

- 7,558 unique OOF score issues, six fixed comparators;
- matched inferential cohort: 85 onset episodes + 1,080 quiet blocks;
- 10,000 shared physical-unit bootstrap draws;
- joint XGB multiplicity contrast: -0.105 point, 95% [-0.146,-0.063];
- joint XGB persistence contrast: -0.184, 95% [-0.247,-0.125];
- past-proton proxy persistence contrast: -0.508, 95% [-0.567,-0.444];
- all three frozen directional intervals below zero;
- joint XGB onset TP=41, FN=44, FP=330, TN=6,984: this is evaluation-sensitivity evidence, not operational readiness;
- joint-minus-no-proton onset -0.040 [-0.163,+0.083]: reliable rank reversal not established.

Receipt: run `34438987251`, source commit `296f302111371e8d421fb5f2a4569ea2c06bd6e1`, artifact `10137507101`, SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`. Supplied and independent reconstruction both PASS.

## Evidence class

The SEP-PRISM public source files were already development-inspected. This is **strong methodological confirmation on exposed historical data**, not an untouched test. Do not relabel it as independent prospective validation.

Known limitations remain:

- not every historical predictor has a publication-time availability receipt;
- no programmed purge at every fit/threshold/score boundary;
- catalogue-derived onset eligibility is retrospective;
- the replay artifact lacks fitted models and threshold-block probabilities for complete training reconstruction;
- no operational readiness claim.

## Protected final evidence

Post-`2025-09-10T00:00:00Z` candidate outcomes are sealed.

Development-side work must not query, count, inspect, score, stratify, or reveal protected labels, positive counts, event timestamps or episode identities. Do not use protected data for power rescue, model selection, threshold changes or narrative tuning.

## Frozen prospective confirmation V1

Study ID: `IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`  
Status: `FROZEN_NOT_EXECUTED`.

This was preregistered on 2026-09-11 without protected-outcome inspection.

Frozen rules:

- daily issue at 00:00 UTC;
- >10 MeV, >=10 pfu target over `(issue,issue+24h]`;
- already active at issue = persistence, never onset;
- every predictor must have an issue-time feature-availability receipt;
- any missing/late required input => `ABSTAIN`;
- historical 259-variable SEP-PRISM set is blocked prospectively unless each field is independently verified causal;
- required model slot: deterministic `past_proton_active_proxy`;
- optional learned joint/proton-free models only if exact feature allowlists, model objects and thresholds are frozen before outcome access;
- 10,000 shared physical-unit bootstrap draws, seed `20260911`;
- minimum information floor: **50 distinct onset episodes + 500 quiet blocks**;
- if floor fails: `INSUFFICIENT_CONFIRMATORY_INFORMATION`; no redesign on the same cohort;
- mandatory contrast: `past_proton_active_proxy NEW_ONSET_CAUSAL TSS - MAPPED_OCCURRENCE TSS`;
- confirmation requires the information floor plus a paired 95% interval strictly below zero;
- aggregate-only custodian output; no row-level protected results returned before conclusion freeze.

## Current prospective implementation

Files:

- `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json`
- `config/prospective_operational_execution_manifest_template_v1.json`
- `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md`
- `architecture/PROSPECTIVE_CUSTODIAN_HANDOFF_2026-09-11.md`
- `tools/run_custodian_prospective_episode_evaluation_v1.py`
- `tests/test_prospective_operational_confirmation_contract.py`
- `.github/workflows/iris-sep-prospective-contract-source-only.yml`

The evaluator is aggregate-only and refuses to run without `--custodian-mode`, a valid `EXECUTION_FROZEN` manifest, valid model/rule and feature-schema hashes, and `features_verified_causal=true` for each model.

The dedicated prospective source-only workflow passed on Python 3.11 and 3.12 at head `749967ccd99a9d4187ea2a82add79353698c708f`. Full source-only verification run `34570209759` also passed.

## Operational feature-availability status

Current NOAA SWPC interfaces show that relevant primary-GOES proton and X-ray streams exist. That **does not** yet establish a guaranteed issue-time publication latency or historical interface equivalence.

Current disposition:

- primary GOES >10 MeV proton flux: `SOURCE_PRESENT — LATENCY RECEIPT REQUIRED`;
- primary GOES XRS: `SOURCE_PRESENT — LATENCY/SCHEMA RECEIPT REQUIRED`;
- historical SEP-PRISM 259 predictor columns as a set: `BLOCKED_AS_A_SET`;
- retrospective event catalogues: `OUTCOME_ONLY`.

Do not change `features_verified_causal` to true until a first-seen/issue-time receipt is actually demonstrated.

## Historical V3 evidence

Preserve but do not center it. The exact prospective V3 interface remains blocked because `CMASKL`, `MEANGBL`, and `USFLUXL` are missing, affecting 18 frozen feature-vector positions. Its prospective fail-closed preflight emitted no forecast probability.

## Freshness V1

Closed negative result. Do not reopen, tune or reinterpret it.

## Next legitimate work

1. Keep PR #5 synchronized; do not merge automatically.
2. Complete **source readiness only** for the minimum prospective feature set: first-seen timing, observation timestamp semantics, source identity, fill/QC behavior, and SHA-bound receipts.
3. If a learned operational model is feasible, freeze it and its threshold before any protected outcome access; otherwise proceed prospectively only with the deterministic mechanism comparator when source readiness is valid.
4. Change the execution manifest to `EXECUTION_FROZEN` only after every input passes the availability gate.
5. Hand the sealed evaluation to an independent custodian. Development-side execution is not authorized.
6. Accept `NO_PROSPECTIVE_CONFIRMATION` or `INSUFFICIENT_CONFIRMATORY_INFORMATION` if that is what the frozen study returns.
7. No post-hoc model rescue.

## Claim to use now

> On preregistered historical replay data, conventional SEP occurrence scores materially change when repeated physical-event representation and already-active persistence are separated. A prospective, issue-time-causal confirmation protocol is now frozen, but protected outcomes remain sealed and no prospective claim has yet been made.
