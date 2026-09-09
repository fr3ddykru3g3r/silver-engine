# IRIS-SEP authoritative current status

**Status date:** 2026-09-09  
**Work branch:** `codex/iris-sep-continuation-20260905`  
**Purpose:** current scientific source of truth for the continuation branch. Historical development results remain preserved but cannot be relabelled as untouched evidence.

## Project in one sentence

Test whether conventional daily-window SEP forecasting metrics overstate genuine onset-forecasting skill because one physical SEP episode can contribute multiple correlated positive windows, already-active/persistence states can remain in the cohort, and issue-time proton information can leak persistence into a nominal 24-hour forecast; then evaluate models with an episode-normalized, causally eligible benchmark.

## Scientific pivot — authorized as a design, not yet as a result

The former freshness-crossover route is closed negative evidence. It is not to be tuned or rescued.

The new central question is methodological rather than architectural:

> **How much do event persistence, repeated positive windows from the same physical SEP episode, and issue-time availability choices change the apparent skill and ranking of 24-hour SEP forecasting systems?**

This is motivated by a concrete benchmark property: modern rolling-window SEP datasets can assign multiple positive windows to the same physical event when an event lasts longer than 24 hours. A window-level score can therefore give a long event more influence than a short event. In addition, proton-history features can encode that an event is already underway or recently active unless eligibility is defined causally.

The intended contribution is not “onset and persistence are different.” Operational agencies already distinguish them. The contribution to test is whether **standard evaluation choices materially change measured model skill, uncertainty, and model ranking**, and whether an episode-normalized causal evaluation provides a more defensible estimate of pre-onset warning ability.

## New candidate benchmark

Working name: `IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`.

Required paired evaluations on the same frozen prediction files:

1. `WINDOW_OCCURRENCE_STANDARD` — ordinary 24-hour window-level occurrence scoring retained as a literature-compatible reference.
2. `NEW_ONSET_CAUSAL` — issue time must be below the operational proton threshold and the next 24 hours must contain the first eligible threshold crossing.
3. `EPISODE_NORMALIZED_ONSET` — same onset eligibility, but every distinct physical SEP episode has total positive weight 1 regardless of duration or number of candidate windows.
4. `PROTON_STATE_BLIND_ONSET` — onset-eligible scoring with issue-time/in-window proton state variables removed from the learned feature set to quantify how much ranking depends on direct particle-history evidence.
5. `PERSISTENCE_DIAGNOSTIC` — already-active cases are scored separately and never mixed into onset skill.

Primary scientific outputs:

- delta in TSS between standard window scoring and episode-normalized onset scoring;
- delta in FAR at a matched episode-level detection target;
- model-rank changes across evaluation definitions;
- effective number of independent positive SEP episodes versus positive prediction windows;
- shared paired episode/bootstrap intervals using the same resampled episodes across all compared models/conditions;
- calibration at the episode/onset level;
- a full cohort/attrition ledger showing exactly why each candidate issue time is included or excluded.

## Claim boundary

No current result establishes that conventional SEP papers are wrong or that their published skill is inflated. The project is testing that hypothesis.

A competition-level claim is authorized only if the effect is large, stable across multiple fixed model families, and survives episode-level paired uncertainty. A null result is still publishable as a benchmark result if the methodology is complete and the result is retained without tuning.

## Protected evidence rule

`config/inspected_evidence_registry_v2.json` is authoritative for exposure control.

All historical periods already used for fitting, threshold selection, monitoring, diagnostics, freshness experiments, or policy selection remain development evidence. They may be used to build and debug the benchmark mechanics but never called untouched final evidence.

The candidate period after `2025-09-10T00:00:00Z` remains protected. Development-side code must not query, count, inspect, score, stratify, or otherwise reveal its outcomes for power rescue. Any independent final evaluation requires a custodian-controlled cohort/hash and pre-outcome contract freeze.

## Why this direction is stronger than another model

The current literature already contains deep neural networks, XGBoost, multimodal solar observations, proton/X-ray histories, and operational 24-hour SEP probabilities. Building another architecture is unlikely to supply competition-level creativity by itself.

A rigorous evaluation-bias study can instead contribute:

- a falsifiable scientific hypothesis about what forecast skill actually represents;
- a mathematical weighting correction for repeated windows from the same event;
- an operationally meaningful decomposition of onset versus persistence;
- reproducible paired uncertainty at the physical-episode level;
- a benchmark that can be applied to simple and sophisticated models alike.

## Existing frozen assets retained

The existing V3 package, missing-feed state machine, fail-closed interface checks, source receipts, negative filter experiments, and freshness closure remain part of the evidence history. They are not discarded, but they are no longer the centerpiece of the research question.

The prospective V3 interface still cannot be represented exactly because `CMASKL`, `MEANGBL`, and `USFLUXL` are absent from the required near-real-time interface, affecting 18 frozen feature-vector positions. The prior prospective run therefore emitted no forecast probability. This remains a valid fail-closed engineering result, not a forecast-skill result.

## What must be built before any new scoring

1. Freeze exact event/episode construction from proton flux without looking at protected outcomes.
2. Freeze causal eligibility at issue time, including treatment of already-active and recently ended episodes.
3. Persist one row per prediction with issue timestamp, episode identifier (only when labels are legally available for that cohort), eligibility reason, observed state, model probability, and threshold decision.
4. Implement episode-normalized weights and prove each positive episode sums to weight 1.
5. Implement a shared-resample episode bootstrap; every contrast and every model must use identical bootstrap draws.
6. Implement a complete attrition ledger.
7. Use fixed simple models before considering any architecture expansion: climatology, persistence diagnostic, elastic-net logistic regression, and frozen XGBoost.
8. Run synthetic/unit tests that establish the evaluator behaves correctly under deliberately constructed long-duration versus short-duration events.
9. Run only on already-exposed development data to debug mechanics; mark those outputs development-only.
10. Do not release protected final labels until an independent custodian verifies the frozen contract and cohort.

## 9/10 target rubric

The project should not be described as 9/10 until the evidence supports it. The engineering target is:

- **Scientific thought:** one sharply falsifiable hypothesis, physical-episode evaluation, causal eligibility, proper uncertainty, explicit failure conditions.
- **Creativity:** contribution is the episode-normalized causal benchmark and quantification of evaluation-induced skill/rank changes, not a generic new classifier.
- **Thoroughness:** complete prediction persistence, attrition ledger, multiple fixed comparators, shared bootstrap draws, sensitivity analyses, and preserved null/negative results.
- **Skill:** reproducible software contract, exact receipts/hashes, unit/property tests, episode-disjoint statistics, deterministic outputs, and independent replay where possible.
- **Clarity:** one judge-facing story: “Are we forecasting a new radiation storm, or partly recognizing one that is already happening?” followed by a single evaluation framework and a small set of decisive figures.

## Immediate next action

Implement and test the benchmark mechanics **without accessing protected outcomes**. The first milestone is a synthetic/property-tested evaluator proving that long-duration events cannot receive extra positive weight simply because they span more 24-hour windows, and that already-active cases are separated from new-onset cases before any model score is computed.
