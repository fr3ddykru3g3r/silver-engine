# IRIS-SEP — START HERE

**Active branch:** `codex/iris-sep-operational-reproducibility-20260921`  
**Active study:** `IRIS_SEP_OPERATIONAL_REPRODUCIBILITY_V1`  
**Frozen:** 21 September 2026

## Research question

> **Can a retrospective 24-hour solar energetic particle forecast be reproduced using only predictor values that were genuinely available before each forecast issue time?**

Secondary question: how much predictor coverage and forecast behavior remain when issue-time causal availability is enforced?

## Why this is the active project

Several earlier directions were abandoned after novelty checks. Onset/persistence verification, proton-history attribution, lag/forecastability curves and physics-guided counterfactual explanations all have close prior SEP literature. They remain historical evidence only.

The active novelty target is narrower: a **feature-by-feature causal-availability audit and quantitative retrospective-to-operational reproducibility gap for a modern multi-source SEP machine-learning interface**. We do not claim that latency or real-time availability themselves are new ideas.

## Read in this order

1. `ACTIVE_PROJECT_2026-09-21.md` — IB Physics-level question and experimental design.
2. `NOVELTY_AUDIT_OPERATIONAL_REPRODUCIBILITY_2026-09-21.md` — hostile novelty check and kill conditions.
3. `config/operational_reproducibility_preregistration_2026-09-21.json` — frozen rules before result hunting.
4. `config/operational_source_manifest_v1.json` — conservative source-family status ledger.
5. `tools/audit_operational_reproducibility_v1.py` — fail-closed interface audit.
6. `tests/test_operational_reproducibility_v1.py` — invariant tests.
7. `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md` — earlier causal-availability contract that motivated the new study.
8. `audit_20260915/AUDIT_REPORT.md` — preserved independent audit and historical evidence.

## Existing evidence boundary

The previous repository audit established a retrospective 259-predictor joint interface, but it did **not** establish that every predictor existed in an operationally equivalent form before its nominal forecast issue time. The prospective contract therefore already blocks the 259-column interface as a set until field-level availability is demonstrated.

That observation is motivation, not the final result. The new study must audit the feature lineage rather than assume the answer.

## Availability rule

A predictor may be labeled `VERIFIED` only when evidence supports all of the following:

- the physical variable and source are identified;
- measurement-time semantics are known;
- issue-time / first-seen availability is supported;
- units and field definitions match;
- retrospective correction/reprocessing is understood; and
- no future or outcome-dependent information is used.

Otherwise it remains `UNVERIFIED_LATENCY`, `SCHEMA_MISMATCH`, `RETROSPECTIVE_ONLY`, or `NO_EQUIVALENT`.

Unknown predictor columns fail closed.

## Protected-data rule

Protected post-`2025-09-10T00:00:00Z` outcomes remain sealed. They may not be used to choose sources, tolerances, features, thresholds, models or narrative. Historical pre-boundary analyses remain development-exposed evidence, not prospective validation.

## What this project is not

- not a new XGBoost algorithm;
- not a claim that the published model is invalid;
- not another generic missing-data experiment;
- not a claim that every retrospective forecast leaks future information;
- not a first-ever claim.

## Judge version

> A forecast can only use measurements that actually existed when it was issued. I am checking a modern solar-radiation forecast feature by feature to see whether its retrospective inputs can really be reconstructed from near-real-time data, then measuring what changes when I enforce that rule.

## Historical material

All earlier branches, fixed forecasts, negative results, source-preflight work and episode/onset analyses are preserved for provenance. They should not be reused as the current novelty narrative.
