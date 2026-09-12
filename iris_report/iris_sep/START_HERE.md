# IRIS-SEP GitHub continuation

**Primary work branch:** `codex/iris-sep-episode-benchmark-v1-20260909`  
**Umbrella continuation:** `codex/iris-sep-continuation-20260905`  
**Dedicated PR:** #5, open against the umbrella branch. Do not merge automatically.

## Read this first

**Authoritative scientific status:** `CURRENT_STATUS.md`.

**Final technical freeze receipt:** `architecture/FINAL_TECHNICAL_RESULTS_AND_FREEZE_RECEIPT_2026-09-11.md`.

**Machine-readable frozen results:** `submission/IRIS_2026_EVIDENCE_BUNDLE_V1/FROZEN_RESULTS.json`.

**End-to-end build checkpoint:** `architecture/PROJECT_BUILD_COMPLETION_2026-09-11.md`.

**Two-week finish / real-world pilot execution plan:** `architecture/TWO_WEEK_FINISH_AND_REAL_WORLD_PILOT_PLAN_2026-09-11.md`.

This entry point is intentionally short. Older modeling, freshness and V3 reliability files remain preserved for audit continuity, but they are not the current scientific centerpiece when they conflict with `CURRENT_STATUS.md`.

## Current project in one sentence

> **Are 24-hour SEP forecasting scores measuring prediction of a new radiation storm, or partly rewarding recognition of a storm that is already active and repeated counting of the same physical event?**

The contribution is an **episode-normalized causal evaluation benchmark**, not a claim of a new state-of-the-art forecasting architecture.

## Current evidence state

`SEP_PRISM_FIXED_MODEL_REPLAY_V1 — STRONG_MODEL_CONFIRMATION_ON_PREVIOUSLY_EXPOSED_PUBLIC_DATA`

The higher-powered preregistered historical replay contains 7,558 chronological score issues. Its matched physical-unit analysis contains 85 onset episodes and 1,080 quiet blocks with 10,000 shared bootstrap draws. All three frozen primary directional TSS intervals are below zero.

This is strong **methodological confirmation on development-exposed historical data**, not untouched prospective validation. The post-`2025-09-10T00:00:00Z` protected outcome pool remains sealed.

## Build state

The project now includes the complete historical benchmark/replay stack, independent verification, competition-facing figures/scaffolds, predictor-side infrastructure for a future sealed prospective confirmation, and a machine-readable final-results contract regression-tested against the submission evidence bundle.

The prospective software can capture timestamped public NOAA/SWPC inputs, create a deterministic pre-issue proton-state prediction, store it in a hash-chained predictor ledger and verify that ledger. A new forecast-validity layer can classify pre-issue records as `VALID`, `DEGRADED` or fail-closed `ABSTAIN` using provenance/missingness facts without reading outcomes. It still **cannot** unseal protected outcomes, set `features_verified_causal=true` by itself, define universal recovery thresholds from development masking tests, or claim prospective skill. Recurring prospective collection is not authorized until the research-plan/approval, independent-custody and execution-freeze gates are satisfied.

The external-facing package is explicitly an **SEP Forecast Validity and Evaluation Kit**, not a replacement operational forecaster. Operator materials are draft/unsent until reviewed.

## Current reading order

### Scientific core

1. `CURRENT_STATUS.md`
2. `architecture/FINAL_TECHNICAL_RESULTS_AND_FREEZE_RECEIPT_2026-09-11.md`
3. `submission/IRIS_2026_EVIDENCE_BUNDLE_V1/FROZEN_RESULTS.json`
4. `architecture/PROJECT_BUILD_COMPLETION_2026-09-11.md`
5. `architecture/TWO_WEEK_FINISH_AND_REAL_WORLD_PILOT_PLAN_2026-09-11.md`
6. `architecture/PROJECT_ARCHITECTURE_2026-09-11.md`
7. `architecture/EPISODE_NORMALIZED_CAUSAL_BENCHMARK_RESULT_2026-09-10.md`
8. `architecture/EVIDENCE_INDEX_2026-09-11.md`
9. `architecture/LITERATURE_NOVELTY_MATRIX_2026-09-11.md`

### Prospective confirmation and validity

10. `architecture/PROSPECTIVE_OPERATIONAL_FEATURE_AVAILABILITY_CONTRACT_2026-09-11.md`
11. `architecture/PROSPECTIVE_PIPELINE_IMPLEMENTATION_2026-09-11.md`
12. `architecture/PROSPECTIVE_CUSTODIAN_HANDOFF_2026-09-11.md`
13. `config/prospective_operational_confirmation_v1_preregistration_2026-09-11.json`
14. `config/prospective_live_input_sources_v1.json`
15. `config/past_proton_active_proxy_rule_v1.json`
16. `config/sep_forecast_validity_contract_v1.json`
17. `src/iris_sep/forecast_validity.py`
18. `config/inspected_evidence_registry_v2.json`

### Competition / student ownership

19. `submission/IRIS_2026_EVIDENCE_BUNDLE_V1/README.md`
20. `architecture/COMPETITION_FINAL_AUDIT_2026-09-11.md`
21. `architecture/IRIS_SUBMISSION_READINESS_2026-09-10.md`
22. `architecture/IRIS_SUBMISSION_PACK_2026-09-10.md`
23. `architecture/POSTER_BLUEPRINT_2026-09-11.md`
24. `architecture/VIDEO_STORYBOARD_2026-09-11.md`
25. `architecture/IRIS_JUDGE_QA_2026-09-10.md`
26. `architecture/STUDENT_OWNERSHIP_DEFENSE_CHECKLIST_2026-09-11.md`
27. `architecture/ISEF_RESEARCH_PLAN_SCAFFOLD_2026-09-11.md`
28. `todo.md`

### External workflow review — draft only

29. `architecture/operator_pilot/SEP_FORECAST_VALIDITY_EVALUATION_KIT_OPERATOR_BRIEF.md`
30. `architecture/operator_pilot/TECHNICAL_VALIDATION_SHEET.md`
31. `architecture/operator_pilot/FIVE_MINUTE_DEMO.md`
32. `architecture/operator_pilot/WORKFLOW_INTERVIEW_REQUEST_DRAFT.md`
33. `architecture/operator_pilot/PILOT_PROPOSAL_DRAFT.md`
34. `architecture/operator_pilot/WORKFLOW_REVIEW_TEMPLATE.md`

## Protected-evidence rule

Development-side work must not query, count, inspect, score, stratify or reveal protected post-2025 labels, event identities, timestamps or model scores. Do not use the protected pool for power rescue, model selection, threshold adjustment, missingness-limit selection or narrative tuning.

## Historical material

The V3 reliability work and the negative freshness-crossover experiment remain valid historical evidence and must stay preserved. They are not to be reopened or reinterpreted to rescue the current result.
