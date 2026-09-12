# IRIS-SEP end-to-end build completion — 2026-09-11

## Disposition

`HISTORICAL_RESEARCH_STACK_COMPLETE — COMPETITION_PACKAGE_COMPLETE_AS_SCAFFOLD — PROSPECTIVE_INFRASTRUCTURE_IMPLEMENTED — PROTECTED_OUTCOMES SEALED`

This file answers one practical question: **what is actually built now, and what cannot legitimately be completed by additional coding on the current exposed data?**

## 1. Scientific core — COMPLETE

Built and frozen:

- physical SEP episode reconstruction;
- issue-time onset/persistence classification;
- ambiguity handling;
- mapped occurrence evaluation;
- per-episode-normalized occurrence evaluation;
- causal new-onset evaluation;
- fixed-model chronological replay;
- physical-unit matched inference;
- shared bootstrap draws;
- model-free public benchmark audit;
- external public-data audit arm;
- historical internal development arm;
- preserved negative freshness experiment;
- preserved V3 reliability/fail-closed evidence.

Current strongest historical evidence:

- 14,464 daily SEP-PRISM windows;
- 650 stored positives with zero reconstruction mismatches;
- 614 uniquely mapped positive windows representing 257 physical episodes;
- EMF 2.389;
- 418 persistence windows and 228 onset windows in the full table;
- 7,558 fixed-replay issue times;
- 85 matched onset episodes + 1,080 quiet blocks;
- 10,000 shared physical-unit draws;
- joint-XGB matched TSS `0.726 -> 0.621 -> 0.437`;
- all three frozen primary directional intervals below zero.

## 2. Computational reproducibility — COMPLETE

Built:

- pinned source hashes;
- frozen configuration contracts;
- persisted per-issue/model predictions;
- persisted thresholds and alerts;
- matched sensitivity population;
- exact shared bootstrap tensors;
- evidence hash manifests;
- supplied replay verifier;
- separately implemented independent reconstruction;
- minimal primary-contrast recheck independent of benchmark scoring functions;
- source-only CI across registered environments;
- pandas 2.x / 3.x compatibility checks;
- explicit inventory of data-bound tests excluded from source-only CI.

The current project can defend **computational reproducibility** of the frozen replay while still correctly saying the scientific cohort is historical/development-exposed.

## 3. Statistical design — COMPLETE FOR THE HISTORICAL CLAIM

Primary inference is appropriately aligned to the dependence structure:

- positive resampling unit = physical onset episode;
- negative resampling unit = seven-day quiet block;
- the same unit draws are reused across compared estimands;
- the result is reported as paired bootstrap intervals rather than pretending repeated daily rows are independent.

No extra ANOVA/p-value layer is required to make the current result scientifically stronger; adding row-level parametric tests would conflict with the dependence problem under study.

## 4. Novelty defense — COMPLETE

Built:

- recent-literature novelty matrix;
- narrow novelty claim;
- explicit list of established concepts not claimed as inventions;
- defense against overclaiming SEPNET/SEPVAL invalidity;
- positioning as an evaluation/measurement contribution rather than another architecture race.

Defensible novelty:

**the combined, quantitative, fixed-prediction decomposition of daily SEP forecast skill into mapped occurrence, physical-episode-normalized occurrence and causal new onset with shared physical-unit uncertainty.**

## 5. Prospective confirmation protocol — DESIGN COMPLETE

Frozen:

- study ID and protected pool;
- 00:00 UTC issue clock;
- >10 MeV, >=10 pfu onset target;
- persistence exclusion rule;
- information floor;
- bootstrap seed/count;
- mandatory primary prospective contrast;
- fail-closed causal-feature gate;
- independent custodian boundary;
- aggregate-only outcome-side evaluator;
- no-rescue disposition rules.

## 6. Prospective predictor infrastructure — IMPLEMENTED

Built on 2026-09-11:

- `config/prospective_live_input_sources_v1.json`;
- `config/past_proton_active_proxy_rule_v1.json`;
- public predictor snapshotter with raw-byte retention, retrieval times and SHA-256;
- strict >=10 MeV proton selection and staleness gate;
- no interpolation/future fill/alternate-satellite substitution;
- deterministic proxy prediction builder;
- rule/schema/source/snapshot hashing;
- append-only hash-chained predictor ledger;
- independent ledger verifier;
- source-only synthetic tests for future-value rejection, staleness, ambiguity and tampering;
- manual-only predictor-capture GitHub workflow with no outcome step.

This completes the **software** required to begin an approved prospective predictor record. It does not authorize data collection or retroactively create genuine forecasts for past issue times.

## 7. Competition-facing scientific package — BUILT

Repository now contains:

- working research-paper draft;
- submission evidence scaffold;
- research-plan scaffold;
- current compliance/forms matrix;
- competition audit;
- submission-readiness audit;
- project architecture map;
- literature/novelty matrix;
- poster blueprint;
- 90-second video storyboard;
- skeptical judge Q&A;
- student-ownership mastery checklist;
- read-only evidence index;
- updated todo/checklist;
- judge-facing figures and plot generators.

Generated prose remains a scaffold for student checking/rewrite where current ISEF/affiliated-fair rules require student-authored final text.

## 8. Claim discipline — COMPLETE

Hard red lines are now repeated across status, paper, submission material and judge prep.

The project does **not** claim:

- all prior SEP studies are biased;
- SEPNET's final SEPVAL result is wrong;
- historical evidence is an untouched final cohort;
- joint XGBoost or proton history is universally superior/inferior;
- operational readiness;
- deployment certification;
- economic savings;
- guaranteed IRIS/ISEF placement.

## 9. What cannot be finished by more coding today

These are now the real remaining gates:

### Student ownership

The student researchers must be able to explain the work themselves, verify citations and write/approve the final competition-facing wording required by the applicable rules.

### Administrative approval

Universal fair/ISEF forms and any required SRC approval must be completed with truthful dates. A prospective extension intended to count toward the same project must be included in the approved plan before it starts where required.

### Genuine prospective evidence

A genuine forecast must exist **before** its target window occurs. Past protected issue times cannot be retrospectively converted into prospective predictions. Recurring future predictor collection therefore needs to start only after approval and execution freeze.

### Independent custody

The development side cannot be the party that opens protected labels, checks event counts for power, changes thresholds and then calls the same cohort independent. The custodian step is a governance requirement, not missing code.

### Information accrual

The frozen prospective study requires at least 50 distinct onset episodes and 500 quiet blocks. SEP events are rare. The study may legitimately return insufficient confirmatory information; the software cannot manufacture independent events.

## 10. Immediate finalization path

Before IRIS submission:

1. student completes the ownership checklist;
2. student verifies each retained citation against the original paper/publisher;
3. student writes/checks final portal/paper/poster wording;
4. complete fair/ISEF forms and disclose support accurately;
5. print-test the two headline figures;
6. regenerate the graphical abstract from the updated plotting code if using it;
7. rehearse the 90-second video and judge defense;
8. archive the frozen replay artifacts outside temporary CI retention;
9. keep protected outcomes sealed;
10. do not reopen model tuning on the historical replay.

After approval for the prospective extension:

1. collect genuine pre-issue predictor receipts;
2. freeze exact execution hashes;
3. begin append-only predictions;
4. transfer outcome authority to the independent custodian;
5. evaluate only under the frozen contract;
6. accept the result without post-hoc rescue.

## Bottom line

The current project is no longer missing a scientific benchmark, a statistical framework, a reproducibility stack, a competition narrative, or prospective software infrastructure.

The main remaining weaknesses are exactly the ones that **should not** be solved by more development-side analysis on exposed data: scientific independence, genuine issue-time prospective evidence, administrative approval and student ownership.
