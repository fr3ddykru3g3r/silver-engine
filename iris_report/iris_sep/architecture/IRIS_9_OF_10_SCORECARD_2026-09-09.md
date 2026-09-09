# IRIS-SEP 9/10 upgrade scorecard

**Purpose:** convert the external 6/3/4/5/6 audit into objective promotion gates.  
**Rule:** this document is a target, not a self-awarded score. A criterion is not treated as `9/10-ready` until every required evidence item below exists and survives independent review.

## Target project sentence

> **When should an AI space-weather forecast refuse to answer?** We test whether a label-free gate based on source provenance, freshness and scientific equivalence can detect under-supported solar-particle forecasts better than the model's own confidence.

This replaces the judge-facing emphasis on “another SEP predictor.”

---

## 1. Scientific thought — target 9/10

### What held the project near 6/10

- tiny positive-event counts were allowed to carry too much interpretive weight;
- freshness eligibility used unavailable proton information under simulated delay;
- bootstrap draws were not shared across delay comparisons;
- some historical preprocessing could not be promoted to forecast-time causal evidence;
- multiple branches accumulated before the core scientific question was stable.

### Evidence required for 9/10-ready status

- [x] failed freshness branch preserved as negative evidence, not tuned after result;
- [x] protected post-2025 cohort explicitly locked from design-side access;
- [x] new question preregistered before any protected result;
- [x] selector prohibited from labels/future measurements;
- [x] outage eligibility must be computable from information available under the outage;
- [x] single shared cluster-bootstrap table required across every primary contrast;
- [x] explicit falsification conditions;
- [ ] complete power/precision table for every primary result cell, including positive-event count and interval width;
- [ ] independent recomputation of every primary metric from prediction-level artifacts;
- [ ] at least two distinct degradation mechanisms and two frozen model outputs showing a consistent direction before any generalization claim;
- [ ] if the strong effect is absent, conclusion rewritten around the null result rather than rescued with tuning.

**Promotion condition:** all boxes checked and an independent reviewer can reproduce the logic without access to hidden assumptions.

---

## 2. Creativity — target 9/10

### What held the project near 3/10

Proton/XRS features, XGBoost, multimodal SEP forecasting and abstention all have close prior art. Architecture complexity is not novelty.

### New contribution boundary

The candidate contribution is:

> **Source-integrity selective forecasting:** decide whether to expose a prediction from the scientific validity of its inputs, independently of the model's own confidence.

New measurable object:

`integrity_confidence_discordance = forecast_confidence - evidence_integrity`.

This asks whether a forecast can remain highly confident while its evidential support collapses.

### Evidence required for 9/10-ready status

- [x] literature boundary explicitly states what is not novel;
- [x] candidate contribution is separable from the forecasting architecture;
- [x] candidate can be falsified against confidence-based abstention;
- [x] model-output-independent integrity score implemented;
- [ ] structured literature matrix covering SEP ML, selective prediction, missing-data forecasting, source-product inconsistency, and forecast-time causality;
- [ ] search finds no close prior implementation of the same provenance-driven SEP integrity selector; if one exists, narrow the claim again;
- [ ] demonstrate at least one high-confidence/low-integrity case from a real source/interface failure, not only synthetic deletion;
- [ ] demonstrate whether the phenomenon transfers across at least two frozen model families.

**Promotion condition:** novelty rests on a new scientific question + testable mechanism, not on branding or model size.

---

## 3. Thoroughness — target 9/10

### What held the project near 4/10

The audited freshness artifact omitted per-issue predictions and fitted models, preventing reproduction of AUPRC, paired p-values and reported bootstrap intervals.

### Mandatory evidence package

- [x] prediction-level evidence is mandatory in the new preregistration;
- [x] source-level integrity evidence is mandatory;
- [x] full attrition ledger is mandatory;
- [x] shared bootstrap receipt/hash is mandatory;
- [x] negative and inconclusive result preservation is mandatory;
- [ ] `per_issue_predictions.csv` emitted for every actual run;
- [ ] `per_issue_source_integrity.csv` emitted for every actual run;
- [ ] `attrition_ledger.json` with input -> exclusion -> role -> scenario counts;
- [ ] frozen model and decision-threshold hashes;
- [ ] environment manifest with Python/package versions;
- [ ] independent audit script that recomputes all result tables without importing the result runner;
- [ ] artifact SHA-256 inventory and README mapping every figure/table to raw evidence;
- [ ] every score in the paper can be traced to a row in the immutable result package.

**Promotion condition:** a skeptical reviewer can reproduce the paper figures and statistics from saved files without asking the authors for missing information.

---

## 4. Skill — target 9/10

### What held the project near 5/10

The engineering was substantial, but scientific mistakes in eligibility/statistics and incomplete result artifacts limited what the implementation proved.

### Evidence required for 9/10-ready status

- [x] deterministic source-integrity library implemented;
- [x] hard causality/schema failures implemented as non-bypassable blockers;
- [x] confidence-vs-integrity discordance implemented;
- [x] threshold-role/score-role selective evaluation implemented;
- [x] deterministic random control implemented;
- [x] shared cluster-bootstrap implementation added;
- [x] local source tests passed before publication;
- [ ] GitHub CI green on exact published head in both supported pandas lanes;
- [ ] data-bound integration test on an immutable existing outage artifact;
- [ ] independent audit implementation reaches exact agreement on primary result tables;
- [ ] complete command/hash receipt for the actual study run;
- [ ] student authors can explain every equation, design choice, failure mode, and statistical limitation without relying on generated prose.

**Promotion condition:** technical sophistication is paired with correct scientific semantics and independent verification.

---

## 5. Clarity — target 9/10

### What held the project near 6/10

The repo accumulated model versions, outage branches, freshness work, prospective-source work and false-alarm filters. A judge could easily mistake the history for the contribution.

### Required judge-facing structure

1. **Problem:** AI can stay confident when required scientific evidence becomes invalid.
2. **Question:** Can evidence integrity decide when a forecast should be withheld better than confidence alone?
3. **Method:** freeze forecasts; corrupt/observe source evidence; compare integrity-vs-confidence selectors at matched coverage.
4. **Result:** one primary risk-coverage figure + one confidence-vs-integrity figure.
5. **Live case:** exact source-interface failure where the system correctly refused to forecast.
6. **Limitation:** retrospective development evidence is not independent prospective superiority evidence.

### Evidence required for 9/10-ready status

- [x] one-sentence research question frozen;
- [x] old freshness route explicitly closed;
- [x] new design document separates contribution from history;
- [ ] one-page judge brief with no model-version archaeology;
- [ ] maximum three primary figures in the oral explanation;
- [ ] 90-second explanation understandable without ML terminology beyond “confidence” and “data quality”;
- [ ] every technical term on poster defined in one line;
- [ ] paper abstract states the result, uncertainty and limitation without promotional language;
- [ ] mock judge Q&A passes: student can answer “what is new?”, “what failed?”, “why should I trust this?”, and “what would falsify it?” in under 30 seconds each.

**Promotion condition:** a non-specialist judge can state the question, novelty and primary result correctly after one explanation.

---

# Overall 9/10 gate

The project is **not** 9/10-ready merely because the new code exists.

Treat all five criteria as 9/10-ready only when:

1. the actual Forecast Integrity Gate experiment has a complete immutable prediction-level evidence package;
2. the primary comparison survives shared-cluster uncertainty analysis or yields a clean falsifying/null conclusion;
3. novelty survives the structured literature matrix;
4. an independent implementation reproduces the metrics;
5. the student authors can defend the work themselves;
6. the final paper/poster/video tell one question rather than the repo's development history.

## Current immediate sequence

1. Build the long-format study table **only from already-inspected evidence**.
2. Recover immutable prediction-level artifacts; if unavailable, rerun only already-authorized retrospective experiments to emit them—do not change models or thresholds.
3. Build the source-integrity rows from outage/source receipts without using outcomes.
4. Run `IRIS_FORECAST_INTEGRITY_GATE_V1` exactly once under the frozen contract.
5. Independently recompute every result.
6. Complete the literature matrix.
7. Only then write the judge-facing paper/figures around the result actually obtained.

The purpose is not to manufacture a 9/10 score. It is to make the evidence package strong enough that a harsh reviewer has concrete reasons to give one.
