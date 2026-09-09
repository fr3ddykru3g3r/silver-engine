# Judge 9/10 upgrade plan

**Date:** 2026-09-09

This file converts the five audit scores into evidence requirements. It is not a self-rating and does not guarantee judge scores.

## Scientific thought — target evidence for ~9/10

Required:

- one falsifiable central hypothesis about evaluation-induced apparent skill;
- explicit null/failure result that is retained;
- physical SEP episode as statistical unit;
- onset eligibility determined before scoring;
- already-active persistence isolated;
- shared paired episode-level bootstrap;
- uncertainty on all decisive deltas;
- protected final cohort not inspected during development;
- at least one sensitivity analysis that was declared before results.

Judge-facing proof: a one-page causal/evaluation diagram plus a table showing how standard and episode-normalized cohorts differ.

## Creativity — target evidence for ~9/10

Required:

- do not sell XGBoost or multimodality as novelty;
- demonstrate a benchmark contribution that changes how SEP forecasts are evaluated;
- quantify Episode Multiplicity Factor for every cohort;
- quantify Episode-Normalization Shift for every fixed model;
- report whether model ranking changes;
- publish the benchmark/evaluator so another model can be tested without retraining IRIS.

A large supported ranking/skill change would make the originality especially strong. A null result can still be creative if the benchmark itself is demonstrably new and reusable.

## Thoroughness — target evidence for ~9/10

Required deliverables:

1. full per-issue prediction table;
2. complete attrition ledger with exact reconciliation;
3. source manifest and hashes;
4. episode catalogue/hash for authorized cohorts;
5. standard + onset + normalized + proton-blind + persistence result tables;
6. fixed comparator set;
7. 10,000 shared bootstrap draws and hash;
8. calibration curves;
9. lead-time analysis;
10. synthetic/property test receipt;
11. negative/null findings;
12. literature comparison table with claim boundaries;
13. reproducibility instructions from clean checkout;
14. independent replay of final result bundle where feasible.

## Skill — target evidence for ~9/10

Required software/reproducibility evidence:

- deterministic episode construction;
- property tests for long-event normalization;
- duplicate/gap/horizon boundary tests;
- one immutable preregistration/config hash;
- prediction/model/environment hashes;
- bootstrap draw tensor generated once and reused;
- no manual spreadsheet scoring;
- machine-readable result JSON/CSV plus plots generated from it;
- independent implementation or black-box replay of the final evaluator where feasible;
- exact CI receipt on the published commit.

## Clarity — target evidence for ~9/10

The project story must fit five sentences:

1. Solar radiation storms can last more than one 24-hour forecasting window.
2. Standard window scoring can therefore count one physical event multiple times, while operationally a new-event warning differs from persistence of one already underway.
3. We built an evaluator that gives each physical event equal total positive weight and separates new onset from persistence.
4. We apply the same frozen models under standard and episode-normalized causal scoring and ask whether measured skill or model ranking changes.
5. The result tells us whether a model is genuinely warning about new storms or partly benefiting from repeated recognition of ongoing ones.

Anything not necessary to those five sentences belongs in methods/appendices, not the first judge explanation.

## Stop conditions

Do not call the project ~9/10 if any of these remain true at submission time:

- prediction-level results cannot be reproduced;
- fewer than two nontrivial fixed model families are evaluated;
- bootstrap conditions are redrawn independently;
- cohort/attrition counts do not reconcile;
- protected outcomes influenced design;
- only three independent positive episodes support the headline result;
- novelty is still described as proton/XRS/XGBoost/multimodality;
- a large result depends on one post-hoc threshold or one cherry-picked delay;
- student authors cannot explain the evaluation logic and failure conditions themselves.

## Current honest rating after redesign, before new real results

The redesign can raise **potential** scientific thought, creativity, skill and clarity substantially, but the project is not yet evidence-complete. Thoroughness and scientific thought reach competition-level only after the evaluator is run on admissible cohorts with complete prediction-level receipts and uncertainty.
