# Novelty boundary — episode-normalized causal SEP evaluation

**Date:** 2026-09-09  
**Purpose:** prevent overclaiming while preserving the strongest defensible originality claim.

## What prior work already does

The project must not claim novelty for any of the following by itself:

- using proton or soft-X-ray histories for SEP/SPE forecasting;
- using XGBoost, logistic regression, neural networks, LSTMs, transformers or multimodal inputs;
- predicting >10 MeV, >=10 pfu occurrence;
- forecasting 24 hours ahead;
- distinguishing operational onset from persistence conceptually;
- using event-disjoint or chronological validation in general;
- noting that class imbalance and false alarms are difficult in SEP forecasting.

Recent SEP work already covers all of those themes.

## Specific opening identified in current literature

A 2026 state-of-the-art SEPNET paper explicitly notes that one physical SEP event can persist longer than 24 hours and therefore contribute multiple positive predictor windows. That observation establishes the mechanism but does not, in the literature reviewed for this project as of 2026-09-09, supply the complete evaluation study preregistered here:

1. compare standard window-level occurrence scoring directly with causally new-onset-only scoring;
2. normalize the total positive contribution of every distinct physical SEP episode to one;
3. isolate persistence cases rather than allowing them to contribute to onset skill;
4. test whether model ranking changes under that normalization;
5. use identical episode/quiet-block bootstrap draws for every paired contrast;
6. persist a complete per-issue eligibility/attrition ledger.

This is the candidate novelty. It is a **methodological evaluation framework**, not a new physics law and not a new forecasting architecture.

## Allowed originality statement before results

> We developed a preregistered benchmark to test whether repeated 24-hour windows from the same physical SEP episode and already-active persistence states alter measured forecast skill or model ranking. The benchmark gives each positive physical episode equal total weight, separates new onset from persistence, and uses shared episode-level resampling for paired uncertainty.

## Forbidden originality statements before results

Do not say:

- “existing SEP models are inflated”;
- “we proved current benchmarks are wrong”;
- “first onset forecasting model”;
- “first use of event-based evaluation”;
- “first causal SEP model”;
- “breakthrough”;
- “industry-leading”;
- “state of the art.”

## Result-dependent originality gate

A stronger statement becomes defensible only if the frozen study demonstrates a material, uncertainty-supported shift or model-rank change across at least two fixed nontrivial model families.

If standard and episode-normalized evaluation agree within uncertainty, the originality becomes the negative benchmark result: a rigorous test showing that the suspected evaluation effect is not large in the tested setting.

## Why this is competition-relevant

The scientific value comes from interrogating what a forecast score means. A system can appear strong because it repeatedly recognizes a long-running event, while the operationally difficult question is whether it can warn before a new event begins. The benchmark turns that conceptual distinction into a falsifiable, reproducible measurement problem.
