# Novelty audit — operational reproducibility of retrospective SEP forecasts

**Audit date:** 21 September 2026  
**Status:** `BOUNDED_SEARCH_PASS — NO FIRST-EVER CLAIM`

## Candidate contribution

The active project asks whether a modern retrospective, multi-source 24-hour SEP forecasting interface can actually be reconstructed using only measurements and data products that were available before forecast issue time. It then measures the gap between the retrospective interface and an operationally reproducible interface.

The intended contribution is **not** the observation that latency, near-real-time products, or data availability matter. Those points are established. The narrower target is a reproducible, feature-by-feature causal-availability audit plus a quantitative retrospective-to-operational interface/forecast gap for a multi-source SEP machine-learning benchmark.

## Hostile search results

| Prior direction | What prior work already does | Consequence for this project |
|---|---|---|
| SEP onset/persistence verification | Existing SEP and operational literature distinguishes onset, persistence, warning time and event-level verification. | We do not claim the distinction itself as new. |
| Historical proton predictors | Published SEP models use preceding proton flux, persistence-style baselines and sub-threshold proton increases. | Proton-history attribution was abandoned as the main project. |
| Lag / forecastability curves | Ji et al. (ICDMW 2025) vary the gap between observation and SEP onset and analyze performance and feature importance versus lag. | The sensor-blackout / forecastability-horizon candidate was abandoned. |
| Physics-guided counterfactuals | Patil et al. (2026) apply physics-guided counterfactual explanations to SEP prediction. | A counterfactual-physics project was abandoned. |
| Near-real-time versus definitive solar products | SEP papers and reviews acknowledge that definitive SHARP/HARP products and near-real-time products can differ and that near-real-time performance must be validated separately. | This motivates the audit but is not claimed as novel. |
| Forecast input latency | Operational validation frameworks and space-weather literature explicitly require pre-event/real-time inputs and discuss latency. | Latency itself is not the novelty claim. |

## What the bounded search did not find

The search did not identify an SEP study that simultaneously does all of the following for a modern multi-source machine-learning forecast interface:

1. enumerates every predictor used by the retrospective model;
2. requires evidence that an equivalent value existed before the model's nominal issue time;
3. distinguishes measurement time from first-seen/publication time and later definitive reprocessing;
4. classifies every predictor as operationally verified, latency-unverified, schema-mismatched, retrospective-only or without an equivalent;
5. refuses silent substitution of later definitive products for near-real-time products;
6. quantifies the fraction of the retrospective interface that can actually be reproduced operationally; and
7. where sufficient causal data exist, compares the retrospective prediction with a causally reproducible replay or separately frozen causal-only model.

This is an **application/method novelty claim**, not a new physical law or a new machine-learning algorithm.

## Novelty confidence

**Moderate-to-high for this exact application design, conditional on continued literature surveillance.**

Reasons for confidence:

- closest papers discuss operational latency, near-real-time data, or real-time demonstrations, but do not appear to publish the same feature-lineage audit plus quantitative interface-reproducibility gap;
- the repository already exposes a concrete scientific tension: a retrospective predictor interface exists, while the prospective contract blocks that interface as a set until field-level availability is proven;
- the experiment can return a negative result. If almost all predictors are operationally reproducible, that is still an informative outcome.

Reasons not to claim priority:

- terminology varies across space-weather forecasting, software reproducibility, data latency and operational validation;
- a similar audit could exist under a different name or in technical documentation rather than a paper;
- an exhaustive full-text search of every SEP publication has not been completed.

## Claims allowed if the experiment succeeds

Safe form:

> In a bounded literature search, we did not find a previous SEP study that performed the same feature-by-feature issue-time availability audit and quantitative retrospective-to-operational replay for a modern multi-source machine-learning interface. Our study applies that test to this benchmark.

Not allowed:

> This is the first operational reproducibility audit ever performed in space weather.

## Kill conditions

Abandon or materially narrow the novelty claim if a prior publication is found that already performs a comparable feature-by-feature issue-time availability audit **and** quantifies the retrospective-versus-operational replay gap for an SEP forecasting model.

Do not rescue the project by changing terminology if the scientific experiment is already present in prior work.
