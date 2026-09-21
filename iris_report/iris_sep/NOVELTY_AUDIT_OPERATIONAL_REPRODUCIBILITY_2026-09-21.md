# Novelty audit — operational reproducibility of retrospective SEP forecasts

**Audit date:** 21 September 2026  
**Status:** `BOUNDED_SEARCH_PASS — MODERATE CONFIDENCE — NO FIRST-EVER CLAIM`

## Candidate contribution

The active project asks whether a modern retrospective, multi-source 24-hour SEP forecasting interface can actually be reconstructed using only measurements and data products that were available before forecast issue time.

The contribution is **not** the observation that latency, near-real-time products, data gaps, or causal forecast cutoffs matter. Those are established. The narrower contribution is the combination of:

1. a cryptographically pinned 259-predictor retrospective interface;
2. feature-family lineage reconstructed from the exact upstream preprocessing code;
3. a fail-closed issue-time equivalence rubric that distinguishes physical-source existence from equality of the archived feature value;
4. a quantitative interface-reproducibility audit; and
5. a preregistered gate that prevents forecast-skill evaluation when an identical causal input interface cannot be reconstructed.

## Hostile search: nearby work that limits our claim

| Prior direction | What existing work already establishes | Consequence |
|---|---|---|
| Operational forecast cutoffs | NASA CCMC explicitly defines pre-event timestamps and treats forecasts using later information as scientifically evaluable but not operational-type forecasts. | Causality at issue time is not new. |
| NRT versus definitive SHARP | Bobra et al. (2014) document NRT/definitive SHARP differences, including different HARPNUMs and definitive products created weeks later using complete region history. | Product mismatch is not new. |
| Real-time versus historical SEP streams | HESPERIA documents that the same forecasting algorithm behaves differently on historical versus real-time streams because real-time data contain more gaps. | Real-time/historical gaps are not new. |
| Explicit latency approximations | Recent SEP forecasting studies account for practical source delays or discuss latency of flare/CME/coronagraph inputs. | “Latency matters” is not new. |
| SEPNET / SEPNET-PRISM operational framing | Recent multi-source SEP work develops real-time/operational models and discusses NRT inputs and operational deployment. | We cannot claim to invent operational SEP validation. |
| Data-gap stress testing | HESPERIA REleASE+ explicitly studies how inserted gaps affect real-time detection. | Missing-data robustness is not our novelty. |
| Forecastability / lag curves | Prior work varies observation-to-onset lag and examines performance versus lag. | Our abandoned blackout/forecastability-horizon idea was not sufficiently novel. |
| Physics-guided counterfactuals | Existing SEP work uses physics-guided counterfactual explanations. | Our abandoned counterfactual direction was not sufficiently novel. |

## What the bounded search did not find

Across targeted searches of SEP forecasting, NRT versus definitive data products, operational validation, data gaps and latency, we did not identify a prior SEP study that performs this same **end-to-end exact-interface audit**:

- pin one modern multi-source model's complete predictor list;
- trace every feature family through the actual preprocessing implementation;
- distinguish measurement time, first-seen time, definitive reprocessing and statistically reconstructed historical values;
- classify exact feature equivalence fail-closed;
- quantify the fraction of the frozen interface that remains exactly issue-time reproducible; and
- use that result as a preregistered gate before any operational skill score is permitted.

This is an **application/method contribution**. It is not a new physical law, a new machine-learning algorithm, or a universal theorem about all SEP models.

## Why the present result strengthens the novelty case

The audit is not a generic checklist applied to hypothetical features. The frozen interface produces a falsifiable quantitative outcome: 259 exact predictors audited; 248 (95.75%) tied directly to retrospective-only or schema-mismatched constructions; 11 (4.25%) unresolved for historical issue-time latency; 0 currently satisfying the complete exact-equivalence rule.

The important methodological choice is that the study **stops** rather than manufacturing an operational replay by zero-filling, replacing definitive products with similarly named NRT fields, or retraining an after-the-fact reduced model.

## Closest conceptual collisions

**NASA CCMC SEP validation challenge.** CCMC explicitly specifies a pre-event time after which data may not be used for an operational-type forecast. Our work does not improve or replace that rule. It asks whether every archived predictor in a specific modern ML interface can be shown to satisfy such a rule. Source: https://ccmc.gsfc.nasa.gov/challenges/sep/

**HESPERIA real-time versus historical forecasting.** HESPERIA preserves both real-time and retrospective forecasts and notes that historical data contain fewer gaps. This establishes that archival and real-time input streams can differ operationally. It does not, in the material found, pin and classify every feature of the SEPNET-PRISM-style interface under the same preprocessing-lineage audit. Source: https://hesperia.astro.noa.gr/data-retrieval-tool/

**NRT versus definitive SHARP.** Bobra et al. establish that definitive SHARP geometry uses full active-region history and is produced later, while NRT SHARP is available quickly but not identical. Our study uses this as evidence for a schema mismatch; the novelty is not the SHARP fact itself. DOI `10.1007/s11207-014-0529-3`.

**Recent multi-source SEP forecasting.** Yu et al. (2026) introduce SEPNET-PRISM and the multi-source 24-hour forecasting framework audited here. Other recent SEP models also account for or discuss input delays. We found no exact duplicate of the feature-by-feature frozen-interface reproducibility test. DOI `10.1038/s41598-026-66110-2`.

## Novelty confidence

**Moderate**, not absolute. The experiment is narrower than generic real-time validation and precisely reproducible, but terminology varies across space-weather operations, data engineering, forecast verification and reproducibility, and a similar audit could exist in technical documentation not indexed under the same vocabulary.

## Defensible novelty statement

> In a bounded adversarial literature search, we did not find a previous SEP study that pins a complete modern multi-source predictor interface, reconstructs its preprocessing lineage feature family by feature family, and quantitatively tests exact issue-time reproducibility under a fail-closed rule before allowing an operational skill evaluation. We therefore treat the contribution as a new application/method combination rather than claiming a universal first.

## Forbidden wording

Do not write: “the first operational reproducibility audit in space weather”; “we proved real-time SEP models use future data”; “95.75% of the sensor measurements were unavailable”; or “SEPNET-PRISM is invalid.”

## Kill condition

If a prior SEP publication is found that already performs a comparable complete-interface lineage audit **and** quantifies exact issue-time reproducibility for a multi-source forecasting model under an equivalent causal gate, materially narrow or abandon the novelty claim rather than renaming the same experiment.
