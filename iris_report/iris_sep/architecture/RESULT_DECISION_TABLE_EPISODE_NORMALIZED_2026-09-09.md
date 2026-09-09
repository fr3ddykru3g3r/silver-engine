# Frozen result-interpretation table — episode-normalized benchmark

**Date:** 2026-09-09

This table is fixed before new real-data scoring so that result language is not selected after seeing the outcome.

| Observed result | Interpretation allowed | Interpretation forbidden |
|---|---|---|
| Large episode-normalization shift with paired CI excluding 0 in >=2 fixed models | Standard window representation materially changes measured skill in the tested cohorts/models | All SEP literature is invalid |
| Model-rank reversal with paired support | Evaluation definition can change which tested model appears best | Our model is universally superior |
| Large shift but CI wide/underpowered | Suggestive development evidence; more independent episodes required | Proven evaluation bias |
| Near-zero shift with narrow CI | Standard window and episode-normalized onset evaluation are empirically similar for tested models/cohort | Episode normalization never matters |
| Near-zero shift with wide CI | Inconclusive / underpowered | No effect exists |
| Proton-blind onset degrades strongly while persistence remains strong | Direct proton history contributes differently across onset/persistence regimes in tested setting | Proton history is leakage in every forecast model |
| Proton-blind and normal onset models similar | Direct proton history is not the dominant mechanism in tested setting | Proton measurements are useless |
| Attrition/causal eligibility removes many standard positives | Conventional occurrence cohort and new-onset cohort answer materially different questions | Prior authors intentionally inflated results |
| No model meets useful skill after onset normalization | Benchmark reveals the scientific difficulty of genuine onset prediction | Project failed / hide result |

## Universal language rule

Always say **“in the tested cohort/models”** unless independent multi-cohort evidence justifies broader generalization.

## Falsification rule

The project succeeds scientifically if it cleanly distinguishes among these outcomes. A competition narrative may be less dramatic under a null result, but the method and result remain fixed. No post-result architecture or threshold search is authorized inside V1.
