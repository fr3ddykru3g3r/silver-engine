# Judge-facing figure plan — episode-normalized benchmark

**Date:** 2026-09-09

The final presentation should use a small number of decisive figures. Do not bury the research question under the historical V3 architecture.

## Figure 1 — The evaluation problem

One horizontal timeline with two example physical SEP episodes:

- Event A lasts ~72 h and intersects three 24-hour outcome windows.
- Event B lasts ~24 h and intersects one 24-hour outcome window.

Top row: conventional window scoring -> A contributes three positive windows; B contributes one.

Bottom row: episode-normalized onset scoring -> A total positive weight = 1; B total positive weight = 1.

Caption question:

> Should one long radiation storm count three times while a short storm counts once?

## Figure 2 — Onset versus persistence state machine

At each issue time:

`PROTON STATE VALID?`

- gap/ambiguous -> unresolved
- already >=10 pfu -> persistence diagnostic only
- below 10 pfu -> eligible for new-onset evaluation

Then check the subsequent 24-hour mature window for a first qualifying crossing.

Make visually explicit that already-active cases cannot increase onset skill.

## Figure 3 — Main scientific result

For every fixed model, show paired standard-window TSS versus episode-normalized-onset TSS with 95% shared-bootstrap intervals.

If model ranking changes, connect the model points to make the reversal immediately visible.

If no ranking changes, show the null result clearly rather than forcing a dramatic chart.

## Figure 4 — Multiplicity and effective evidence

Show:

- positive prediction windows;
- distinct physical SEP episodes;
- Episode Multiplicity Factor;
- number of onset-eligible episodes;
- number of persistence-only windows;
- attrition counts by reason.

This figure exists to prevent a judge from mistaking window count for independent-event count.

## Figure 5 — Mechanism test

Compare fixed XGBoost/elastic-net performance under:

- normal onset-eligible features;
- proton-state-blind onset features;
- persistence diagnostic.

Purpose: determine whether direct particle-history information is disproportionately useful for recognizing persistence versus anticipating new onset.

## Figure 6 — Reproducibility receipt

A compact pipeline:

`SOURCE HASHES -> FROZEN EPISODES -> ATTRITION LEDGER -> PREDICTION FILE -> SHARED BOOTSTRAP DRAW HASH -> RESULT TABLE HASH`

Show CI/replay receipt below it.

## What not to show early

Do not lead with:

- 259-feature V3 architecture;
- detailed source-family diagrams;
- every historical failed filter;
- long lists of model hyperparameters;
- every development result ever produced.

Those belong in appendix/supporting material. The first five minutes must answer one scientific question only.
