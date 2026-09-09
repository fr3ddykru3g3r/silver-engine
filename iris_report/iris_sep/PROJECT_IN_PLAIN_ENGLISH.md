# IRIS-SEP — project in plain English

## The project in one sentence

**When a solar-radiation warning system loses an important sensor feed, can it tell when a reduced-sensor forecast is still useful — and when it should refuse to guess?**

That is the project.

The forecast target is fixed: the probability that a **new** >10 MeV solar energetic particle event will cross **10 pfu within the next 24 hours**.

You do not need to know solar physics to understand the problem. Think of it as a rare radiation-storm warning: measurements arrive from several sources, and the system has to decide whether tomorrow is unusually risky.

## Why this matters

A normal forecasting model assumes its inputs are there.

Real sensor systems do not always behave that way. A feed can disappear, become stale, or be unavailable for part of the historical record.

The dangerous failure mode is not necessarily a crash. The model can still output a perfectly normal-looking number — for example, `17% risk` — even though important evidence is missing.

IRIS-SEP asks a different question from ordinary forecasting:

> **When is a forecast probability trustworthy enough to show to a human?**

## The simple idea

Instead of inventing a missing measurement at runtime, we train small forecast combinations in advance for the information that might actually remain available.

There are four information states:

1. **FULL** — all main information families are available.
2. **NO_XRS** — the X-ray information is unavailable.
3. **NO_PROTON** — the proton-context information is unavailable.
4. **NO_XRS_OR_PROTON** — both are unavailable, leaving only the remaining solar information.

When a feed disappears, the system does not retrain itself and does not fabricate a replacement measurement. It switches to the already-frozen state that matches the information actually available.

Then it makes a second decision:

- **NORMAL** — sufficient evidence for the ordinary forecast;
- **DEGRADED** — reduced information, but development evidence says a limited forecast may still be shown with a warning;
- **ABSTAIN** — too much useful evidence has disappeared, so the system should not pretend the probability is decision-worthy.

## What the controlled development experiment found

This is **development-only evidence**, not the untouched final test.

Using event-bearing sensor-loss cases and quiet controls:

| Information missing | Development permission |
|---|---|
| X-ray feed | **DEGRADED** |
| proton-context feed | **DEGRADED** |
| both X-ray + proton-context feeds | **ABSTAIN** |

The important part is the last row.

Even the solar-only fallback still produces finite probabilities. The system therefore cannot use “the model returned a number” as proof that the number is useful. When both major context feeds are removed, detection on the controlled affected cohort falls below the predeclared minimum, so the operator rule refuses the forecast.

That is the central design principle:

> **A forecasting system should know when it no longer has enough evidence to act normal.**

## What about 24-hour, 72-hour and 168-hour outages?

The availability-conditioned fallback uses only the information present at the current forecast issue time. It does not use or reconstruct history from the missing feed.

Therefore, once a feed is absent, the fallback probability depends on the **availability state**, not on whether that feed has already been absent for one, three or seven days. The earlier terminal-outage experiment consequently produced the same endpoint cohort and state prediction at those durations.

Those receipts remain preserved for audit history, but we do **not** present them as three independent scientific findings.

Duration matters for methods such as forward-fill or reconstruction, because those methods depend on how old the last available value is. It does not create extra evidence for a state-only fallback.

## The uncomfortable number we do not hide

On the inspected development score block, the full-input model has useful ranking/discrimination but a very high row-level false-alarm ratio under the frozen binary threshold.

That means a naive story such as “send an alert every time the threshold is crossed” is not good enough.

Instead of hiding this weakness, the project turns it into a second research question:

> **Can the model concentrate rare events into a small, fixed fraction of days that a human analyst could realistically review?**

Before any locked-test access, we froze a primary human-review budget of **5% of eligible forecast days**, about **18 days of review per 365 issue-days**. We will report how many event-positive issue days fall inside that fixed high-risk slice, plus the enrichment over random review at the same workload.

This metric is frozen after development inspection but before the untouched test; we do not pretend it was preregistered before seeing development data.

## Why this is more than “another AI solar-flare model”

The headline is not a new neural network.

The project separates two questions that are often mixed together:

1. **What is the probability of a new radiation event?**
2. **Do the available measurements justify exposing that probability as a normal forecast?**

That second question creates a reliability layer around the model.

The interesting scientific result can therefore be negative:

- if one missing feed still leaves useful evidence, show a **DEGRADED** forecast;
- if too much evidence disappears, **ABSTAIN**;
- if a supposedly clever missing-data method does not improve the final forecast, remove it.

## What is built

The repository now contains:

- a fixed NEW-SEP target and chronological role construction;
- specialist forecast models for different information families;
- a cross-fitted evidence combination;
- calibration and threshold selection separated from fitting;
- availability-conditioned reduced-sensor models;
- explicit NORMAL / DEGRADED / ABSTAIN rules;
- event-bearing outage stress tests;
- independent auditing of saved predictions and gates;
- hash-pinned public development inputs and immutable CI evidence;
- a frozen 5% operator-review-budget policy for the future locked evaluation;
- strict claim boundaries separating development evidence from untouched final evidence.

## What we are NOT claiming

We are not claiming:

- operational certification;
- perfect prediction of solar radiation storms;
- that false alarms have been solved;
- that missing measurements can always be reconstructed;
- a full simulation of the Sun;
- economic savings;
- superiority on an untouched test that has not been opened;
- any competition or award outcome.

## Thirty-second explanation for a judge

“Solar-radiation forecasts depend on several sensor feeds. I found that losing a sensor does not necessarily make a model crash — it can keep giving a convincing-looking probability even when the warning is no longer trustworthy. So we trained separate forecast combinations for the sensors that remain and added a second decision: normal, degraded, or abstain. In development tests, losing either one of two major information feeds still supported a degraded forecast, but losing both failed our detection requirement and forced the system to refuse the forecast. We are now testing whether the risk ranking can concentrate rare events into a fixed 5% of days worth human review, with the final metric frozen before an untouched evaluation.”

## The research question

**Can availability-conditioned forecasting preserve useful 24-hour NEW-SEP risk information when solar sensor feeds are missing, while a frozen reliability rule identifies when the system should degrade or abstain rather than expose an unjustified probability?**

## The rule that keeps the project simple

Every component must answer one of two questions:

> **Does this improve the forecast?**

or

> **Does this make the system better at knowing when not to trust the forecast?**

If the answer to both is no, it does not belong in the final project.
