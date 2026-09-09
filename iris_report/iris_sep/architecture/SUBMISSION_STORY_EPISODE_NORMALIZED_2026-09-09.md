# Submission story — episode-normalized causal SEP evaluation

## 15-second version

Solar radiation storms can last for days. A normal 24-hour forecasting benchmark may therefore count the same physical storm several times. We built a benchmark that gives every physical storm equal total weight and separates predicting a new storm from recognizing one already underway, then test whether those choices change measured model skill or model ranking.

## 45-second version

Solar energetic-particle forecasts are often evaluated in fixed time windows. But one physical radiation storm can persist through several windows, so a long event may appear as several positive examples. Operationally, warning that a new event will start is also different from saying an ongoing event will persist. We therefore built an episode-normalized causal evaluation: already-active cases are separated, every distinct new SEP episode receives equal total positive weight, and all models are compared using the same episode-level bootstrap draws. The experiment asks a simple question: does the way we count events change which forecast looks best?

## Central judge question

> **Are we predicting a new radiation storm, or partly getting credit for recognizing the same storm again?**

## What makes the work technically hard

The hard part is not training one more classifier. It is making the evaluation scientifically fair:

- deterministic threshold-crossing episode construction;
- exact 24-hour causal eligibility;
- separating already-active persistence;
- preventing long events from receiving extra positive mass;
- preserving per-issue predictions and attrition reasons;
- using the identical resampled physical episodes for every paired comparison;
- protecting final outcomes from development.

## What result would matter

A strong positive result would show that episode normalization or onset-only eligibility materially changes skill or reverses model ranking across multiple fixed model families. A strong null result would show that conventional scoring is robust to the suspected effect in the tested setting. Both outcomes answer the preregistered question.

## What not to say

Do not describe this as a new XGBoost model, a new solar simulation, a new onset concept, or proof that published work is wrong. Those are not the contribution.
