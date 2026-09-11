# 90-second IRIS video storyboard — 2026-09-11

**Use:** student rehearsal scaffold, not a final generated script. The student researchers should speak naturally in their own words and keep the final recording within the current IRIS time limit.

## 0–10 s — the question

Visual: title + a simple radiation-storm icon / solar image.

Student must communicate:
- SEP forecasts are often scored on daily 24-hour windows;
- the project asks whether that score means “warned before a new storm” or can partly mean “recognized a storm already happening.”

Do not start with ML architecture.

## 10–27 s — the physical mechanism

Visual: `episode_benchmark_mechanism_2026-09-10.svg`.

Student must explain:
- one physical SEP episode may last across multiple daily windows;
- every such window can otherwise count as a full positive;
- some windows begin after the 10 pfu threshold has already been crossed.

Judge takeaway: repeated representation and persistence are physically distinct from new onset.

## 27–44 s — what we built

Visual: three labeled boxes.

1. mapped occurrence;
2. episode-normalized occurrence;
3. new-onset causal evaluation.

Student must say that predictions and alert thresholds were **kept fixed**. Only the evaluation question changes.

For episode normalization, one physical episode contributes total positive weight one even if it generates multiple positive windows.

## 44–57 s — dataset finding

Visual: large numbers, no dense table.

- `14,464` daily windows;
- `614` uniquely mapped positive windows;
- `257` represented physical episodes;
- `EMF = 2.389`;
- `418` persistence windows vs `228` onset windows.

Do not say that 228 equals 228 independent onset episodes.

## 57–76 s — fixed-model replay result

Visual: `sep_prism_fixed_model_tss_2026-09-10.svg`.

Student should point to the joint-XGBoost line:

`TSS 0.726 -> 0.621 -> 0.437`

Then communicate that the matched analysis contains `85` onset episodes and `1,080` quiet blocks, and that all three frozen primary paired intervals were below zero.

Do not call this an operational model result.

## 76–86 s — scientific boundary

Visual: small “What we claim / What we do not claim” split.

Claim:
- evaluation design materially changes measured skill in the frozen historical replay.

Do not claim:
- SEPNET is wrong;
- all prior papers are biased;
- state-of-the-art forecasting;
- operational deployment;
- untouched prospective confirmation.

State that the public historical data were already development-exposed and that protected future evidence remains sealed.

## 86–90 s — takeaway

End on the benchmark contribution, not the model:

When a paper claims to forecast **new** SEP storms, conventional window scores should be accompanied by episode-normalized and onset-specific reporting so judges/readers know what the score actually measures.

## Recording checks

- target 82–88 seconds in rehearsal so the final recording has margin;
- one speaker can explain the full causal chain without reading;
- no school, city or state identifier if the current IRIS submission rule still requires anonymity;
- use the same denominator language as the poster;
- pronounce `pfu` once as “proton flux units”; after that, abbreviation is fine;
- use “false-alarm ratio” rather than “false-positive rate” for the 88.95% alert-precision quantity if it is mentioned at all;
- keep citations/credits in the end frame rather than consuming spoken time.
