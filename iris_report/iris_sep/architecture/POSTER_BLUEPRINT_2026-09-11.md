# Judge-facing poster blueprint — 2026-09-11

**Purpose:** layout and evidence scaffold for the student researchers to rewrite in their own words. This is not final ISEF/IRIS poster prose.

## Poster thesis

The whole poster should answer one question:

**When a daily SEP system receives a high forecasting score, how much of that score represents warning before a new radiation storm, and how much can come from repeated windows or an event that is already active?**

Do not center the poster on XGBoost architecture. The model is a measurement probe; the benchmark is the contribution.

## Three-column layout

### Left column — the physical measurement problem

**Panel A: Why this matters**
- SEP radiation storms can affect spacecraft, aviation and crewed missions.
- Operational threshold used here: >10 MeV proton flux crossing 10 pfu.
- Daily supervised forecasts often ask whether the next 24 hours contain a positive occurrence.

**Panel B: One storm, several labels**
Use `figures/episode_benchmark_mechanism_2026-09-10.svg` or the updated graphical abstract.

Judge takeaway:
- a long event can overlap several daily target windows;
- some issue times occur after threshold crossing;
- these are legitimate operational states but are not identical to pre-onset prediction.

**Panel C: three evaluation questions**
Use a compact visual/table:

| View | Scientific question | Positive weighting |
|---|---|---|
| Mapped occurrence | Is the target present in the following window? | every mapped window full weight |
| Episode-normalized occurrence | How well do alerts generalize across physical storms? | each mapped storm totals weight 1 |
| New onset | Did we warn before a new threshold crossing? | already-active persistence excluded |

Student must be able to draw these three definitions from memory.

### Center column — experiment and evidence

**Panel D: public benchmark audit**
Headline numbers only:
- 14,464 daily windows;
- 650 stored positives;
- zero stored-vs-reconstructed target mismatches;
- 614 uniquely mapped positive windows;
- 257 represented physical episodes;
- EMF = 2.389;
- 418 persistence windows;
- 228 onset windows;
- four onset-state ambiguous positives;
- 36 separate multi-episode-overlap positives.

Do not confuse 228 onset windows with the 85 matched onset episodes used for replay inference.

**Panel E: frozen replay design**
Diagram:

`chronological fit block -> preceding threshold block -> held-forward score block`

Repeat across the three frozen OOF periods. Under the diagram state:
- six fixed comparators;
- same persisted predictions and thresholds under all evaluation definitions;
- 7,558 unique scored issues;
- matched inference = 85 onset episodes + 1,080 quiet blocks;
- 10,000 shared physical-unit bootstrap draws.

**Panel F: strongest figure**
Use `figures/sep_prism_fixed_model_tss_2026-09-10.svg`.

Large callout for joint XGBoost on the matched population:

`0.726 -> 0.621 -> 0.437 TSS`

Caption must explicitly say **same fixed alerts**; the model was not retrained across the three evaluation views.

### Right column — inference, limits and contribution

**Panel G: preregistered contrasts**
Use a compact three-row interval table:

| Frozen contrast | Point change | Paired 95% interval |
|---|---:|---:|
| XGB episode-normalized - mapped | -0.105 | [-0.146,-0.063] |
| XGB onset - episode-normalized | -0.184 | [-0.247,-0.125] |
| proton-state proxy onset - mapped | -0.508 | [-0.567,-0.444] |

One sentence below: all three intervals were below zero under the frozen replay rule.

**Panel H: mechanism control**
Explain the deliberately trivial proton-state comparator:
- it only asks whether proton flux is already above the storm threshold;
- therefore it can recognize persistence but cannot create genuine warning before onset;
- its large occurrence-to-onset change makes the distinction intuitive.

**Panel I: limitations — keep visible**
Five short boxes:
1. public historical files were already development-exposed;
2. publication-time availability is not proven for all 259 historical predictors;
3. the historical replay does not have a universal temporal purge at every role boundary;
4. onset eligibility is retrospectively catalogue-defined;
5. model objects and threshold-block probabilities are not all preserved for full fit-to-score replay.

Do not shrink this panel. Judges should see that the project knows its evidence boundary.

**Panel J: contribution / next falsification**
Supported contribution:
- report conventional occurrence performance;
- also separate already-active persistence;
- report genuine onset performance;
- add physical-event normalization when the scientific claim concerns generalization across storms.

Prospective extension status:
- source/prediction infrastructure implemented;
- outcomes still sealed;
- no prospective claim yet.

## What should NOT appear as a headline

- “We proved previous SEP papers are inflated.”
- “SEPNET is wrong.”
- “Our XGBoost is state of the art.”
- “88.95% false-positive rate.” The 88.95% value is false-alarm ratio; the corresponding false-positive rate is about 4.51%.
- guaranteed IRIS/ISEF outcome.

## Visual hierarchy

At a distance, a judge should be able to read only five things and still understand the project:

1. the research question;
2. `614 windows -> 257 storms`;
3. `same fixed alerts`;
4. `0.726 -> 0.621 -> 0.437`;
5. `historical methodological confirmation; prospective outcomes sealed`.

## Student print test

Before export:
- test at actual poster scale;
- stand one arm-length away from each figure;
- verify every axis label and interval is readable;
- verify school/city/state identifiers are absent wherever IRIS requires anonymity;
- verify every number against the evidence index rather than copying from memory.
