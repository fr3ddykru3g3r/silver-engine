# IRIS 2026 submission pack — episode-normalized causal SEP benchmark

**Status:** submission-ready working draft grounded only in the verified development result.  
**Do not include:** school name, city, state, protected post-2025 outcomes, state-of-the-art claims, award claims, or claims that SEPNET's final SEPVAL score is wrong.

## Recommended project title

**Are We Forecasting a New Solar Radiation Storm? An Episode-Normalized Causal Benchmark for SEP Prediction**

Shorter alternative:

**Forecasting New Solar Radiation Storms, Not Just Recognizing Active Ones**

## One-sentence judge hook

**A forecasting system can score well simply because a radiation storm is already happening; we built a benchmark that asks whether it actually warned before the storm began and prevents one long event from being counted repeatedly.**

## Project abstract — 231 words

Solar energetic particle (SEP) forecasting systems are commonly evaluated on fixed 24-hour windows. I investigated whether such window-level scores can mix three different abilities: forecasting a new radiation storm, recognizing that a storm is already active, and repeatedly counting one long physical event. I built an episode-normalized causal benchmark for >10 MeV proton events crossing 10 pfu. It separates already-active persistence from new onset, gives each uniquely mapped physical episode total positive weight one, and uses shared episode-level bootstrap resampling. A public audit of 11,773 windows from a pinned SEP benchmark found 610 uniquely mapped positive windows representing only 256 physical episodes, while 411 reconstructed positive windows were already-active persistence cases and 227 were new-onset cases. I then evaluated fixed climatology, persistence, elastic-net and XGBoost models on an exposed 2014–2017 out-of-fold development cohort. A simple “proton already active” diagnostic scored TSS 0.556 under ordinary occurrence evaluation but TSS 0.000 for new onset. Joint XGBoost changed from TSS 0.657 under standard scoring to 0.443 after episode normalization and 0.043 for new onset. The mapped-episode bootstrap estimated a multiplicity-only TSS shift of -0.133 [95%: -0.225,-0.041] and a persistence-exclusion shift of -0.400 [-0.800,-0.125] for joint XGBoost. The onset arm contained only five events, so model-rank conclusions remain underpowered. These results show that the definition of the evaluation unit can materially change apparent SEP forecasting skill and motivate event-level, onset-specific reporting.

## Introduction & objective — 111 words

Solar energetic particles can raise >10 MeV proton flux above the 10 pfu threshold used for NOAA solar-radiation-storm classification. Machine-learning forecasts are often tested using fixed 24-hour windows: the model observes one window and predicts whether an SEP event appears in the next. But a long physical event can occupy several windows, and some forecast times may occur after the event has already started. This creates a measurement question: is a high score evidence of forecasting a new storm, or partly of recognizing persistence and counting the same event repeatedly? Our objective was to build and test an evaluation framework that separates these effects while keeping the underlying forecasting models fixed.

## Innovation — 54 words

We introduce an episode-normalized causal SEP benchmark. Instead of treating every positive 24-hour window as an independent event, it maps windows to physical SEP episodes, assigns each uniquely mapped episode total positive weight one, and separates already-active persistence from genuine new-onset opportunities. The contribution is therefore an evaluation method, not a new neural-network architecture.

## Methodology — 157 words

First, we audited a pinned public SEP rolling-window benchmark by reconstructing operational >10 MeV, ≥10 pfu episode intervals and classifying each prediction window as new onset, already-active persistence, negative, or ambiguous. We measured how many positive windows represented the same physical episodes. Second, we evaluated fixed comparators—climatology, a current-proton-active diagnostic, elastic-net logistic regression, joint XGBoost, and XRS-only XGBoost—on an exposed 2014–2017 expanding out-of-fold development cohort. No model was retuned after results were seen. We scored each model under ordinary window occurrence, episode-normalized occurrence, and new-onset definitions using True Skill Statistic (TSS). For uncertainty, ambiguous positive windows were excluded only from a separately labelled uniquely mapped sensitivity cohort. We generated 10,000 shared bootstrap draws at the physical-episode level for positives and quiet-block level for negatives, so every paired model/evaluation contrast used identical resamples. An independent verifier then recomputed persisted metrics, weights, bootstrap contrasts, attrition counts and file hashes without importing the benchmark runner.

## Results and conclusions — 125 words

The public audit contained 11,773 windows. Among uniquely mapped positives, 610 windows represented only 256 physical SEP episodes, a multiplicity factor of 2.38; 411 reconstructed positive windows were already-active persistence cases and 227 were new onset. In the internal development cohort, a diagnostic that only recognized current proton activity scored TSS 0.556 under ordinary occurrence scoring but 0.000 for new onset. Joint XGBoost fell from TSS 0.657 to 0.443 after episode normalization and to 0.043 for new onset. Its multiplicity-only TSS shift was -0.133 (95% bootstrap interval -0.225 to -0.041), and persistence exclusion shifted TSS by -0.400 (-0.800 to -0.125). Only five onset episodes were available internally, so rank-change conclusions remain underpowered. The results support reporting physical-event and onset-specific skill alongside conventional window scores.

## Acknowledgement and reference links — 67 words

We used publicly available space-weather benchmark data and NOAA operational definitions, together with open-source scientific Python tools. Key references include NOAA Space Weather Scales; Yu et al. (2026), “Solar Energetic Particle Forecasting With Multi-Task Deep Learning: SEPNET,” DOI 10.1029/2026JH001247; and Yu et al. (2026), “SEP-PRISM Data,” arXiv:2607.16160. All computational claims in this submission are tied to frozen Git commits, workflow receipts and independently verified artifacts.

## 90-second video script

Hi, I’m Kyros Goyal, and our question is simple: **are solar-radiation forecasting systems actually warning us before a storm begins, or can their scores partly reward them for noticing a storm that is already happening?**

Solar energetic particle forecasts are often tested in 24-hour windows. That creates two problems. One long radiation storm can appear in several positive windows, so the same physical event may be counted multiple times. And some forecast times can occur after proton levels have already crossed the storm threshold.

We built an **episode-normalized causal benchmark**. It separates new onset from already-active persistence and gives each uniquely mapped physical storm a total positive weight of one.

In a public benchmark, 610 mapped positive windows represented only 256 physical storms. In our fixed-model development test, a rule that only checks whether proton levels are already high achieved a TSS of 0.556 under ordinary scoring—but exactly zero for forecasting new onset. Joint XGBoost dropped from 0.657 under standard scoring to 0.043 for new onset.

Our onset sample is small, so we do not claim every published model is overrated. Our result is narrower: **how we define the test can materially change what “forecasting skill” actually means.**

## 30-second fallback explanation

Most SEP models are scored in 24-hour windows. But a long storm can be counted several times, and a model may receive credit even when the storm has already started. We built a benchmark that gives each physical storm equal weight and scores only genuine new-onset opportunities separately. A trivial “storm already active” rule scored TSS 0.556 conventionally and 0.000 for onset. So evaluation design can change the apparent forecasting skill.

## What to say when a judge asks “what is actually new?”

The novelty is not that onset and persistence are different. The novelty we tested is **the quantitative effect of separating them while also normalizing repeated 24-hour windows to physical SEP episodes under one reproducible benchmark**. We then measured how fixed models change without changing their predictions after seeing the result.

## What NOT to say

- “We proved SEP forecasting papers are inflated.”
- “SEPNET is wrong.”
- “Our model is better than state of the art.”
- “We solved solar-radiation forecasting.”
- “Our five-event onset sample proves the ranking reversal.”
- “This is operationally certified.”
- “This guarantees ISEF qualification.”

## Evidence receipt to retain outside spoken presentation

Authoritative audit-corrected development artifact:

- workflow run: `34371418428`
- commit: `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact ID: `10112207048`
- artifact SHA-256: `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`
- independent V2 verification: PASS
- final evidence-manifest verification: PASS

Public benchmark audit:

- upstream commit: `d0eb54e46b7dd6c760325e123d2ad86f9420fbff`
- workflow run: `34362893938`
- artifact ID: `10108593291`
- artifact SHA-256: `498104f66efb59ebd85795d628d799006f60266a8e3454bd2df3b127b988673d`
