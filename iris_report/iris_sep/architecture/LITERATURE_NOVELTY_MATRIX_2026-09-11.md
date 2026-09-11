# Recent literature / novelty matrix — 2026-09-11

**Purpose:** student source-checking and oral-defense scaffold. Every cited paper must be personally read/verified by the student researchers before final submission. Do not convert this matrix into a claim that earlier studies are wrong.

## Novelty claim to defend

Established ideas include SEP onset, persistence, event catalogues, TSS/HSS/FAR, chronological testing, machine-learning forecasting and event-based operational validation. The defensible contribution here is narrower:

> **For the same frozen 24-hour SEP predictions and alerts, this project jointly decomposes measured skill into mapped occurrence, physical-episode-normalized occurrence and causal new onset, and quantifies their differences with shared physical-unit uncertainty.**

The empirical contribution is the measured effect size under a pinned public rolling benchmark and a frozen historical replay, not the invention of onset or persistence as concepts.

## Threat matrix

| Work | What it already does | Why it threatens an overbroad novelty claim | Remaining gap addressed here |
|---|---|---|---|
| Yu et al., **SEPNET**, 2025/2026 publication cycle, *JGR: Machine Learning and Computation* | Multi-task deep-learning SEP forecasting using flare/CME/SHARP inputs and comparison with reference methods on SEPVAL | Blocks claims that sophisticated multi-source SEP ML, transformer/LSTM forecasting or high-skill model comparison is new | This project is not an architecture claim; it asks what a daily occurrence score measures when physical-event multiplicity and already-active state are separated while predictions are fixed |
| Yu et al., **SEP-PRISM Data**, 2026 | Builds the 14,464-sample, 24-hour multi-source forecasting table with 650 operational positives | Directly establishes the windowed representation used by the higher-powered audit | This project maps those windows back to physical episodes/onset states and quantifies how fixed-model skill changes under alternative estimands |
| Ali et al., 2024, *ApJS* | Uses GOES proton + soft-X-ray features over solar cycles 22–24; compares SVM/XGBoost, persistence and transfer across cycles | Blocks claims that proton history, XRS-only/simple ML comparisons or persistence baselines are new | The new contribution is not the feature family; it is the physical statistical unit and causal-onset evaluation decomposition |
| Kasapis et al., 2024, *ApJ* | Interpretable ML for SEP forecasting with cross-cycle data and operationally motivated imbalance | Blocks generic “interpretable ML / cross-cycle validation” novelty | The benchmark contribution is orthogonal to architecture interpretability and can be applied to such forecasting systems |
| Papaioannou et al., 2025, *Space Weather* (ASPECS validation) | Rigorous SEP model validation using POD, FAR, PC, HSS and TSS on SEPVAL-related samples | Blocks the claim that careful SEP validation or use of TSS/FAR is itself novel | This project isolates repeated physical-event representation and already-active persistence under fixed forecasts rather than introducing validation metrics generally |
| Kasapis et al., 2026 review | Surveys contemporary ML SEP models, their inputs, datasets, outputs and good-practice issues | Raises the bar for any “first ML benchmark” language and requires accurate positioning against a broad field | The safest novelty is a specific benchmark estimand/dependence problem with quantified effect, not another model family |

## Defense paragraph ingredients

A student-written literature paragraph should make four moves:

1. **Credit the field:** modern SEP work already includes multi-source neural networks, XGBoost/SVM baselines, persistence forecasts, multi-cycle tests and standardized validation metrics.
2. **Identify the measurement gap:** fixed daily target windows can be statistically different from physical SEP episodes, and forecast issue times can occur in an already-active state.
3. **State the contribution precisely:** hold forecasts fixed; report mapped occurrence, per-episode-normalized occurrence and genuine onset as separate estimands; resample physical units rather than daily rows.
4. **Avoid accusation:** the result motivates complementary reporting; it does not by itself prove a published SEPVAL result is invalid.

## Claims explicitly blocked by the literature audit

Do not say:

- “We are the first to use machine learning for SEP forecasting.”
- “We are the first to use proton/XRS inputs.”
- “We invented event-based validation.”
- “We invented onset versus persistence.”
- “TSS has never been used properly before.”
- “Every previous daily SEP result is inflated.”

## Claims still defensible

- The project tests a concrete **combined** evaluation decomposition under one reproducible fixed-prediction framework.
- The public SEP-PRISM audit shows physical-event multiplicity and persistence are quantitatively large enough to matter as benchmark properties.
- The frozen replay shows the choice of estimand materially changes measured skill for some fixed models.
- A benchmark-level contribution can be useful across architectures precisely because it does not depend on inventing another classifier.

## Sources to personally verify before submission

- Yu, Y.; Chen, Y.; Zhao, L.; Whitman, K.; Manchester, W.; Gombosi, T. **Solar Energetic Particle Forecasting With Multi-Task Deep Learning: SEPNET.** *Journal of Geophysical Research: Machine Learning and Computation*.
- Yu, Y.; Chen, Y.; Zhao, L.; Whitman, K.; Manchester, W.; Gombosi, T. **SEP-PRISM Data: A multi-source dataset for solar energetic particle forecasting.** 2026 preprint/data paper.
- Ali, A. et al. **Predicting Solar Proton Events of Solar Cycles 22–24 Using GOES Proton and Soft-X-Ray Flux Features.** *The Astrophysical Journal Supplement Series* 270 (2024).
- Kasapis, S. et al. **Forecasting Solar Energetic Particle Events During Solar Cycles 23 and 24 Using Interpretable Machine Learning.** *The Astrophysical Journal* 974 (2024).
- Papaioannou, A. et al. **Exploring the Validation Results of the Advanced Solar Particle Events Casting System (ASPECS).** *Space Weather* 23 (2025).
- Kasapis, S. et al. **Review of Machine Learning Models for Solar Energetic Particle Prediction.** 2026 review/preprint.

Final bibliography details/DOIs should be checked against the publisher or official record by the student researchers rather than copied blindly from an AI-assisted note.
