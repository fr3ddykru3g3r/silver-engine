# Literature gap check — episode-normalized SEP evaluation

**Date checked:** 2026-09-09

This note records the literature boundary used to choose the new research direction. It is a scoped gap check, not a proof that no prior paper anywhere has ever used a related concept.

## Recent state of the field

Recent SEP forecasting work includes:

- multi-task deep learning with LSTM/transformer components and SHARP/flare/CME inputs;
- operational 24-hour probability forecasting for >10 MeV, >=10 pfu events;
- multimodal models using historical proton and soft-X-ray flux;
- classical XGBoost/SVM/logistic-regression approaches;
- real-time/near-real-time operational demonstrations;
- explicit operational distinction between expected onset and persistence of an event already underway.

Therefore none of these are defensible novelty claims for IRIS-SEP.

## Key observation motivating the new benchmark

Yu et al. (2026, SEPNET) explicitly state that because one physical SEP event can persist for longer than 24 hours, a single event may contribute positive labels to multiple consecutive prediction windows. Their dataset therefore contains more positive predictor windows than distinct SEP events.

That fact matters because ordinary window-level metrics weight each positive prediction window, not necessarily each independent physical event.

## Gap identified in the literature reviewed

The searches performed for this redesign did not identify a published SEP forecasting study whose central experiment simultaneously:

1. quantifies positive-window multiplicity per physical SEP episode;
2. recomputes forecast skill after normalizing every positive physical episode to equal total weight;
3. separates already-active persistence cases from genuinely new-onset cases before onset scoring;
4. tests whether model ranking changes under that evaluation change;
5. uses one shared episode/quiet-block bootstrap draw set for all paired comparisons;
6. publishes a complete per-issue attrition/eligibility ledger.

This combination is the candidate originality of `IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1`.

## Threatening prior work and how this project differs

### SEPNET / SEPNET-PRISM (Yu et al., 2026)

Threat: highly relevant modern 24-hour SEP forecasting, including operational models and historical proton/XRS information. It explicitly acknowledges repeated positive windows from persistent events.

Boundary: IRIS-SEP does not claim a stronger architecture. It tests the evaluation consequence of repeated physical-event representation and onset/persistence eligibility.

### Sadykov et al. / Ali et al.

Threat: daily SPE forecasts using preceding proton and soft-X-ray flux; XGBoost and proton dominance are already established directions.

Boundary: IRIS-SEP treats direct proton-history dependence as a mechanism to measure, not as novelty.

### Stumpo et al.

Threat: rigorous discussion of class imbalance, base rates, false alarm rate, and cross-validation issues in statistical proton-event forecasting.

Boundary: IRIS-SEP's question is physical-event multiplicity and onset/persistence weighting rather than generic class imbalance.

### NOAA/SWPC operational warning semantics

Threat: onset versus persistence is already an operational distinction.

Boundary: the project does not claim to invent the distinction. It asks whether evaluation metrics and model ranking change when that operational distinction is enforced in a rolling-window ML benchmark.

## Novelty statement permitted now

> We designed a preregistered evaluation benchmark that tests whether repeated 24-hour windows from the same physical SEP episode and already-active persistence states change measured forecast skill or model ranking. Every positive physical episode receives equal total weight, onset and persistence are scored separately, and paired uncertainty uses shared episode-level resampling.

## Stronger claim permitted only after results

Only if the frozen benchmark shows a robust effect may the project say that standard window-level evaluation materially changes apparent skill or model ranking in the tested cohorts/models.

No general claim about all SEP literature is allowed from one benchmark result.
