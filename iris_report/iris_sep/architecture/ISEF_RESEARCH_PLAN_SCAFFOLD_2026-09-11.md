# ISEF research-plan scaffold — 2026-09-11

**Status:** planning/checking scaffold only. The final research plan must reflect what the student researchers actually understand, did, and have approval to do. Do not backdate approvals or paste generated prose as student-authored work.

## 1. Project title

Working title:

**Are We Forecasting a New Solar Radiation Storm? An Episode-Normalized Causal Benchmark for SEP Prediction**

## 2. Rationale / background points

Student-written rationale should cover:

- solar energetic particle (SEP) events are radiation hazards relevant to spacecraft, human spaceflight and some aviation operations;
- the operational event in this study is a >10 MeV proton flux crossing 10 proton flux units (pfu);
- contemporary SEP machine-learning work frequently represents predictors and targets in fixed time windows;
- a single physical event can occupy several positive daily windows;
- some daily issue times occur after the event is already active;
- therefore a window-level occurrence score can answer a different question from pre-onset warning skill.

## 3. Research question

Primary scientific question:

**How much do repeated representation of physical SEP episodes and already-active persistence change the measured skill of fixed 24-hour SEP forecasts?**

The project is an evaluation-methodology study, not a claim to invent SEP onset/persistence or a state-of-the-art forecasting architecture.

## 4. Hypothesis and falsification

Frozen mechanism hypothesis:

1. repeated windows from one physical SEP episode can change measured skill when each window is given full positive mass;
2. already-active persistence can contribute to occurrence-style skill while providing a different kind of information from a pre-onset forecast.

Falsifying/weakening observation:
- if fixed forecasts retain essentially the same TSS after physical-episode normalization and causal onset restriction, with paired physical-unit intervals centered near zero, the proposed evaluation effect is not supported.

Prospective falsification:
- a later frozen custodian-controlled cohort may return `NO_PROSPECTIVE_CONFIRMATION` or `INSUFFICIENT_CONFIRMATORY_INFORMATION`; neither outcome may be repaired by changing the same cohort after labels are inspected.

## 5. Data sources

Historical methodology arm:
- pinned public SEP-PRISM rolling table;
- pinned public GOES operational SEP event catalogue;
- persisted replay predictions and frozen chronological roles generated from development-exposed historical data.

Prospective predictor-only extension, if approved before execution:
- public NOAA/SWPC primary GOES integral-proton input;
- public NOAA/SWPC XRS/routing interfaces for source readiness and possible separately frozen optional models;
- protected outcome labels constructed only by an independent custodian after the execution contract is frozen.

No human participants, vertebrate animals, PHBAs, or wet-lab hazardous materials are part of the current methodology.

## 6. Historical procedure

### 6.1 Reconstruct operational physical episodes

- parse pinned event starts/ends;
- define the target threshold as >10 MeV proton flux >=10 pfu;
- independently reconstruct whether each daily future window overlaps a qualifying episode;
- verify reconstructed labels against the stored table;
- preserve ambiguity rather than forcing a physical assignment.

### 6.2 Classify issue-time state

For each issue time:
- already active at issue -> persistence;
- not active at issue + new threshold crossing in `(issue, issue+24h]` -> onset positive;
- no qualifying crossing -> onset negative;
- multiple/ambiguous physical mappings -> explicit ambiguity code.

### 6.3 Define three evaluation views

1. **Mapped occurrence:** ordinary occurrence scoring on uniquely mapped positives.
2. **Episode-normalized occurrence:** an episode producing `m` positive windows gives each such window weight `1/m`.
3. **New-onset causal:** already-active persistence is excluded from onset-positive credit; only a future crossing from a non-active issue state is positive.

Forecast probabilities and frozen alert thresholds remain unchanged between views.

### 6.4 Fixed model replay

Frozen comparator families:
- prevalence climatology;
- deterministic past-proton >=10 pfu state proxy;
- joint XGBoost ensemble;
- proton-free XGBoost ensemble;
- joint elastic-net logistic regression;
- proton-free elastic-net logistic regression.

Chronological fit, threshold and held-forward score periods are fixed. Do not retune after inspecting replay results.

## 7. Statistical analysis

Primary metric: True Skill Statistic (TSS).

Supporting metrics:
- sensitivity / probability of detection;
- false-positive rate;
- false-alarm ratio;
- HSS/Brier where defined by the persisted analysis.

Uncertainty:
- resample distinct mapped onset episodes as positive units;
- resample Monday-anchored seven-day quiet blocks as negative units;
- use 10,000 bootstrap draws;
- reuse the same draw tensors across paired comparisons;
- report paired percentile 95% intervals.

Do not treat daily windows from the same physical storm as independent bootstrap observations.

## 8. Historical primary contrasts

Predeclared primary contrasts:

1. joint XGBoost: episode-normalized occurrence minus mapped occurrence;
2. joint XGBoost: new onset minus episode-normalized occurrence;
3. deterministic past-proton proxy: new onset minus mapped occurrence.

The final project should report the frozen outcomes regardless of direction and preserve negative/null secondary results.

## 9. Reproducibility controls

- pin source hashes;
- freeze JSON analysis contracts;
- retain one prediction row per issue/model;
- retain physical mapping/eligibility state;
- retain shared bootstrap draw tensors;
- verify evidence hashes;
- use an independent verifier that does not import the benchmark runner;
- preserve correction history and failed/negative experiments;
- keep protected final outcomes unavailable to development code.

## 10. Prospective extension — include only if approved before it starts

Study ID:
`IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`.

Key precommitted rules:
- daily issue at 00:00 UTC;
- predictor data must be demonstrably captured before issue time;
- missing/late required predictor -> `ABSTAIN`;
- deterministic `past_proton_active_proxy` is mandatory;
- model/rule and feature-schema hashes frozen before protected outcome access;
- predictions stored in an append-only hash-chained ledger;
- protected labels/episode identities held by independent custodian;
- 10,000 shared physical-unit bootstrap draws, seed `20260911`;
- minimum information floor: 50 distinct onset episodes and 500 quiet blocks;
- mandatory prospective contrast: proxy `NEW_ONSET_CAUSAL TSS - MAPPED_OCCURRENCE TSS`;
- result cannot be tuned/rescued on the same protected cohort.

Important timing point: if this prospective phase is intended to count as part of the same ISEF project, obtain the required fair/SRC approval before beginning that phase.

## 11. Risks / safety / ethics

Current work is computer-based analysis of public space-weather data and protected computational evidence. Primary risks are research-integrity risks rather than physical hazards:

- leakage of future information;
- repeated-event pseudo-independence;
- post-result tuning;
- accidental protected-outcome inspection;
- overclaiming exposed historical evidence as prospective validation.

Controls are frozen temporal roles, explicit target exclusions, physical-unit inference, hash-bound evidence, fail-closed prospective input rules, independent custody and written claim boundaries.

## 12. Expected outcomes

Do not preregister a guaranteed direction for every model. Different architectures/features may respond differently to the change in estimand.

The scientific output is the measured difference between evaluation definitions, with uncertainty and explicit failure/null outcomes.

## 13. Current evidence boundary

Historical replay evidence is development/methodology evidence even though its computation has been independently verified. Public origin does not make already-inspected source hashes an untouched test.

No statement of operational readiness, state-of-the-art superiority, universal literature bias, economic savings or award outcome is part of the hypothesis.

## 14. Bibliography verification checklist

At minimum, student researchers should personally verify and cite the authoritative records for:
- NOAA Space Weather Scales / operational SEP threshold;
- SEP-PRISM data paper;
- SEPNET;
- representative multi-cycle proton/XRS ML work;
- representative SEP model-validation work;
- SEPVAL resource used in the surrounding literature.

Bibliographic metadata should be checked against publisher/official records before final submission.
