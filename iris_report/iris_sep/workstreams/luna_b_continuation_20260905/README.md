# Luna B continuation — SEP comparator fidelity and bounded research synthesis

Reviewed 2026-09-05. This continuation reads the existing IRIS-SEP source notes
first and adds only the newer comparator evidence and a bounded optimization
proposal. It does not train a model, download a table, inspect locked identities
or outcomes, tune against `validation_monitor`, or alter the frozen benchmark
and evaluation policy.

## Decision-relevant result

The strongest directly relevant public comparator is the 2026
SEPNET-PRISM preprint: it uses a 24-hour horizon and the operational
`>10 MeV, >10 pfu` event definition, and reports a best five-seed median TSS of
0.6784 and FAR of 0.3027 for its S+PF+X+F configuration (the S+PF+F arm reports
TSS 0.6703 and FAR 0.2743). Those numbers are not
comparable evidence for IRIS because the paper uses a random i.i.d. split,
selects thresholds with a combined HSS/TSS criterion, does not disclose
forecast-time publication-latency gates in the reported experiment, and uses a
newly constructed cohort. Its own discussion calls for temporally ordered
evaluation and flags input availability as unresolved. See the
[SEPNET-PRISM paper](https://arxiv.org/html/2606.14440), especially its data,
results, and evaluation sections.

The original SEPNET/SEPNET-O paper is closer in target and horizon, but still
does not establish equivalence to the local corrected V5 adapter. It reports
568 general events, 267 operational events, 11,773 predictor windows, and 1,726
operational-positive windows; its operational result is approximately TSS 0.36
on a random split, with high false alarms acknowledged. The published training
contract first learns the broad/general SEP target and re-optimizes an
operational threshold. The paper's SEPVAL panel contains 33 SEP and 30
non-event periods from 2011–2023. See the
[published SEPNET article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026JH001247)
and the [NASA CCMC SEPNET contract](https://ccmc.gsfc.nasa.gov/models/SEPNET~1/).

The local V5 adapter is therefore a corrected dense development comparator,
not a verified reproduction. It has 98 train-selected predictors; KNN imputation
(`k=10`), train-fitted min-max scaling and train-fitted `SelectKBest(f_classif,
k=98)`; shared 256→128→64→16 layers; classification on the general target;
`log1p(max flux)` regression; weighted BCE plus focal classification loss and
regression loss; Adam at 1e-4 with weight decay 1e-5; batch size 128; five
seeds; validation-monitor early stopping; validation-calibration intercept;
and validation-threshold TSS selection. V5 fixes earlier training-weight and
artifact-integrity defects, but its fixed V6 role construction can depend on
operational labels, its target/configuration equivalence to public SEPNET-O is
unresolved, and its general-episode weighting was rejected by the already
inspected monitor (0.074 versus 0.237 for row weighting). These are development
facts, not fresh evidence.

## Fidelity matrix

| Comparator | Target / cadence | Inputs and availability | Cohort / event count | Architecture and loss | Preprocess / configuration | Calibration, FAR, and reported skill | Fidelity to V5 / IRIS use |
|---|---|---|---|---|---|---|---|
| NASA CCMC SEPNET v1 | Probability of >10 MeV, >10 pfu in next 24 h; preceding 24 h; time-dependent | HMI SHARP NRT and SWPC flare list; CCMC documents timeliness caveats and probability, symmetric uncertainty, all-clear output | Live page gives no cohort count; public paper uses CLEAR-derived windows | Public operational contract does not expose full weights; paper describes dense multitask and SEPNET-TS temporal variant | Must preserve NRT versus definitive products and source latency; no peak-flux/time-of-peak output is documented | All-clear threshold configurable; no live FAR or calibration guarantee | Interface target/horizon match; executable weights and exact provenance do not, so `NOT_REPRODUCED` until receipted |
| Published SEPNET-O | Operational threshold applied after broad/general SEP training; 24 h non-overlapping windows | SHARP, flare summaries, CME summaries; 24 h min/mean/max summaries | 568 general events, 267 operational events; 11,773 windows; 1,726 operational-positive windows; SEPVAL 33 events + 30 non-events | Shared dense 256→128→64→16 representation; multitask regression/classification; SEPNET-TS uses LSTM + one Transformer; focal/BCE and auxiliary regression | KNN imputation and train scaling described; selected example lr 4.536e-4, wd 1.0856e-4, dropout 0.1039, batch 128; paper reports random splits and 50-seed aggregates in places | Operational TSS approximately 0.36; high FAR acknowledged; operational threshold re-optimized by HSS on training/validation pool; no IRIS-style calibration receipt | Structural predecessor only. V5 deliberately fixes role-safe calibration, thresholding, restart/evidence binding, and uses a different 98-feature local cohort; public equivalence unresolved |
| SEPNET-PRISM (2026 preprint) | Operational >10 MeV, >10 pfu in subsequent 24 h; fixed non-overlapping daily windows | SMHARP (SHARP + aligned SMARP), flare, CME, GOES XRSB, historical >10 MeV proton flux; SHARP 12 min, SMARP 96 min, proton 5 min; latency caveat remains | 14,464 samples, 650 operational positives (~4.5%); CLEAR event definition; 1986–2025 coverage | Two summarized daily steps (48 h) through BiLSTM + Transformer plus current-step MLP; task heads for SEP, future proton flux, future XRSB; external meta-probabilities; weighted BCE + focal + regression, regression warm-up, balanced mini-batches/positive weights | Per-feature min/mean/max; kNN k=5 then median for feature-importance analysis; random i.i.d. 20% test, 25% validation of train pool; Optuna search; five seeds | Best reported S+PF+X+F: TSS 0.6784, FAR 0.3027; S+PF+F: TSS 0.6703, FAR 0.2743; threshold maximizes the minimum of TSS and HSS; no independent reliability/calibration receipt | Highest-priority research lead, not a comparator score. Target/horizon and particle context are informative, but cohort, split, threshold policy, latency, and architecture differ materially |
| MEMPSEP-I | >10 MeV, 5 pfu within 6 h of flare onset; event probability plus properties | Remote sensing and in-situ inputs; CNN ensemble | SHINE 2022 validation: 8 events (2012–2017, M/X flares) and 14 non-events (2012–2022) | Multivariate CNN ensemble; calibrated ensemble probabilities | Event-triggered setup, not daily all-window forecast; exact training configuration is paper-specific | Brier score 0.2 on challenge predictions; not a 10 pfu/24 h FAR/TSS comparator | Useful uncertainty/ensemble precedent only; incompatible target, horizon, cohort, and issue process |
| UMASEP / ESPERTA family | Event-triggered >10 MeV warnings, generally after flare/SXR or particle evidence; horizons and thresholds vary by version | SXR, proton flux, flare location/size, radio evidence, empirical connectivity; some variants split well- and poorly-connected events | Cycle-specific samples and event-triggered evaluations; exact cohort comparability is unresolved here | Empirical lag-correlation / dual-model logic or logistic reinterpretation; not SEPNET architecture | Triggered observations and event-specific availability; not the daily row contract | Secondary-review numbers are intentionally omitted here because the original primary evaluation was not reverified in this continuation | Physically useful hypothesis source for connectivity and particle context, but incompatible evaluation. Never use as leaderboard evidence |
| Ali et al. 2025 RF/SVM study | >10 MeV, 10 pfu; event-focused datasets, not the IRIS daily new-crossing contract | Flare, CME, sweep/fixed radio; 1997–2022 source period | Dataset-specific, with balanced/hybrid/imbalanced settings; counts and event windows differ from CLEAR | Decision tree, RF, linear/nonlinear SVM; nested CV | Dataset-specific feature engineering and resampling; class balancing changes the base rate | Imbalanced sweep RF POD 0.85±0.08, FAR 0.30±0.05, TSS 0.78±0.07; fixed RF POD 0.76±0.12, FAR 0.31±0.08, TSS 0.71±0.11 | Comparable only as a classical physical-feature hypothesis. The reported scores are not transportable to IRIS without cohort, latency, and base-rate alignment |

The older local Luna B source receipt remains authoritative for the pinned
2025 SEPNET repository, V2 repository, CLEAR release, and known code defects.
The matrix above adds the newer PRISM paper and makes the incompatibility of
headline scores explicit. “FAR” is false alarm ratio in the cited studies;
IRIS must report its frozen matched-detection FAR policy separately.

## Ranked physical improvements

1. **Causal pre-event proton context, with a strict publication-time mask.**
   This is the strongest repeated signal: Sadykov et al. found that removing
   preceding proton features materially reduced skill, and PRISM's PF-containing
   configurations dominate its own table. Add only historical values available
   by issue time, plus missingness/freshness indicators. Falsifier: a
   predeclared PF-ablation has no positive paired inner-fold TSS delta or worsens
   matched-detection FAR/calibration. Risk is target leakage through already
   enhanced flux or retrospective event-list updates.

2. **Radiative context (GOES XRS history) alongside flare summaries.** PRISM
   reports a strong S+PF+X configuration and XRS is physically adjacent to
   eruptive energy. This is lower risk than adding a new head, provided XRS
   publication latency is audited. Falsifier: XRS adds no benefit after PF and
   flare features under chronological episode-disjoint folds.

3. **Magnetic connectivity proxies from causal source geometry.** Western-limb
   and far-side active-region context improved AR-based predictions in the
   Sadykov study; SEP transport theory also makes connectivity a defensible
   mechanism. Use declared geometric summaries or a precomputed connectivity
   proxy only when its inputs are available at issue time. Do not use an
   approximate AARP/HMI join. Falsifier: connectivity ablation is null or
   unstable across time blocks, or missingness causes degradation.

4. **Distribution-shift handling across magnetic products/cycles.** SMARP→SHARP
   alignment in PRISM extends coverage, but it is a source transformation that
   needs a train-only, overlap-period audit. Prefer source/product indicators and
   train-fitted standardization before a learned cross-product map. Falsifier:
   held-out cycle/product performance does not improve, or the map depends on
   future overlap information.

5. **CME/radio additions only after latency and incremental-value checks.** CME
   variables are physically relevant, but PRISM reports weaker/inconsistent
   incremental value after magnetic, PF, XRS, and flare inputs. Radio is
   promising in event-triggered systems but its cadence/availability is not yet
   a verified input for the daily IRIS cohort. Add neither to the first batch.

No image fusion, secondary forecast heads, hourly operation, or spacecraft
control claim is proposed. The frozen IRIS primary remains one daily probability
of a new >10 MeV, >=10 pfu crossing in the following 24 hours.

## Limitations and evidence boundary

Published scores are heterogeneous: event-triggered versus all-window issue
processes, 5 versus 10 pfu, 6 versus 24 hours, general enhancements versus
operational events, random versus chronological splits, and different base
rates. FAR and TSS cannot be ranked across those settings. PRISM's 2026 paper is
newer and highly informative, but its own random split and missing latency
receipt prevent claiming a fair win over IRIS or V5. The paper reports
threshold-based FAR/TSS, not a calibration curve or Brier/ECE analysis.

The V5 monitor has already been used for development. Its values are retained
only as historical negative/inconclusive evidence; they must not be relabeled as
fresh evidence for the proposed batch. No locked-test identity, outcome,
prediction, or threshold was accessed in preparing this continuation.

## Sources

- Yu et al., *Realtime forecasting ... SEPNET-PRISM* (arXiv v2, 2026):
  [paper](https://arxiv.org/html/2606.14440).
- Yu et al., *Solar Energetic Particle Forecasting With Multi-Task Deep
  Learning: SEPNET* (JGR ML&C, 2026):
  [article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026JH001247),
  [official code](https://github.com/yuyian/SEP-Prediction).
- NASA CCMC, [SEPNET v1 model contract](https://ccmc.gsfc.nasa.gov/models/SEPNET~1/).
- Chatterjee et al., MEMPSEP-I, [DOI and article](https://doi.org/10.1029/2023SW003568).
- Sadykov et al., [daily SPE prediction and all-clear study](https://arxiv.org/abs/2107.03911).
- Ali et al., [multi-source RF/SVM study](https://doi.org/10.1038/s41598-025-92207-1).
- CLEAR Center, [benchmark and cross-mission provenance](https://clear.engin.umich.edu/science/).
