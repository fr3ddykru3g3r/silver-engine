# Results and exact evidence status

## Dataset-wide event audit: recorded frozen evidence

The archived model-free audit reports 14,464 windows, 650 positive windows, 614 uniquely mapped positives, 257 represented physical episodes and multiplicity 614/257=2.389105. It reports 418 persistence windows, 228 genuine-onset windows and four ambiguous positives. Thirty-six multi-episode overlaps are a separate mapping category; do not add them again to those target classes. These are dataset-wide values, not the smaller out-of-time prediction cohort.

Source: upstream `yuyian/SEP-Prediction-V2` revision `e138dcd72c1952a00e11e1a0b025337f9e7c93fb`. Rolling-table SHA256 `4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb`; event-table SHA256 `0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0`. Model-free artifact 10136909164 / run 34438057070. Its reported zero reconstruction mismatches tests agreement with catalog semantics, not independent physical flux truth. This audit cryptographically verified the archive but did not re-interpret its complete outcomes because of the protected-boundary conflict.

## Original frozen prediction endpoint: recorded values, unchanged

Replay artifact 10137507101 / [run 34438987251](https://github.com/fr3ddykru3g3r/silver-engine/actions/runs/34438987251), execution `296f302111371e8d421fb5f2a4569ea2c06bd6e1`. Full hash is in the manifest. The original endpoint and original uncertainty are preserved; they are not replaced by the new diagnostic below.

| Comparator | Mapped TSS | Episode-normalized TSS | Onset TSS |
|---|---:|---:|---:|
| Joint XGBoost | 0.726455 | 0.621128 | 0.437234 |
| No-proton XGBoost | 0.508922 | 0.514758 | 0.477391 |
| Past-proton proxy | 0.601224 | 0.446983 | 0.092894 |
| Elastic net joint | −0.122768 | −0.182816 | −0.266990 |
| Elastic net no-proton | −0.318921 | −0.332090 | −0.333015 |
| Fit-prevalence climatology | 0 | 0 | 0 |

Original paired 95% intervals: joint normalized−mapped [−0.146175,−0.063165]; joint onset−normalized [−0.246639,−0.125070]; proxy onset−mapped [−0.566675,−0.443973]. Original joint−no-proton onset interval [−0.162967,0.082920] includes zero. Original draws: 10,000, seed 20260910, physical positive episodes and quiet blocks with shared draws. Do not report the fraction of negative draws as a p-value.

Original aggregate confusion matrices and Brier/HSS tables are retained in the immutable archive. The complete independently recomputed metric table below is for the separately labeled strict cohort only. Full original numerical reverification remains blocked; all original aggregate metric rows are provided verbatim in [original_frozen_aggregate_metrics.csv](results/original_frozen_aggregate_metrics.csv), with this recorded-evidence qualification.

## Independently recomputed strict historical diagnostic

7,509 distinct admitted issues; 197 positive windows; 85 represented onset episodes; 112 active positive windows; 7,312 negatives and 1,080 quiet blocks. Every admitted outcome horizon ends strictly before the protected boundary. Models, probabilities and original operating thresholds are fixed. This is retrospective, development-exposed evidence.

Episode normalization assigns each positive row weight 1/n_i and each negative row weight one. Fractional TP/FN are weighted masses, not fractional observed events. Removing active windows yields the onset view. The same negative rows/weights are retained. FAR, HSS and Brier can change with the resulting class prior; their cross-view differences do not describe a newly trained model.

| Model | View | TP mass | FN mass | FP | TN | POD | FPR | FAR | TSS | HSS | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elastic_net_joint | MAPPED_STANDARD | 40.000000 | 157.000000 | 2383.000000 | 4929.000000 | 0.203046 | 0.325903 | 0.983492 | -0.122857 | -0.018908 | 0.085720 |
| elastic_net_joint | EPISODE_NORMALIZED_OCCURRENCE | 12.154762 | 72.845238 | 2383.000000 | 4929.000000 | 0.142997 | 0.325903 | 0.994925 | -0.182905 | -0.012674 | 0.075231 |
| elastic_net_joint | NEW_ONSET | 5.000000 | 80.000000 | 2383.000000 | 4929.000000 | 0.058824 | 0.325903 | 0.997906 | -0.267079 | -0.018561 | 0.075949 |
| elastic_net_no_proton | MAPPED_STANDARD | 19.000000 | 178.000000 | 3038.000000 | 4274.000000 | 0.096447 | 0.415481 | 0.993785 | -0.319035 | -0.039566 | 0.105476 |
| elastic_net_no_proton | EPISODE_NORMALIZED_OCCURRENCE | 7.078571 | 77.921429 | 3038.000000 | 4274.000000 | 0.083277 | 0.415481 | 0.997675 | -0.332204 | -0.018243 | 0.093292 |
| elastic_net_no_proton | NEW_ONSET | 7.000000 | 78.000000 | 3038.000000 | 4274.000000 | 0.082353 | 0.415481 | 0.997701 | -0.333128 | -0.018294 | 0.093408 |
| fit_prevalence_climatology | MAPPED_STANDARD | 197.000000 | 0.000000 | 7312.000000 | 0.000000 | 1.000000 | 1.000000 | 0.973765 | 0.000000 | 0.000000 | 0.026059 |
| fit_prevalence_climatology | EPISODE_NORMALIZED_OCCURRENCE | 85.000000 | 0.000000 | 7312.000000 | 0.000000 | 1.000000 | 1.000000 | 0.988509 | 0.000000 | 0.000000 | 0.012885 |
| fit_prevalence_climatology | NEW_ONSET | 85.000000 | 0.000000 | 7312.000000 | 0.000000 | 1.000000 | 1.000000 | 0.988509 | 0.000000 | 0.000000 | 0.012885 |
| past_proton_ge10_proxy | MAPPED_STANDARD | 121.000000 | 76.000000 | 95.000000 | 7217.000000 | 0.614213 | 0.012992 | 0.439815 | 0.601221 | 0.574274 | 0.022773 |
| past_proton_ge10_proxy | EPISODE_NORMALIZED_OCCURRENCE | 39.097619 | 45.902381 | 95.000000 | 7217.000000 | 0.459972 | 0.012992 | 0.708439 | 0.446980 | 0.347722 | 0.019049 |
| past_proton_ge10_proxy | NEW_ONSET | 9.000000 | 76.000000 | 95.000000 | 7217.000000 | 0.105882 | 0.012992 | 0.913462 | 0.092890 | 0.083650 | 0.023117 |
| xgb_joint | MAPPED_STANDARD | 152.000000 | 45.000000 | 330.000000 | 6982.000000 | 0.771574 | 0.045131 | 0.684647 | 0.726442 | 0.426351 | 0.013870 |
| xgb_joint | EPISODE_NORMALIZED_OCCURRENCE | 56.630952 | 28.369048 | 330.000000 | 6982.000000 | 0.666246 | 0.045131 | 0.853527 | 0.621115 | 0.225559 | 0.008495 |
| xgb_joint | NEW_ONSET | 41.000000 | 44.000000 | 330.000000 | 6982.000000 | 0.482353 | 0.045131 | 0.889488 | 0.437222 | 0.164196 | 0.011394 |
| xgb_no_proton | MAPPED_STANDARD | 136.000000 | 61.000000 | 1326.000000 | 5986.000000 | 0.690355 | 0.181346 | 0.906977 | 0.509010 | 0.123421 | 0.022738 |
| xgb_no_proton | EPISODE_NORMALIZED_OCCURRENCE | 59.176190 | 25.823810 | 1326.000000 | 5986.000000 | 0.696190 | 0.181346 | 0.957279 | 0.514845 | 0.060151 | 0.011362 |
| xgb_no_proton | NEW_ONSET | 56.000000 | 29.000000 | 1326.000000 | 5986.000000 | 0.658824 | 0.181346 | 0.959479 | 0.477478 | 0.055906 | 0.011719 |

## Mechanism and uncertainty

| Model | Multiplicity contribution | Persistence contribution | Mapped−onset | Algebra residual |
|---|---:|---:|---:|---:|
| elastic_net_joint | 0.060048 | 0.084174 | 0.144222 | 1.2e-16 |
| elastic_net_no_proton | 0.013169 | 0.000924 | 0.014094 | 4e-17 |
| fit_prevalence_climatology | 0.000000 | 0.000000 | 0.000000 | 3.6e-17 |
| past_proton_ge10_proxy | 0.154241 | 0.354090 | 0.508331 | 1.7e-16 |
| xgb_joint | 0.105327 | 0.183894 | 0.289221 | 8.3e-17 |
| xgb_no_proton | -0.005835 | 0.037367 | 0.031532 | 1.8e-16 |

New paired bootstrap: 10,000 draws, seed 20260915; shared event/quiet-block draws. Point contrasts are in the mechanism table; bootstrap medians are not substituted for them.

| Contrast | Median | 95% interval | 98.333% interval |
|---|---:|---|---|
| elastic_net_joint__normalized_minus_mapped | -0.058903 | [-0.097232, -0.022075] | [-0.106121, -0.012680] |
| elastic_net_joint__onset_minus_normalized | -0.083978 | [-0.132355, -0.038513] | [-0.144847, -0.029533] |
| elastic_net_no_proton__normalized_minus_mapped | -0.012813 | [-0.043879, 0.016622] | [-0.050899, 0.023213] |
| elastic_net_no_proton__onset_minus_normalized | -0.001373 | [-0.034230, 0.033137] | [-0.041317, 0.042307] |
| fit_prevalence_climatology__normalized_minus_mapped | 0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] |
| fit_prevalence_climatology__onset_minus_normalized | 0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] |
| past_proton_ge10_proxy__normalized_minus_mapped | -0.152200 | [-0.191141, -0.114853] | [-0.198236, -0.106462] |
| past_proton_ge10_proxy__onset_minus_normalized | -0.354090 | [-0.421210, -0.285882] | [-0.437779, -0.270793] |
| xgb_joint__normalized_minus_mapped | -0.104031 | [-0.146905, -0.063668] | [-0.155014, -0.055152] |
| xgb_joint__onset_minus_normalized | -0.183193 | [-0.245378, -0.125063] | [-0.260319, -0.111856] |
| xgb_no_proton__normalized_minus_mapped | 0.004622 | [-0.043503, 0.054737] | [-0.054054, 0.066766] |
| xgb_no_proton__onset_minus_normalized | -0.037059 | [-0.090224, 0.014875] | [-0.101870, 0.024585] |
| proxy_onset_minus_mapped | -0.506919 | [-0.567610, -0.441737] | [-0.579049, -0.428049] |
| xgb__joint_minus_no_proton_onset | -0.040652 | [-0.164140, 0.083801] | [-0.192713, 0.107983] |
| elastic_net__joint_minus_no_proton_onset | 0.067084 | [0.014702, 0.110773] | [0.001837, 0.120895] |

The 98.333% intervals are a conservative descriptive three-contrast sensitivity, not a cure for all previous model selection or a familywise guarantee covering every exploratory contrast listed. No null hypothesis p-values were generated.

### Nearby-event dependence

Complete calendar bins, including empty bins, are resampled with the same indices across views/models. Each positive episode follows its onset bin. This tests temporal co-clustering, not event boundary definitions. Negative terms cancel for these within-model contrasts. Each grouping uses 10,000 draws, seed 20260916.

| Bin | Joint normalized−mapped 95% | Joint onset−normalized 95% | Proxy onset−mapped 95% |
|---|---|---|---|
| 27 | [-0.146179, -0.063131] | [-0.247802, -0.125407] | [-0.560693, -0.450262] |
| 90 | [-0.144749, -0.067678] | [-0.245739, -0.125896] | [-0.563812, -0.440411] |
| quarter | [-0.145996, -0.064977] | [-0.239156, -0.128815] | [-0.560697, -0.438678] |
| year | [-0.153825, -0.058372] | [-0.245565, -0.117524] | [-0.548458, -0.450563] |

All four groupings preserve the signs of the three principal contrasts. They do not establish stationarity across solar cycles or out-of-sample model-selection validity.

Other executed checks: three outer folds (9,40,36 onset episodes), positive weighting exponents alpha=0,0.25,0.5,0.75,1; fixed descriptive threshold multipliers 0.5,0.75,1,1.25,1.5,2; 14/28-day quiet-block variants; seven-day role-edge exclusion without refitting. Joint multiplicity/persistence terms remain positive in the specified fold/threshold checks. The no-proton control is not consistently zero or negative in every fold. Quiet-block changes leave the principal within-model contrasts unchanged because FPR cancels. Role-edge exclusion cannot fix contamination of already-trained models. No diagnostic threshold was promoted.

## Direct Phase-II and controller archive reconstruction

Fit 2011–14, calibration 2015, threshold 2016, score 2017; 210 score issues, one true onset. Threshold selection itself had one onset. Five model seeds were used; the engineered predictor family has 15 derived features. Calibration is intercept-only. Recorded engineered-minus-baseline bootstrap improvement interval includes zero. These experiments were locally frozen before their execution, but the data had already been development-exposed.

| Model/policy | Threshold or rule | TP | FN | FP | TN | POD | FPR | FAR | TSS | HSS | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_onset_engineered | 0.00129808433257 | 1.000000 | 0.000000 | 107.000000 | 102.000000 | 1.000000 | 0.511962 | 0.990741 | 0.488038 | 0.008997 | 0.004786 |
| direct_onset_joint | 0.000403947324501 | 1.000000 | 0.000000 | 205.000000 | 4.000000 | 1.000000 | 0.980861 | 0.995146 | 0.019139 | 0.000186 | 0.006184 |
| direct_onset_xrs | 0.0013255810151 | 1.000000 | 0.000000 | 167.000000 | 42.000000 | 1.000000 | 0.799043 | 0.994048 | 0.200957 | 0.002389 | 0.004831 |
| occurrence_trained_xrs_baseline | 0.0039173288699 | 1.000000 | 0.000000 | 123.000000 | 86.000000 | 1.000000 | 0.588517 | 0.991935 | 0.411483 | 0.006615 | 0.004550 |
| threshold_selected_blend | 0.00129808433257 | 1.000000 | 0.000000 | 107.000000 | 102.000000 | 1.000000 | 0.511962 | 0.990741 | 0.488038 | 0.008997 | 0.004786 |
| Controller V1 | rolling quiet quantile 0.85 | 0.000000 | 1.000000 | 41.000000 | 168.000000 | 0.000000 | 0.196172 | 1.000000 | -0.196172 | -0.009384 | 0.004786 |
| Controller V2 | rolling quiet quantile 0.8 | 1.000000 | 0.000000 | 50.000000 | 159.000000 | 1.000000 | 0.239234 | 0.980392 | 0.760766 | 0.029395 | 0.004786 |

The controller uses the prior 120 days, at least 30 resolved quiet examples, a 25-hour outcome-resolution delay and a floor equal to the frozen base threshold. Its thresholds and alerts were independently reconstructed, and probabilities matched the archived Phase-II model. This verifies mathematical past-label use under that delay assumption; it does not establish that NOAA labels really became available within 25 hours.

One detection from one event gives an exact two-sided binomial 95% interval [0.025,1] under an independent Bernoulli assumption. Selection on that same event makes even this insufficient as a post-selection validation interval. FAR 98% means 50 of 51 issued alerts were false in this small retrospective cohort, not that 98% of quiet days were alerted.

See the negative ledger for the distinct PR #6 experiment and every NOAA source failure. None establishes a live-compatible trained model.
