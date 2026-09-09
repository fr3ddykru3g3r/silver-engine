# IRIS-SEP Freshness Crossover Study V1 — Result (2026-09-09)

## Decision

`NEGATIVE RESULT — NO FRESHNESS CROSSOVER DEMONSTRATED; NO SWITCHING POLICY PASSED THE PRACTICAL GATE`

This is retrospective development evidence only. It is not an independent final test, not an operational replay, and not evidence of award-winning forecast skill.

## Execution identity

- Branch: `codex/iris-sep-continuation-20260905`
- Executed commit: `ed3aba1def18bb022382efa4afaaf251c76e6cff`
- GitHub Actions run: `34341335588`
- Evidence artifact: `10100069342`
- Artifact SHA-256: `ba907ffae86301c22b1afb3a41a0bf98c825b562b90b5319ce347a4af1db7868`
- Study: `IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1`
- Two-satellite measurement definition: GOES-13 NASA/SPDF OMNI five-minute `Av` >10 MeV integral proton flux plus GOES-15 NOAA/NCEI operational XRS.
- Historical SWPC-primary-stream equivalence: **not established**.

## Cohort support

| Role | Eligible daily issues | Positive opportunities / episodes | Prevalence |
|---|---:|---:|---:|
| Fit (2011–2014) | 968 | 30 | 3.10% |
| Calibration (2015) | 225 | 2 | 0.89% |
| Threshold/rule selection (2016) | 250 | 1 | 0.40% |
| Retrospective score (2017 Jan–Nov) | 211 | 3 | 1.42% |

The very small 2016 and 2017 positive counts are a major statistical limitation. Any apparently strong TSS or threshold result must be interpreted accordingly.

## Clean-signal screen

The original preregistered clean-signal screen was preserved. It passed only because the XRS-only model met the original criteria on the 2016 threshold-selection role.

2016 threshold-selection results:

| Model | TP | FP | FN | TN | TSS | FAR | Brier | AUPRC | 2016 prevalence | Original gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Joint XRS+proton | 1 | 41 | 0 | 208 | 0.835 | 0.976 | 0.00403 | 0.0238 | 0.004 | Fail |
| XRS only | 1 | 5 | 0 | 244 | 0.980 | 0.833 | 0.00392 | 0.1667 | 0.004 | Pass |
| Proton only | 1 | 154 | 0 | 95 | 0.382 | 0.994 | 0.00401 | 0.00645 | 0.004 | Fail |

The corrected diagnostic also reports AUPRC against the **2016 cohort prevalence**, while the Brier comparator remains the fit-derived constant probability. All three XGBoost AUPRC values exceed 2016 prevalence, but this does not establish useful forecasting because threshold selection was performed on a cohort containing only one positive opportunity. `MAX_TSS > 0` is a screening authorization only.

## Retrospective clean controls on 2017

At zero injected delay on the 211-issue retrospective score cohort:

| Model | TP | FP | FN | TN | TSS | FAR | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint XRS+proton | 3 | 152 | 0 | 56 | 0.269 | 0.981 | 0.141 |
| XRS only | 2 | 73 | 1 | 135 | 0.316 | 0.973 | 0.0235 |
| Proton only | 3 | 102 | 0 | 106 | 0.510 | 0.971 | 0.257 |

This is the central explanatory result: the reduced-input proton-only model already outperformed the clean joint model before XRS was made stale. Therefore a later advantage of fresh proton-only over XRS-stale joint cannot be interpreted as a freshness-induced model-order reversal.

Likewise, XRS-only already slightly exceeded the clean joint TSS at zero proton delay (0.316 vs 0.269), although it missed one of the three positive opportunities. The delayed-proton comparison therefore also does not begin from a clean joint superiority condition.

## Performance versus delay

### XRS delayed, proton fresh

Fresh proton-only TSS remained `0.510` at every delay because the proton input was unchanged. Joint-model TSS across XRS delays was:

| XRS delay (min) | Joint TSS | Fresh proton-only TSS | Difference (reduced − joint) |
|---:|---:|---:|---:|
| 0 | 0.269 | 0.510 | +0.240 |
| 5 | 0.293 | 0.510 | +0.216 |
| 15 | 0.332 | 0.510 | +0.178 |
| 30 | 0.313 | 0.510 | +0.197 |
| 60 | 0.308 | 0.510 | +0.202 |
| 120 | 0.313 | 0.510 | +0.197 |
| 360 | 0.298 | 0.510 | +0.212 |
| 720 | 0.341 | 0.510 | +0.168 |
| 1440 | 0.192 | 0.510 | +0.317 |

Paired block/bootstrap intervals for `reduced − joint` were positive at every XRS delay, including delay zero; for example, delay 0 CI was approximately `[0.082, 0.385]`, and 1440 min was approximately `[0.177, 0.457]`.

This is **not a crossover**. It is evidence that, on this tiny retrospective cohort and with these fixed models, the proton-only model was superior to the joint model even when both inputs were fresh.

The delayed-XRS-only control also varied substantially with delay and sometimes degraded, confirming that the manipulation actually altered the information available. But it does not rescue a crossover interpretation because the ordering was already reversed at delay zero.

### Proton delayed, XRS fresh

Fresh XRS-only TSS was `0.316` at each delay. Joint-model TSS ranged from about `0.173` to `0.293` across the delay grid. The point estimate `reduced − joint` was positive at every delay, but paired bootstrap intervals were broad and crossed zero at every proton-delay setting because only three positive score opportunities existed.

Again, no crossover bracket was found because the reduced model was not demonstrably worse than the clean joint model at delay zero.

## Frozen switching-policy result

The preregistered policy selector chose `always_reduced_fresh` as the non-abstaining reference for both delayed-feed families. The fixed expiry rule consequently selected TTL = 0 minutes in both families, i.e. immediately use the reduced model rather than retain the stale joint model.

On the 2017 retrospective score cohort:

### XRS delayed

- Reference (fresh proton-only) per delay: TP=3, FP=102, FN=0, TN=106, TSS=0.510, coverage=1.00.
- Fixed TTL: identical to reference at every delay.
- State-dependent TTL produced **more** false alerts when pooled across delays: 988 vs 816, with no reduction in misses.
- Pooled false-alert reduction versus reference: `0%` for fixed TTL; `-21.1%` for the state-dependent rule.
- Practical gate: **FAIL**.

### Proton delayed

- Reference (fresh XRS-only) per delay: TP=2, FP=73, FN=1, TN=135, TSS=0.316, coverage=1.00.
- Fixed TTL and state-dependent TTL were identical to the reference across the score delay grid.
- Pooled false-alert reduction: `0%`.
- Additional misses: `0` relative to reference, but the reference itself missed one of three positives.
- Practical gate: **FAIL**.

No confidence-rejection policy was selected. No policy achieved the frozen ambition of at least 20% fewer false alerts, zero additional missed crossings, and at least 80% coverage.

## Scientific interpretation

The preregistered discovery claim is **not supported** by this run:

> Under identifiable conditions, using stale additional measurements is worse than using fewer fresh measurements—and a simple rule can exploit that relationship.

The experiment did observe cases where the reduced model beat the stale joint model, but the clean controls show that the reduced model was already competitive or superior at zero delay. Therefore the study cannot isolate a freshness-induced reversal in model ordering.

The strongest defensible conclusion is narrower:

> In this retrospective two-satellite dataset, adding the second measurement family did not reliably improve the fixed joint forecaster. Because the reduced model was already as good as or better than the joint model before delay injection, the experiment did not demonstrate a freshness crossover, and the frozen switching policies did not improve the warning trade-off.

This is a scientifically valid negative result, not a reason to post-hoc change thresholds, delay grids, features, or policy families.

## Statistical limitation

Only one positive opportunity was available in the 2016 threshold-selection role and three in the 2017 retrospective score role. Those counts are too small for a strong population-level claim. Replaying each event under multiple delays increases experimental conditions, **not independent event support**; the paired uncertainty procedure therefore keeps all delay variants of the same underlying unit together.

## Disposition

`STOP THIS FRESHNESS-CROSSOVER POLICY BRANCH AS A WINNING-CLAIM PATH.`

Do not tune another switching rule on these outcomes. Preserve the result as negative evidence. Any next research direction must be justified independently and must not relabel this retrospective score period as untouched evidence.
