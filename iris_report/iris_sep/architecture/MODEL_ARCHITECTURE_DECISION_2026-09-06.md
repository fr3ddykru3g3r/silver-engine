# IRIS-SEP model architecture decision — 2026-09-06

## Decision

Freeze **cross-fitted specialist evidence stacking** as the current IRIS-SEP development architecture.

Do not promote:

- the original softmax-gated compact fusion as the primary architecture;
- the hard-anchored solar residual model;
- the two-state current/lag/delta expansion.

Do not add an LSTM, Transformer, diffusion model, graph neural network or larger neural backbone on the current daily aggregate table. The information interface does not justify those additions.

This remains development-only. The score and 2023–2025 monitor have already been inspected and cannot become fresh final evidence.

## Final development architecture

```text
causal solar features ──> five-seed XGBoost specialist ──┐
causal XRS features   ──> five-seed XGBoost specialist ──┼─> cross-fitted evidence stack
causal proton context ──> five-seed XGBoost specialist ──┘          │
                                                                   v
                                                        calibration intercept
                                                                   │
                                                                   v
                                                        frozen decision threshold
                                                                   │
                                                                   v
                                                         validity/admission envelope
                                                                   │
                                                    VALID / DEGRADED / ABSTAIN
```

The fusion weights are learned only from out-of-fold predictions generated inside the outer fit role. Calibration is therefore calibration again, rather than the place where the architecture learns its modality mixture.

## Why the old compact fusion is structurally wrong for the evidence

`IRISSEPTabularModel` encodes modalities separately but then makes their embeddings compete through a softmax-weighted average. The empirical diagnostics show that modality separation helps but that the relative importance of solar, XRS and proton evidence can change sharply. A convex representation average is therefore an unnecessary bottleneck.

The stronger system lets each modality make a complete probability forecast first and combines only the resulting evidence.

## Experiment 1: hard solar anchor — rejected

Preregistered architecture:

`logit(p) = logit(p_solar) + b + w_xrs * e_xrs + w_proton * e_proton`

with nonnegative bounded context residuals.

The fitted context weights collapsed (`w_xrs≈0.0368`, `w_proton≈0.0081`). On the operational POD>=0.8/min-FAR policy, the 2023–2025 monitor paired TSS difference versus the existing late-fusion model was about `-0.255`, with the complete 95% interval below zero. The solar coefficient must not be hard-wired to one.

Artifact: GitHub Actions run `33984943106`, head `8b008f915fa635e602265e52ad68cbddaec3c380`, artifact digest `sha256:1f7e6318bb797525160433c90760bec7692a25932420cefcdfdc5776f070db38`.

## Experiment 2: cross-fitted evidence stack — promoted

Architecture:

1. Train independent five-seed median XGBoost specialists for solar, XRS and proton context.
2. Generate out-of-fold specialist predictions using four expanding chronological folds entirely inside the outer fit role.
3. Convert probabilities to prevalence-centred log-odds evidence.
4. Learn three nonnegative evidence weights from those out-of-fold fit predictions only.
5. Refit specialists on the complete outer fit role.
6. Fit only one final calibration intercept on the calibration role.
7. Select decisions on the threshold role only.

Learned meta weights:

- solar: `0.1269`
- XRS: `0.2447`
- proton: `0.1823`

Operational POD>=0.8/min-FAR TSS:

| model | older score | 2023–2025 monitor |
|---|---:|---:|
| solar-only | 0.4596 | -0.0765 |
| previous late fusion | 0.5126 | 0.1894 |
| **cross-fitted stack** | **0.5120** | **0.2359** |

The cross-fitted stack preserves the older operational score while improving the later development monitor. Its paired monitor advantage over the previous late-fusion system is positive in the point estimate (`~+0.0425`) but its 95% interval crosses zero, so superiority is not established.

Probability quality on the 2023–2025 monitor also improved:

- AUROC: `0.6724 -> 0.6825`
- Brier: `0.03464 -> 0.03107`
- ECE: `0.02680 -> 0.01098`
- matched-POD=0.8 FAR: `0.95695 -> 0.95227`

AUPRC was slightly lower (`0.1088 -> 0.1058`) and remains disclosed.

Artifact: GitHub Actions run `33985243824`, head `5d90a177462401d1486891e06c6fa6794ea4fbdb`, artifact digest `sha256:ed04d5b34df094e2eb7f7db3f07b426619494b38987f1993c0faddca90d58f99`.

## Experiment 3: explicit two-state dynamics — rejected

A final bounded architecture test gave each specialist exactly one extra causal state:

`[current 24 h, previous 24 h, current - previous]`

with the same XGBoost recipe and the same cross-fitted evidence stack.

The causal transform passed its tests and found an exact 24-hour lag for 13,876 of 14,045 eligible rows; 169 rows had no allowed lag and remained missing rather than being synthetically imputed.

The two-state system improved older-block ranking:

- AUROC: `0.8688 -> 0.8762`
- AUPRC: `0.0767 -> 0.0988`
- Brier: `0.006348 -> 0.006330`

But it failed the preregistered promotion rule at the actual operational policy:

| model | older score TSS | 2023–2025 monitor TSS |
|---|---:|---:|
| static cross-fitted stack | **0.5120** | **0.2359** |
| two-state cross-fitted stack | 0.4619 | 0.2350 |

The paired POD80 score difference versus the static cross-fitted stack had median `-0.0501` with a wide 95% interval crossing zero. The monitor was essentially tied (`-0.0058` median paired difference). The temporal expansion therefore adds ranking information but makes the frozen threshold decision less robust. It is not promoted.

Artifact: GitHub Actions run `33985567816`, head `3f5013bd4723f35b6b090586d03004a2421b25d2`, artifact digest `sha256:546e49353aa0616a3b5183e10b1e7aa0421830482591c8ec7a71262ca2c4bea2`.

## What was taken from modern weather forecasting

The useful lesson from WeatherNext, GraphCast, NeuralGCM, ECMWF AIFS and Aurora is **system decomposition**, not model size:

- heterogeneous observations need specialized ingestion/representations;
- state/evidence processing should be separate from data ingestion;
- probabilistic ensembles matter for uncertain extreme events;
- calibration and observation quality should be explicit;
- richer sequence/spatial architectures are valuable only when the input actually contains resolved sequence/spatial state.

IRIS currently has a rare-event daily aggregate table. A billion-parameter weather backbone would therefore add inductive machinery without adding the spatial and temporal information those systems exploit.

## Stop rule for architecture work

Architecture expansion stops here on the current aggregate dataset.

The next performance gains must come from one of the following, in this order:

1. a verified training-only NEW-crossing cohort with source-latency and event-semantic provenance;
2. richer causal observations that genuinely contain resolved dynamics rather than duplicate engineered aggregates;
3. independent same-date operational comparison (including the preregistered NOAA/SWPC RSGA comparison);
4. an untouched final evaluation.

Only if richer verified observations become available should a new encoder/temporal processor be reconsidered.
