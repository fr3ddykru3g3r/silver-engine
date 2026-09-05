# IRIS-SEP model architecture decision — 2026-09-06

## Decision

Promote **cross-fitted specialist evidence stacking** as the current development architecture. Do not promote the hard-anchored residual fusion model. Do not scale the compact gated neural model merely by adding layers or attention.

This remains development-only. The score and 2023–2025 monitor have already been inspected and cannot become fresh final evidence.

## Why the old compact fusion is structurally wrong for the evidence

`IRISSEPTabularModel` encodes modalities separately but then makes their embeddings compete through a softmax-weighted average. The empirical diagnostics show that modality separation helps but that the relative importance of solar, XRS and proton evidence can change sharply. A convex representation average is therefore an unnecessary bottleneck.

The successful late-fusion system instead lets each modality make a complete forecast and combines only the resulting evidence.

## Experiment 1: hard solar anchor — rejected

Preregistered architecture:

`logit(p) = logit(p_solar) + b + w_xrs * e_xrs + w_proton * e_proton`

with nonnegative bounded context residuals.

Result: the fitted context weights collapsed (`w_xrs≈0.0368`, `w_proton≈0.0081`). On the operational POD>=0.8/min-FAR policy, the 2023–2025 monitor paired TSS difference versus the existing late-fusion model was about `-0.255`, with the complete 95% interval below zero. The solar coefficient must not be hard-wired to one.

Artifact: GitHub Actions run `33984943106`, head `8b008f915fa635e602265e52ad68cbddaec3c380`, artifact digest `sha256:1f7e6318bb797525160433c90760bec7692a25932420cefcdfdc5776f070db38`.

## Experiment 2: cross-fitted evidence stack — promoted as development architecture

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
| cross-fitted stack | 0.5120 | 0.2359 |

The cross-fitted stack therefore preserves the older operational score while improving the later development monitor. Its paired monitor advantage over the previous late-fusion system is positive in the point estimate (`~+0.0425`) but its 95% interval still crosses zero, so superiority is not established.

Probability quality also improved on the 2023–2025 monitor:

- AUROC: `0.6724 -> 0.6825`
- Brier: `0.03464 -> 0.03107`
- ECE: `0.02680 -> 0.01098`

AUPRC was slightly lower (`0.1088 -> 0.1058`), which must remain disclosed.

Artifact: GitHub Actions run `33985243824`, head `5d90a177462401d1486891e06c6fa6794ea4fbdb`, artifact digest `sha256:ed04d5b34df094e2eb7f7db3f07b426619494b38987f1993c0faddca90d58f99`.

## Architecture principle taken from modern weather forecasting

Do not copy model scale. Copy system structure:

- heterogeneous observations are handled by specialized ingestion/representations;
- a separate processor represents state/evidence evolution;
- outputs are probabilistic and ensemble-based;
- observation quality and forecast validity are explicit rather than hidden in one monolithic network.

For the current IRIS daily aggregate dataset, a graph network, diffusion model or billion-parameter Transformer is not justified by the information interface.

## Next bounded structural experiment

Add one explicit causal state transition to each specialist:

`[current 24 h state, previous 24 h state, current - previous]`

Then retain the same cross-fitted evidence stack. This tests whether temporal evolution adds information without introducing an LSTM/Transformer or changing the target/evaluation protocol.

If this does not beat the static specialist stack, stop architecture expansion until richer verified observations are available.
