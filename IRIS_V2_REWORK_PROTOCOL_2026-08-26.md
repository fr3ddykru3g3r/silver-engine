# IRIS v2 — Physics-to-Utility Transfer Protocol

Date frozen: 2026-08-26

## Why v2 exists

The v1 experiment is retained unchanged as an exploratory/pilot phase. Its already-opened locked test must not be used for further hyperparameter selection. V1 established three useful facts: (1) Hale/Joy population regularization moved its intended physical-distribution metric, (2) the present scalar PIL-gradient regularizer did not reproduce the real strong-gradient PIL distribution, and (3) synthetic augmentation changed downstream operating characteristics but did not yet establish a statistically significant TSS improvement over real-only or duplicated-positive controls.

V2 therefore does **not** optimize for a larger score on the already-seen v1 test. It changes the scientific question from “can synthetic data improve TSS?” to a mechanistic question:

> **Which measurable magnetic structures must a synthetic LOS magnetogram reproduce before it transfers useful information to flare forecasting?**

The primary scientific object is the relationship between **physical-distribution fidelity** and **downstream utility**, not a single leaderboard number.

## Main hypotheses

H1 — A physics regularizer must first pass a manipulation check: the targeted physical distribution in synthetic positive magnetograms must become significantly closer to real training positives.

H2 — Improvements in physical fidelity that matter for flare production will predict downstream utility better than generic image similarity alone.

H3 — Selectively destroying an otherwise preserved magnetic structure will reduce transfer if that structure is causally useful to the downstream learner.

## V1 is frozen

No v1 test threshold, architecture, reward, diffusion hyperparameter, physics coefficient, crop rule, or augmentation count may be changed in response to v1 test performance and then re-evaluated on the same test as a confirmatory result. V1 results remain reported as Phase-I exploratory evidence.

## V2 data protocol

- Same definitive HMI SHARP CEA LOS source and M1+ / forward-24-hour onset label definition.
- Same connected HARPNUM↔NOAA physical-region grouping.
- No physical region may cross train/evaluation boundaries.
- Unresolved M1+ attribution windows remain censored rather than silently negative.
- QUALITY==0 and the predeclared central-meridian criterion remain required.
- Every crop records unsigned-flux retention. Crops that truncate an unacceptable fraction of real unsigned flux are flagged rather than silently treated as equivalent.

### Development/evaluation separation

Because the original terminal test has already been viewed, v2 uses two forms of replication:

1. **Frozen rolling-origin backtest:** all v2 code and hyperparameters are frozen before running a set of calendar-time outer folds. No per-fold tuning is allowed. Each fold trains only on earlier active regions and evaluates on later active regions. Results are aggregated across folds with connected-region resampling.
2. **Prospective blind track:** once v2 is frozen, predictions are timestamped before their GOES 24-hour outcomes are known and accumulated through the submission period. This is a small-power but genuinely prospective confirmation track.

The already-opened v1 terminal test is never rebranded as a fresh v2 test.

## Generator rework

### 1. Always-on positive physics batch

V1 applied physics only when a random diffusion batch happened to contain >=2 positive examples at sufficiently low diffusion timestep. With batch size 24, positive sampling near 0.4, and a 0.25 low-noise timestep fraction, the expected number of eligible positive physics examples was only about 2.4 per step, so many updates received no physics gradient at all.

V2 uses a separate positive-only physics batch every generator update. Physics timesteps are explicitly drawn from the predeclared low-noise range. Thus every constrained update actually receives the intended physical gradient.

### 2. Strong-field-aware Hale/Joy descriptor

V1 centroid weights used `softplus(B/T)` and `softplus(-B/T)`. At B=0 both are nonzero, allowing large quiet-background areas to contaminate polarity centroids.

V2 uses thresholded smooth polarity weights centered on a strong-field threshold before centroid calculation. The descriptor matches hemisphere-conditioned orientation and separation distributions without hard-coding image-axis polarity convention.

### 3. Replace one-number PIL averaging with a distributional strong-PIL descriptor

V1 compressed each magnetogram to one contact-weighted mean gradient. This can reward many mediocre-gradient pixels while missing the high-gradient tail that makes strong-field PILs physically distinctive.

V2 remains a **PIL-gradient** constraint, but represents that one physical concept with a small vector of differentiable statistics measured only near opposite-polarity strong-field contact:

- log mean |∇B| along the soft PIL;
- log RMS |∇B| along the soft PIL;
- log high-tail mean |∇B|, using a smooth high-gradient gate;
- smooth exceedance fractions at predeclared G/Mm thresholds.

The generator matches multiple quantiles of these descriptors between generated and real positive batches. This prevents a generator from passing by matching only the mean while missing the real distribution tail.

The hard diagnostic implementation remains independent of the differentiable training proxy.

### 4. Multi-scale contact

Strong opposite polarities can be separated by a narrow weak-field corridor. V1 used only immediate four-neighbour contact. V2 evaluates soft opposite-polarity contact at predeclared physical radii, expressed in Mm and converted to pixels after fixed-FOV resampling.

### 5. Physics coefficient selection is train-only

For each constraint, a small predeclared coefficient grid is selected **only** using training-set manipulation checks:

- targeted physical distance must improve by a minimum predeclared fraction versus the unconstrained generator;
- generic reconstruction/generation diagnostics must not collapse;
- synthetic diversity must stay above a predeclared floor;
- no downstream test metric is used.

The smallest coefficient that passes the physical gate is selected.

## Constraint matrix

Primary generator arms:

- BASE: no magnetic-structure regularizer.
- HJ: strong-field Hale/Joy population-distribution regularizer.
- PIL: v2 multi-scale strong-PIL-gradient distribution regularizer.
- HJ+PIL: both.

Mechanistic destruction controls, applied to generated samples with matched counts:

- HJ-destroyed: polarity geometry perturbed while preserving field-value histogram as closely as possible.
- PIL-blurred: local smoothing targeted around the PIL to destroy high-gradient structure while preserving coarse bipole geometry and approximate unsigned flux.
- spatial-phase/shuffle control: preserves coarse marginal statistics but destroys spatial magnetic organization.

These controls ask whether a structure is merely correlated with “realism” or actually carries transferable information.

## Downstream models

### Primary

Keep a compact static LOS CNN as the primary utility probe. A deliberately simple probe makes the generator the experimental variable.

### Secondary architecture replication

Use the already-implemented temporal CNN-BiLSTM as a secondary comparator. It is not used to tune generator physics. A synthetic-data conclusion is stronger if the sign of transfer is consistent across both downstream architectures.

CDR remains a secondary methodological comparator, not the central project narrative.

## Primary metrics

### Physical manipulation

For each arm, report train/validation-only physical distances to real positives and bootstrap uncertainty by connected region.

### Downstream utility

Primary: TSS.
Secondary: HSS2, AUPRC, AUROC, Brier/BSS, recall, FPR, precision, calibration.

All pairwise utility comparisons use the same evaluation observations and a paired connected-region bootstrap.

The duplicated-positive control is mandatory. Synthetic augmentation is not considered useful merely because it beats real-only; it must be compared with the same extra exposure supplied by duplicated real positives.

## Physics-to-utility analysis

For each generator arm, coefficient level, seed, and outer fold define:

- `DeltaF_HJ`: improvement in Hale/Joy physical-distribution fidelity relative to BASE;
- `DeltaF_PIL`: improvement in strong-PIL-gradient fidelity relative to BASE;
- `DeltaU`: downstream TSS change relative to duplicated-positive augmentation;
- generic image-fidelity diagnostics and diversity diagnostics.

The central analysis estimates whether physical-fidelity improvement predicts utility across folds/seeds/arms, rather than cherry-picking the highest-scoring arm. Report rank correlation and a predeclared regression/mixed-effects analysis with fold and seed effects where sample size permits.

## Predeclared success criteria

A constraint is called **successfully imposed** only if its physical manipulation check passes.

A synthetic arm is called **utility-improving** only if:

1. it passes its physical manipulation check;
2. its paired `Delta TSS` versus duplicated-positive control is >0;
3. the connected-region 95% interval excludes 0 in the aggregate frozen evaluation; and
4. the direction is not driven by a catastrophic precision/FPR tradeoff hidden by TSS.

If a physical manipulation succeeds but utility does not improve, that is a valid negative scientific result. If manipulation fails, no conclusion about the physical structure's usefulness is permitted.

## Why this is competition-level

The novelty is not “we generated solar magnetograms.” That has prior art. The contribution is a falsifiable, constraint-by-constraint test of **which magnetic structures must be reproduced for synthetic observations to transfer useful information**, with physical manipulation checks, destructive controls, exposure-matched baselines, connected-region chronology, and prospective confirmation.
