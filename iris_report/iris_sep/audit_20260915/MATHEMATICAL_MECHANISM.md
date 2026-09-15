# Exact identities and their limits

All expectations and covariances below are finite-population quantities, with denominator K (ddof=0), unless a different base measure is stated. These are derivations using established reweighting mathematics, not claims of a new theorem.

## Multiplicity

Let K>0 represented positive episodes have n_i>0 admitted positive windows. Fixed alerts a_ij lie in {0,1}; r_i=(sum_j a_ij)/n_i. With uniform probability 1/K over episodes,

POD_row = sum(n_i r_i)/sum(n_i) = E[nr]/E[n].
POD_episode = E[r].

Subtracting gives

**POD_row − POD_episode = Cov(n,r)/E[n].**

Proof: E[nr]=E[n]E[r]+Cov(n,r). No stochastic independence assumption is needed. For TSS=POD−FPR the same difference holds **only when the negative measure and alerts are unchanged**. Otherwise add FPR_episode−FPR_row. Equal negative counts alone do not establish equal negative measures.

The gap is positive when longer represented episodes are more detectable on average, negative for the reverse, and zero iff covariance is zero. Independence is sufficient but unnecessary: n=(1,2,3), r=(1,0,1) has zero covariance with dependence. Constant n or constant r gives zero gap. K=1 gives zero covariance. K=0, zero total positive mass or zero negative mass makes relevant scores undefined; do not turn undefined sensitivity/TSS into zero. Using sample covariance divides by K−1 and requires a compensating (K−1)/K factor.

“Episode-normalized” here means average fraction of an episode's windows alerted. It is not max_j(a_ij), or the probability that an event ever receives an alert. Joint-model values in the admitted cohort are respectively 0.666246 and 0.800000; onset sensitivity is 0.482353.

## Arbitrary weighting and continuous estimands

For any base probability b_i>0 summing to one and nonnegative reweight w_i with E_b[w]>0,

E_bw[r] − E_b[r] = Cov_b(w,r)/E_b[w],

where E_bw uses masses b_i w_i/E_b[w]. This applies to any bounded episode summary and does not require integer weights. It connects uniform-over-event sampling with size-biased sampling of a random positive window. Inverse n weights on positive rows recover uniform event mass; they are not automatically a better estimand for every operational decision.

Along weights n_i^(1−alpha), alpha=0 is row weighting and alpha=1 is equal-event weighting. Differentiating the ratio yields

d E_alpha[r]/d alpha = −Cov_alpha(log n,r).

A monotone decline requires the covariance sign along the entire path, not just at alpha=0 or 1. Tests of several alpha values are descriptive, not proof of monotonicity for every alpha.

## Bounds

Cauchy–Schwarz gives |gap| ≤ CV(n) SD(r) ≤ CV(n)/2 for 0≤r≤1.

A tighter sharp distributional bound for fixed n is

|gap| ≤ (max r−min r) × (1/2) sum_i |n_i/sum n−1/K|.

The range-one version is attained by setting r=1 on the positive part of the weight difference and zero on its negative part. More multiplicity variance permits a larger discrepancy; it does not force one. The admitted cohort's total-variation bound is 0.237504, while the joint-model gap is 0.105327.

## Persistence decomposition

Assume each admitted episode contains exactly one onset window, with alert u_i, and n_i−1 active windows with mean alert v_i. Define the product (n_i−1)v_i=0 for singleton episodes. Then

r_i = u_i/n_i + (1−1/n_i)v_i,
E[r]−E[u] = E[(1−1/n_i)(v_i−u_i)].

Consequently

**TSS_row−TSS_onset = Cov(n,r)/E[n] + E[(1−1/n)(v−u)] + FPR_onset−FPR_row.**

The admitted matched cohort has a zero final term. Joint contributions are 0.105327+0.183894=0.289221. The second term can be negative; it is an algebraic contrast, not automatically a causal benefit from observing an active storm. Removing active windows changes the target population.

For k_i>0 onset windows per episode, let u_i be their mean alert, q_i=k_i/n_i, and v_i the mean over the remaining windows. Row-weighted onset sensitivity is E[k u]/E[k]. The decomposition becomes

POD_row−POD_onset = Cov(n,r)/E[n] + E[(1−q)(v−u)] − Cov(k,u)/E[k].

Episodes with k_i=0 require an explicit population-selection term or a restriction to represented-onset episodes. Do not apply the one-onset identity to a different cohort without checking this assumption. No new catalog definition was selected to maximize these contrasts.

## Ranking

For two models A and B under a common negative measure, D_row=D_episode+G_A−G_B, where G is each model's multiplicity gap. A strict rank reversal occurs iff D_episode(D_episode+G_A−G_B)<0. The equality case is a tie. This is necessary and sufficient for these observed point estimates, not statistical evidence for a population ranking.

The observed onset ordering of joint and no-proton XGBoost differs from occurrence ordering, but their paired onset contrast includes zero. Therefore a reliably superior onset model is not demonstrated.

## Brier and proper scoring rules

Let L_i be mean positive-row squared error within event i, L_minus the unchanged negative mean loss, pi_row=Nplus/(Nplus+Nminus), and pi_episode=K/(K+Nminus). Then

BS_row−BS_episode = pi_row Cov(n,L)/E[n] + (pi_row−pi_episode)(E[L]−L_minus).

This separates positive reweighting from class-prior change. Joint values are −0.003526+0.008901=0.005375. A smaller normalized Brier score is not evidence of improved forecast calibration. Weighting by observed outcome generally changes the optimal probability away from the original population probability; assess proper scoring against the declared target distribution. The same linear-loss derivation applies to other integrable per-row losses, with appropriate endpoint treatment for logarithmic loss.

## AUROC and HSS

For fixed negative scores, let phi_i be the mean of pairwise positive-vs-negative wins (ties count 1/2) in episode i. AUROC_row−AUROC_episode=Cov(n,phi)/E[n]. If negative weighting also changes, include its separate change; the simple one-sided identity is insufficient. The admitted joint values are 0.944546 and 0.917332. This is occurrence AUROC, not an onset alert metric.

With prevalence pi, sensitivity s, FPR f and predicted-positive rate q=pi*s+(1−pi)*f,

HSS = 2*pi*(1−pi)*(s−f) / [pi+(1−2*pi)*q].

HSS is nonlinear and prevalence-dependent. Reweighting alters both s and pi; a single covariance term cannot generally describe its difference. Degenerate denominators need explicit handling.

## Sampling precision

(sum n)^2/sum(n^2)=61.3096 is a Kish weight-concentration diagnostic for these episodes, not an empirically established effective sample size. Variance depends on within/between-episode dependence, temporal nonstationarity and the target statistic. A design-effect approximation 1+(m−1)rho requires equal cluster sizes and a specified intraclass model; it is not justified here by the counts alone.

Paired cluster bootstrap keeps an event's windows together and uses the same sampled units for both sides of each contrast. Positive episodes and quiet blocks address different classes. Our primary contrasts cancel the identical FPR, so changing only quiet-block length cannot test their positive-event dependence. Calendar-bin checks therefore group nearby positive episodes. Bootstrap uncertainty is conditional on frozen forecasts and does not include the full development/model-selection process. All draws having one sign is not a calibrated null p-value: no null resampling test was performed.
