# Small bounded training-side batch proposal

This is a planning proposal for development selection only, not yet an
executable preregistration. It is not a request to open the locked test and it
does not turn the already inspected `validation_monitor` into new evidence.
Execution is `NOT_RUN` until a verified new-crossing training-only cohort,
latency manifest, and reproduced published comparator are supplied.

## Objective

Estimate whether two physically motivated, causal feature additions improve the
existing compact IRIS primary and the corrected V5 dense adapter on a future
verified new-crossing training-only cohort. Include climatology, elastic-net
logistic regression, and XGBoost as fixed classical references. The target
remains `new_sep_10mev_10pfu_within_24h`; no secondary head is enabled.

## Folds and roles

Use four chronological inner folds from `role=train` rows only. Do not carve
folds from `validation_monitor`, `validation_calibration`, or
`validation_threshold`. Each fold must be episode-disjoint, keep complete SEP
episodes together, and apply a strict 24-hour purge across train/validation
boundaries. Fit imputation, scaling, feature selection, and any feature map on
the inner-train portion only. Inner validation is used for model selection;
calibration and operating threshold remain separate inner roles. The existing
outer roles remain untouched and the already inspected `validation_monitor` is
not reread as fresh evidence.

## Arms

| Arm | Change from frozen baseline | Purpose |
|---|---|---|
| A0 | Fixed reference suite: climatology, elastic-net, XGBoost, existing compact primary, and corrected V5 adapter with current settings | References; V5 remains unverified |
| A1 | Add causal historical >10 MeV proton summaries and observed/freshness masks, if present in the audited cohort | Test the strongest repeated physical signal |
| A2 | A1 plus causal GOES XRS summaries and freshness masks | Test radiative context conditional on particle history |
| A3 | A2 plus one predeclared connectivity proxy using only authoritative source geometry already present in the cohort | Test transport geometry; run only if bridge/latency receipt is complete |

For every added variable, require an explicit issue-time publication timestamp,
units, missingness policy, and provenance row. An arm with unavailable or
ambiguous inputs is `NOT_RUN`, never backfilled with future data or excluded
quietly. AIA/HMI image fusion and approximate AARP/HMI joins remain disabled.

## Fixed limits

- Exactly four inner folds from `role=train`, five seeds (`7, 13, 26, 42, 73`),
  and the same existing optimizer, batch size, maximum steps/epochs, and
  preprocessing policy across neural arms.
- Freeze the existing baseline settings. No new hyperparameter grid is defined
  in this proposal; therefore no hyperparameter search is permitted.
- No Optuna, architecture search, random split, post-hoc gain, or monitor-driven
  retuning.
- No use of locked-test rows, outcomes, predictions, or thresholds.

## Selection rule

For each neural feature arm, aggregate inner-fold predictions by seed median and
compare first against the compact-primary member of A0. Select the simplest arm
whose paired episode/quiet-block bootstrap interval for TSS versus that compact
reference has lower 95% bound above zero in the predeclared inner aggregate, while also
meeting all of:

- no increase in matched-detection FAR at POD 0.8;
- Brier increase no larger than 0.01 and ECE increase no larger than 0.02;
- no median warning-lead decrease over one hour when lead is defined;
- no fold with an integrity, freshness, or role-boundary failure.

If no arm passes, retain A0 and report the negative result. This rule mirrors
the frozen parent policy; it cannot be relaxed after scores are seen. The
candidate is frozen before any independent evaluation and before any publisher
comparison request is scored.

## Expected value and falsifiers

The expected benefit ordering is A1 > A2 > A3: historical proton context has
the clearest prior signal; XRS adds a plausible complementary eruption proxy;
connectivity is physically attractive but data-riskier. A1 is the preferred
first experiment because it adds information without architectural complexity.

The proposal is falsified if A1 does not improve paired inner-fold TSS, if its
FAR/calibration trade-off fails the limits, or if the source-latency manifest
cannot prove availability. A positive inner result is only a selection result;
it does not establish SEPVAL performance, superiority, or operational readiness.

## Cost and risk register

| Candidate | Expected benefit | Data requirement | Leakage / validity risk | Compute cost | Falsifiable check |
|---|---|---|---|---|---|
| A1 proton context | High | Causal GOES proton history, publication times, units, missingness | Already-enhanced values or revised event products can leak the outcome | Low to moderate; summary features only | Paired inner-fold TSS, FAR, Brier, ECE, and freshness audits |
| A2 XRS context | Moderate | Causal XRS history and latency manifest | Feed revisions or post-issue XRS values | Low to moderate | Incremental A2-versus-A1 interval under the same folds |
| A3 connectivity proxy | Moderate but uncertain | Authoritative source geometry and exact identity bridge | Approximate joins and cross-product maps can create false skill | Moderate | Held-out cycle/product stability and bridge audit |
| V5 dense adapter | Reference only | Verified new-crossing training data and published comparator mapping | Current target/cohort equivalence is unresolved | Moderate | Reproduction receipt before comparator use |

Before this is executable, the team must add the exact source feature names,
units, latency cutoffs, fold manifest rule, and immutable configuration hash.
