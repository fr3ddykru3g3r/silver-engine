# Student ownership / defense checklist — 2026-09-11

This file is a **mastery test**, not submission prose. Before presenting the project, each student researcher should be able to answer every item without reading a generated answer.

## Level 1 — explain the physics

- [ ] Define a solar energetic particle event in plain language.
- [ ] Explain what `>10 MeV` means.
- [ ] Explain what `10 pfu` means and why it is used here.
- [ ] Draw one physical SEP episode as a time interval.
- [ ] Place three daily issue times around the same episode and show which are pre-onset, already active and negative.
- [ ] Explain why persistence recognition can be operationally useful while still being a different scientific task from pre-onset warning.

## Level 2 — explain the benchmark

- [ ] Define `MAPPED_OCCURRENCE` without using code words.
- [ ] Define `EPISODE_NORMALIZED_OCCURRENCE` and derive why each of `m` windows receives weight `1/m`.
- [ ] Define `NEW_ONSET_CAUSAL` and explain why already-active cases receive no onset-positive credit.
- [ ] Explain why ambiguous multi-episode positives are not arbitrarily assigned to one bootstrap episode.
- [ ] Explain why the same forecast probabilities and thresholds are retained across evaluation views.
- [ ] State what observation would have falsified the proposed evaluation effect.

## Level 3 — know the denominators

Write these from memory, then verify them against the frozen receipts:

- [ ] `14,464` total SEP-PRISM daily windows.
- [ ] `650` stored positive windows.
- [ ] `614` uniquely mapped positive windows.
- [ ] `257` represented physical episodes.
- [ ] `614/257 = 2.389` episode multiplicity factor.
- [ ] `418` already-active persistence windows.
- [ ] `228` onset windows in the full table.
- [ ] `4` onset-state ambiguity windows.
- [ ] `36` separate positives overlapping multiple episodes.
- [ ] `7,558` unique fixed-replay score issues.
- [ ] `85` matched onset episode units.
- [ ] `1,080` matched quiet blocks.
- [ ] `10,000` shared bootstrap draws.

Be able to explain why `228 onset windows` and `85 matched onset episodes` are not contradictory.

## Level 4 — statistics

- [ ] Define TSS as `sensitivity - false-positive rate`.
- [ ] Explain why raw accuracy is weak for rare events.
- [ ] Explain why a row bootstrap would recreate the pseudo-independence problem.
- [ ] Explain why positives are resampled by physical episode.
- [ ] Explain why negatives are resampled by seven-day quiet block.
- [ ] Explain why the draws are shared across compared estimands.
- [ ] Explain what a 95% paired bootstrap interval means here.
- [ ] Explain why “all 10,000 draws are negative” is not automatically a conventional `p < 0.0001` claim.
- [ ] Explain why ANOVA is not the primary inference method for this clustered nonlinear skill statistic.

## Level 5 — headline result

Know cold:

- [ ] joint XGBoost matched TSS: `0.726 -> 0.621 -> 0.437`.
- [ ] normalization contrast: about `-0.104 [-0.146,-0.063]`.
- [ ] persistence-removal contrast: about `-0.183 [-0.247,-0.125]`.
- [ ] past-proton proxy onset-minus-mapped contrast: about `-0.506 [-0.567,-0.444]`.
- [ ] joint-minus-no-proton onset contrast: about `-0.041` with interval crossing zero.

Then answer: which of these support an evaluation-effect claim, and which do **not** support a model-superiority claim?

## Level 6 — confusion matrix language

For joint-XGBoost onset:

- [ ] TP = 41.
- [ ] FN = 44.
- [ ] FP = 330.
- [ ] TN = 6,984.
- [ ] sensitivity = 48.24%.
- [ ] false-alarm ratio = 88.95%.
- [ ] false-positive rate ≈ 4.51%.

Be able to derive why FAR and FPR have different denominators.

## Level 7 — reproduce the computation

- [ ] Identify the frozen replay artifact and SHA-256.
- [ ] Explain what is persisted in `predictions.csv`.
- [ ] Explain what the shared bootstrap tensor stores.
- [ ] Run or explain `verify_sep_prism_external_model_replay_v1.py`.
- [ ] Run or explain `recheck_sep_prism_primary_contrasts_v1.py`.
- [ ] Explain why a separate verifier is better evidence than merely rerunning the same scoring function.
- [ ] Explain the earlier result-layer correction and why it did not refit models or reselect thresholds.

## Level 8 — limitations without defensiveness

For each limitation, give a one-sentence consequence:

- [ ] historical source hashes were already development-exposed;
- [ ] publication-time availability is not proven for all 259 historical predictors;
- [ ] there was no universal programmed temporal purge at every role boundary;
- [ ] historical onset eligibility comes from retrospective catalogue boundaries;
- [ ] complete fitted-model and threshold-block probability objects are absent from the replay archive;
- [ ] onset alert burden does not support operational readiness.

## Level 9 — prospective protocol

- [ ] Explain why protected outcomes remain sealed.
- [ ] Explain the 00:00 UTC issue clock.
- [ ] Explain why a public predictor snapshot must be captured before the issue time.
- [ ] Explain the 15-minute maximum age frozen for the deterministic proton-state input.
- [ ] Explain `ABSTAIN` as a scientific safeguard rather than a software failure.
- [ ] Explain the minimum information floor: 50 onset episodes + 500 quiet blocks.
- [ ] Explain what `INSUFFICIENT_CONFIRMATORY_INFORMATION` means.
- [ ] Explain why the development side must not inspect labels to decide when/how to redesign the study.

## Level 10 — novelty and literature

Answer without saying “nobody has ever done this”:

- [ ] What parts are established concepts?
- [ ] What exact combination did this project test?
- [ ] Why is a benchmark contribution scientifically useful even if it does not improve model accuracy?
- [ ] Why does the project not claim the SEPNET/SEPVAL final published result is wrong?
- [ ] What would another research group need to report to make its “new SEP forecasting” claim directly comparable under this framework?

## Graduation rule

A student is presentation-ready only when they can:

1. draw the evaluation mechanism from a blank page;
2. derive the 1/m episode weighting;
3. explain TSS/FAR/FPR correctly;
4. state the headline denominators without mixing cohorts;
5. describe the strongest result and strongest limitation in under 30 seconds each;
6. explain the prospective falsification path without promising its result.
