# IRIS / ISEF judge defense Q&A — episode-normalized causal SEP benchmark

**Purpose:** hard-question rehearsal notes grounded in the verified SEP-PRISM replay. These are prompts and answer ingredients, not a memorized script. Student researchers should answer in their own words and be able to derive every number they use.

## 1. Your central result is a lower TSS after changing the evaluation. Why should I believe you did not design the benchmark to make the model look worse?

**Answer ingredients:** The models, probabilities, thresholds and binary alerts are held fixed. Only the estimand changes. The benchmark also does not force all models downward: proton-free and elastic comparators respond differently. The hypothesis was falsifiable because a near-zero paired change would have counted against the proposed mechanism. The three primary contrasts and decision rule were frozen before replay score inspection.

## 2. What exactly is novel if onset, persistence and event-based validation already exist?

**Answer ingredients:** Those ideas are established and should be credited as such. The contribution tested here is their **combined quantitative effect on the same fixed SEP predictions**: mapped occurrence, per-episode-normalized occurrence and causal new onset are compared under one reproducible framework, with physical-event/quiet-block paired uncertainty. The empirical contribution is measuring how much the reported skill changes when the statistical unit is changed without changing the forecast.

## 3. Why are there 228 onset windows in the full SEP-PRISM audit but only 85 onset episodes in the matched replay?

**Answer ingredients:** They are different denominators and must never be mixed. The 228 count is an onset-window classification in the full 1986–2025 model-free table. The 85 units are distinct onset episodes that survive the frozen 2005–2025 replay score periods, unique physical mapping and matched bootstrap-population rules. The latter is the inferential positive-unit count.

## 4. Why use a physical-unit bootstrap instead of an ordinary row bootstrap, ANOVA or a conventional p-value?

**Answer ingredients:** Daily windows from one SEP episode are correlated, which is the very problem under study. Row resampling would pretend those windows are independent. The bootstrap resamples distinct onset episodes as positive units and seven-day quiet blocks as negative units, and reuses the same 10,000 draws across compared estimands. ANOVA is not the natural test for paired changes in a nonlinear skill statistic with clustered rare events. The reported quantities are paired bootstrap intervals, not classical p-values.

## 5. Why are the negative units Monday-anchored seven-day quiet blocks? Could your result depend on that arbitrary choice?

**Answer ingredients:** The block is a pragmatic way to reduce day-to-day dependence among negatives while preserving a fixed reproducible unit and avoiding one-day pseudo-independence. The exact choice should be treated as part of the frozen estimand, not a universal truth. A strong follow-up is a preregistered block-definition sensitivity analysis on development-only data, but this replay must not be retuned after score inspection.

## 6. Your joint XGBoost used 259 predictor-side variables. How do you know those were really available at forecast time?

**Answer ingredients:** We do **not** know that for every historical predictor, and we do not claim it. Explicit target/timestamp/`Future_*` exclusions prevent obvious leakage but do not establish publication-time availability, backfill or revision history. That is why the result is methodological historical evidence, not operational validation. The prospective contract now requires a field-level first-seen availability receipt and fails closed if any required feature is late or unverifiable.

## 7. Why was there no purge between every fit, threshold and score boundary if your targets cover the next 24 hours?

**Answer ingredients:** That is a real limitation. The audit checked the examined catalogue boundaries, but there was no programmed purge enforcing separation everywhere. This can make adjacent role boundaries less clean than an ideal prospective study. It does not invalidate the within-replay observation that fixed predictions receive different scores under different estimands, but it prevents treating the replay as final independent operational evidence. The prospective protocol freezes purged causal execution rules before outcome access.

## 8. Your joint XGBoost drops from 0.726 to 0.437 on onset. Does that prove proton history is harmful?

**Answer ingredients:** No. The matched joint-minus-proton-free onset contrast is about -0.041 with a 95% interval crossing zero, so a reliable model-rank reversal is not established. The supported claim is about **evaluation sensitivity**, not about a universal feature family or architecture being superior or harmful.

## 9. The onset confusion table has 330 false positives and an 88.95% FAR. Why is this useful at all?

**Answer ingredients:** That is exactly why the result is not presented as an operational model. The 88.95% value is the **false-alarm ratio** FP/(TP+FP), while the false-positive rate is about 4.51%. The model detects 41 of 85 onset episodes in the matched cohort, but the alert burden is too high for an operational-readiness claim. The model is a probe showing that evaluation definitions change what measured TSS represents.

## 10. Why should a long storm not count several times if an operator makes a new decision every day?

**Answer ingredients:** It can and sometimes should. Window-level occurrence answers a legitimate repeated-decision question, which is why it is retained. Episode normalization answers a different question: performance across physical SEP events when a long event should not have more positive statistical mass merely because it spans more windows. The recommendation is complementary reporting, not replacing all window-level metrics.

## 11. How strong is the statistical confirmation?

**Answer ingredients:** In the matched replay, the frozen primary paired intervals are all below zero: joint-XGBoost normalization about -0.104 [-0.146,-0.063], joint-XGBoost persistence removal about -0.183 [-0.247,-0.125], and past-proton-proxy onset-minus-mapped about -0.506 [-0.567,-0.444]. All 10,000 stored paired draws are negative for each of these three contrasts. That last fact is a bootstrap diagnostic, **not** a classical p-value or a family-wise 0.0001 significance guarantee.

## 12. Was the result independently reproduced, or are you trusting the same code twice?

**Answer ingredients:** The persisted replay artifact contains 45,348 prediction rows and shared bootstrap arrays. A verifier separate from the benchmark runner recomputed the evidence, and a second minimal recheck script now reconstructs the three primary contrasts directly from the persisted matched predictions and stored physical-unit draw tensors without importing the runner. The artifact SHA-256 matches the frozen receipt. This establishes computational reproducibility of the stored replay, not scientific independence of the already-exposed cohort.

## 13. You corrected a result after the first successful run. How do I know that was not result shopping?

**Answer ingredients:** The independent audit identified an OOF threshold-summary bug and a labeling ambiguity between descriptive and inferential cohorts. The correction was restricted to the evidence/result layer: no refit, no feature change, no threshold reselection, no hyperparameter search and no protected-outcome access. Legacy uncorrected files were preserved. The correction boundary is part of the audit trail.

## 14. Why is the public historical replay not an independent validation if the upstream data are external?

**Answer ingredients:** External provenance and statistical independence are different. The exact public source hashes had already been inspected during development, so they cannot be relabeled as unseen final evidence. The correct description is preregistered methodological confirmation on development-exposed public data.

## 15. What would falsify the project’s main claim?

**Answer ingredients:** If fixed forecasts retained essentially the same skill after episode normalization and onset restriction, with paired contrast intervals centered around zero, then multiplicity and persistence would not materially change the measured score. A future custodian-controlled, issue-time-causal prospective study could also fail the information floor or return a contrast including zero; that outcome must be accepted without redesigning the same cohort.

## 16. What is the single biggest remaining scientific weakness?

**Answer ingredients:** Evidence independence and causal feature availability. The 85-event replay is much stronger than the earlier five-onset study, but it is still historical and development-exposed, and not every historical predictor has a publication-time receipt. The next decisive step is the already-frozen prospective custodian protocol, not more tuning on the same replay.

## 17. Why is this scientifically useful if it does not improve forecast accuracy?

**Answer ingredients:** A benchmark changes what future model improvements are required to demonstrate. If a model claims pre-onset warning, it should not receive indistinguishable credit for persistence recognition or extra weight from a long event. Measurement methodology can improve the reliability and comparability of a field even without proposing a new classifier.

## 18. Which number should a judge remember?

**Answer ingredients:** Use one mechanism number and one replay line. `614 mapped positive windows / 257 physical episodes = 2.389x`, then `joint XGBoost 0.726 -> 0.621 -> 0.437` on the matched occurrence -> normalized -> onset evaluation. Immediately add that this is historical methodological evidence, not operational performance.

## 19. Why not claim SEPNET is overestimated or wrong?

**Answer ingredients:** The public data audit establishes properties of pinned rolling tables under the audited event semantics. It does not reconstruct every final published SEPVAL condition or prove a published score is invalid. The project therefore critiques an evaluation possibility and proposes complementary metrics rather than accusing a paper.

## 20. What is your prospective validation plan, and why has it not been run yet?

**Answer ingredients:** Protected post-2025 outcomes remain sealed. Before execution, each model input must pass a frozen issue-time availability contract, model/rule and schema hashes must be frozen, and an independent custodian must execute the aggregate-only evaluation. The protocol requires at least 50 distinct onset episodes and 500 quiet blocks; if that information floor is not met, the correct result is `INSUFFICIENT_CONFIRMATORY_INFORMATION`, not a redesigned analysis.

## Numbers to know cold

- SEP-PRISM full table: **14,464 windows, 650 positives**
- Uniquely mapped positives / episodes: **614 / 257 = 2.389x**
- Full-table persistence / onset windows: **418 / 228**
- Fixed replay score issues: **7,558**
- Matched inference: **85 onset episodes + 1,080 quiet blocks**
- Joint XGB matched TSS: **0.726 -> 0.621 -> 0.437**
- Joint XGB normalization contrast: **-0.104 [-0.146,-0.063]**
- Joint XGB persistence-removal contrast: **-0.183 [-0.247,-0.125]**
- Past-proton persistence contrast: **-0.506 [-0.567,-0.444]**
- Joint-XGB onset confusion: **TP 41, FN 44, FP 330, TN 6,984**
- Sensitivity: **48.24%**
- False-alarm ratio: **88.95%**; false-positive rate: **~4.51%**
- Joint-minus-no-proton onset: **about -0.041, interval crosses zero**
- Model-free artifact SHA-256: `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`
- Fixed-model artifact SHA-256: `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
