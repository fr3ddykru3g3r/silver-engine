# Are We Forecasting a New Solar Radiation Storm?
## An Episode-Normalized Causal Benchmark for Solar Energetic Particle Prediction

**Working research-paper draft — 2026-09-10**  
**Submission boundary:** do not add school name, city, or state.  
**Evidence status:** independently verified development/methodology evidence; protected final outcomes remain sealed.

## Abstract

Solar energetic particle (SEP) forecasting systems are commonly evaluated on fixed 24-hour windows. This study asks whether such window-level scores can mix three scientifically different abilities: forecasting a new radiation storm, recognizing that a storm is already active, and repeatedly counting one long physical event. We developed an episode-normalized causal benchmark for operational >10 MeV proton events crossing 10 pfu. The benchmark separates already-active persistence from new onset, maps positive windows to physical SEP episodes when possible, assigns each uniquely mapped episode total positive statistical weight one, and applies shared physical-unit bootstrap resampling. A pinned public benchmark audit contained 11,773 windows; 610 uniquely mapped positive windows represented only 256 physical episodes, while 411 reconstructed positive windows were already-active persistence cases and 227 were new-onset cases. Fixed climatology, persistence, elastic-net and XGBoost models were then evaluated on an exposed 2014–2017 expanding out-of-fold development cohort. A simple current-proton-active diagnostic obtained True Skill Statistic (TSS) 0.556 under ordinary occurrence scoring but TSS 0.000 for new onset. Joint XGBoost changed from TSS 0.657 under standard occurrence scoring to 0.443 after episode normalization and 0.043 for new onset. On the uniquely mapped physical-episode sensitivity cohort, the joint-XGBoost multiplicity-only TSS shift was -0.133 with 95% bootstrap interval [-0.225,-0.041], while persistence exclusion shifted TSS by -0.400 [-0.800,-0.125]. The internal onset arm contained only five events, so model-ranking conclusions remain underpowered. These results show that evaluation-unit and causal-eligibility choices can materially change measured SEP forecasting skill and support reporting onset-specific, physical-event-normalized performance alongside conventional window scores.

## 1. Introduction

Solar energetic particles are high-energy charged particles associated with solar activity. Operationally important SEP events can increase the flux of >10 MeV protons above 10 proton flux units (pfu), the threshold used by NOAA for the S1 solar radiation storm level. Such events matter because energetic particles can affect spacecraft electronics, satellite operations, communications, aviation at high latitudes, and astronaut radiation exposure.

Because SEP events are rare and complex, machine learning is increasingly used to combine information from solar flares, coronal mass ejections, active-region magnetic fields, soft X-ray measurements and historical proton flux. A common approach is to summarize observations over a fixed historical interval and predict whether an SEP event occurs during the following 24 hours. Recent systems such as SEPNET and the SEP-PRISM dataset use this type of 24-hour supervised-learning representation.

However, a fixed-window representation creates an evaluation problem that is separate from model architecture. A single physical SEP episode may last long enough to overlap several forecast windows. If every positive window receives full statistical weight, one long event may influence the final score several times while a short event influences it once. In addition, a forecast issue time can occur after proton flux has already crossed the operational threshold. A predictor using current proton history may then receive credit for recognizing an event already in progress rather than forecasting its onset.

These cases answer different scientific questions. A system can be useful at recognizing persistence, but persistence recognition should not automatically be interpreted as pre-onset warning skill. Likewise, repeated windows may be legitimate prediction opportunities, but window-level scoring does not tell us how much performance is driven by the number of windows per physical event.

This study therefore asks:

> **How much do repeated representation of physical SEP episodes and already-active persistence change the apparent skill and ranking of 24-hour SEP forecasting systems?**

The goal is not to claim that existing SEP forecasting studies are invalid. Instead, we construct an evaluation framework that keeps predictions fixed while changing only the definition and weighting of the evaluation task. This makes the hypothesis directly falsifiable: if the standard and causal episode-normalized evaluations produce similar scores, then these evaluation choices do not materially affect the measured skill; if the scores differ substantially, then they are measuring partly different abilities.

## 2. Research hypothesis

We preregistered the hypothesis that conventional window-level occurrence scoring may over-represent two mechanisms for some models:

1. **episode multiplicity** — one physical SEP episode contributes multiple correlated positive windows; and
2. **persistence recognition** — the issue time occurs while the proton event is already active.

The benchmark was designed so that these mechanisms can be separated without changing the fitted models after outcomes are inspected.

The primary expected observation was not a specific numerical decrease in every model. Different models could respond differently because they use different input information. A proton-aware model, for example, might be much more sensitive to persistence exclusion than an X-ray-only model. Therefore, the scientifically relevant output is the change in measured skill under explicitly defined estimands, together with physical-unit uncertainty, rather than a requirement that all scores decrease.

## 3. Data and benchmark design

### 3.1 Operational event definition

The operational event threshold used in this work is a >10 MeV proton flux crossing of 10 pfu. For causal new-onset scoring, the forecast issue time must be below the threshold and a qualifying crossing must occur within the frozen future horizon. Already-active cases are classified as persistence and are never counted as new-onset positives.

### 3.2 Public benchmark audit

We first audited the public rolling-window construction in a pinned version of the `yuyian/SEP-Prediction` repository. The upstream commit, rolling table, event table and construction script were hash-bound before the audit.

For each public forecast window, the audit independently reconstructed whether an operational ≥10 pfu episode overlapped the nominal subsequent 24-hour interval, whether the issue time was already inside an active episode, whether a new physical episode began in the future interval, and which physical episode could be uniquely associated with the positive window.

This arm did not fit or score a model. Its purpose was to test whether multiplicity and persistence are substantial enough in a current public rolling benchmark to justify a separate causal episode-level evaluation.

### 3.3 Internal fixed-model development cohort

The model-score arm used exposed 2014–2017 data under an expanding out-of-fold design. These dates are development evidence, not independent final evidence. Protected post-2025 outcomes were not accessed.

The fixed comparators were deliberately simple and heterogeneous:

- fit-prevalence climatology;
- a current-proton-active diagnostic;
- elastic-net logistic regression using joint inputs;
- joint-input XGBoost;
- XRS-only XGBoost.

The current-proton-active diagnostic is especially important as a mechanism control. It does not attempt to forecast the solar drivers of a future event. It simply indicates whether the proton state is already active at the issue time. If it performs well under ordinary occurrence scoring but cannot detect new onset, that demonstrates directly that the two evaluation questions are not equivalent.

Models, feature definitions, chronological roles, thresholds and evaluation rules were frozen before the authoritative scoring run. No post-result model rescue was allowed.

## 4. Evaluation framework

### 4.1 Standard window occurrence

`WINDOW_OCCURRENCE_STANDARD` reproduces the familiar question: for each eligible issue window, does the following forecast interval contain the target occurrence? Every scored positive window receives unit weight.

This view is useful because it remains comparable with common window-based forecasting practice. However, it does not by itself distinguish whether multiple positives come from one physical episode or whether an event is already active at issue time.

### 4.2 Episode-normalized occurrence

`EPISODE_NORMALIZED_OCCURRENCE` retains occurrence-style scoring but changes the positive statistical unit. For uniquely mapped positive windows, all windows belonging to one physical episode share a total positive weight of one.

If an episode produces `m` eligible positive windows, each receives positive weight `1/m`. Therefore:

\[
\sum_{i \in E_j} w_i = 1
\]

for every uniquely mapped physical episode `E_j`.

This prevents a long episode from receiving more total positive mass solely because it spans more forecast windows.

### 4.3 New-onset causal scoring

`NEW_ONSET_CAUSAL` asks the narrower operational question: was the event not active at issue time, and did a new qualifying threshold crossing occur within the future horizon?

Already-active persistence cases receive zero onset weight. This isolates pre-onset warning ability from continued recognition of an existing event.

### 4.4 Persistence diagnostic

Already-active cases are retained as a separate diagnostic group rather than silently discarded. This allows us to measure whether a model is good at identifying persistence while preventing that ability from being mislabeled as new-event forecasting.

## 5. Statistical uncertainty

Ordinary row-level bootstrapping would recreate the same independence problem that motivated the study. Therefore, inference was based on physical units.

Ambiguous positive windows were retained in full-window descriptive point tables but were not assigned arbitrarily to a physical event for the bootstrap. Instead, a separately labelled uniquely mapped sensitivity cohort was used for physical-episode inference.

Positive resampling units were distinct physical SEP episodes. Negative resampling units were contiguous quiet blocks. Ten thousand bootstrap replicates were generated. Crucially, the same resampling draws were reused across compared models, evaluation definitions and contrasts. This paired design reduces unnecessary Monte Carlo noise and ensures that a difference between two methods is evaluated on the same resampled physical situations.

The study reports medians and 95% bootstrap intervals for TSS differences. A confidence interval excluding zero is treated as evidence of a directional effect within the exposed development cohort, not as universal evidence across all SEP datasets.

## 6. Reproducibility and audit correction

The computational workflow used a pinned Python environment, frozen JSON contracts, source and file hashes, persisted candidate attrition, one prediction row per scored issue/model, serialized fixed models, shared bootstrap-draw tensors and immutable workflow artifacts.

An independent verifier was implemented separately from the benchmark runner. It recomputed metrics, physical-episode weights, attrition counts, paired bootstrap contrasts and hashes directly from persisted CSV and NPZ evidence rather than importing the benchmark's scoring functions.

This independent audit found two defects in the first successful evidence package:

1. the concatenated expanding-out-of-fold summary used one fold's threshold when summarizing all folds, even though each persisted prediction row already contained the correct fold-specific frozen threshold and alert; and
2. the bootstrap operated on the uniquely mapped physical-episode cohort, but the first package did not label the difference between that inferential cohort and the full descriptive cohort clearly enough.

The correction was deliberately restricted to the result/evidence layer. No model was refitted, no feature was changed, no hyperparameter was searched, no threshold was reselected, and no protected outcome was accessed. The original result files were preserved as legacy evidence.

The authoritative corrected workflow passed the independent verifier, regenerated the evidence hashes including the verification receipt, and then verified the final hash manifest again before upload.

## 7. Results

### 7.1 Public benchmark multiplicity and persistence

The pinned public rolling table contained 11,773 fixed 24-hour windows and 1,726 stored operational-positive rows.

Independent threshold-episode reconstruction found 643 windows with an operational ≥10 pfu episode overlapping the nominal future interval. Under the audited event-table semantics, 1,083 of the stored operational-positive rows did not overlap such a reconstructed episode in the nominal future interval. This is a target-reconciliation finding for the pinned public table and is **not** a claim that SEPNET's final published SEPVAL score is wrong.

The reconstructed windows also separated into 411 already-active persistence cases and 227 new-onset cases. Among uniquely mapped positives, 610 positive windows represented 256 distinct physical episodes. The resulting episode multiplicity factor was:

\[
\frac{610}{256}=2.3828125.
\]

Thus repeated representation of physical events is not a hypothetical edge case in this public benchmark.

### 7.2 Development cohort composition

The exposed expanding-OOF model-score cohort contained:

- 936 scored issue rows;
- 27 standard positive windows;
- 10 uniquely mapped positive physical episodes;
- 5 distinct new-onset episodes;
- 11 already-active persistence windows;
- 11 ambiguous positive windows retained for full-window descriptive scoring.

The mapped episode multiplicity factor was 1.6.

The most important limitation is immediate: only five distinct positive onset episodes were available. Therefore, onset-specific ranking claims have low statistical power and are not presented as final independent evidence.

### 7.3 Fixed-model True Skill Statistic

| Fixed model | Standard occurrence TSS | Episode-normalized occurrence TSS | New-onset TSS |
|---|---:|---:|---:|
| Current-proton-active diagnostic | 0.556 | 0.567 | 0.000 |
| Elastic-net joint | 0.642 | 0.765 | 0.665 |
| XGBoost joint | 0.657 | 0.443 | 0.043 |
| XGBoost XRS-only | 0.320 | 0.266 | 0.216 |
| Fit-prevalence climatology | 0.000 | 0.000 | 0.000 |

The clearest mechanism result is the current-proton-active diagnostic. Under ordinary occurrence scoring, it achieved TSS 0.556. Under new-onset scoring it achieved exactly 0.000. A method that only recognizes that the storm is already occurring can therefore appear skillful under one evaluation definition while having no new-onset forecasting skill.

The joint XGBoost showed the largest point change among learned models: TSS 0.657 under standard occurrence, 0.443 after episode normalization, and 0.043 under new-onset scoring.

The elastic-net joint model behaved differently, increasing from 0.642 standard TSS to 0.765 episode-normalized occurrence and retaining 0.665 onset TSS. This is important because it shows the framework is not constructed to force every model downward. It measures a change in estimand; individual models can benefit or worsen depending on which physical cases their predictions handle well.

### 7.4 Multiplicity-only uncertainty

On the uniquely mapped physical-episode sensitivity cohort, joint XGBoost had a median multiplicity-only TSS shift of:

\[
\Delta TSS=-0.133
\]

with 95% bootstrap interval:

\[
[-0.225,-0.041].
\]

The estimated probability of a negative shift across the shared bootstrap draws was 0.9883. Within this exposed cohort, repeated window weighting therefore materially increased the measured TSS of joint XGBoost relative to episode-normalized occurrence scoring.

### 7.5 Persistence-exclusion uncertainty

For joint XGBoost, excluding already-active persistence and evaluating causal new onset produced a median mapped TSS shift of -0.400 with 95% interval [-0.800,-0.125].

For the current-proton-active diagnostic, the persistence-exclusion shift was -0.567 with 95% interval [-0.867,-0.267].

Both intervals exclude zero. This supports the interpretation that persistence contributes materially to occurrence-style skill for these proton-aware predictors in this development cohort.

### 7.6 Model ranking

Point rankings changed across evaluation definitions. Joint XGBoost exceeded XRS-only XGBoost under standard occurrence scoring (0.657 versus 0.320), while under new onset the point values were 0.043 and 0.216 respectively.

However, the paired mapped-episode bootstrap for the onset difference `joint - XRS-only` had median -0.168 and 95% interval [-0.746,+0.244]. The interval crosses zero. Although the standard-to-onset ordering reversed in 71.8% of bootstrap replicates, a statistically secure rank reversal is not established.

By contrast, elastic-net joint exceeded XRS-only on the mapped onset cohort with median TSS difference +0.432 and 95% interval [+0.195,+0.894].

## 8. Discussion

The central result is not that one particular machine-learning architecture fails. The result is that **forecasting skill depends on the scientific question represented by the evaluation protocol**.

A conventional 24-hour occurrence question can be useful. An operator may care whether hazardous proton levels will be present during the next day, including continuation of an ongoing event. But this should not automatically be interpreted as evidence that the system warned before onset. If the intended claim is pre-onset forecasting, issue-time causal eligibility must be enforced.

Similarly, multiple forecast windows during one physical episode may represent legitimate repeated decisions. Yet if the scientific claim concerns generalization across independent SEP events, allowing each window to carry full positive mass makes long events more influential. Episode-normalized scoring provides a complementary answer in which each mapped physical event contributes the same total positive mass.

The public audit shows that both effects are large enough to matter in real benchmark construction. The internal model experiment then demonstrates that these are not merely semantic distinctions: measured TSS changes substantially for some fixed models.

The current-proton-active diagnostic is useful pedagogically because it strips the issue to its simplest form. It can do well at occurrence recognition while having no onset ability. This makes the benchmark understandable without requiring a judge to understand XGBoost internals.

The elastic-net result is equally important scientifically. Its performance does not collapse under onset scoring. Therefore, the benchmark is not a penalty engineered to make sophisticated models look worse. Instead, it reveals which models are robust to a stricter causal question and which depend more heavily on repeated or already-active cases.

## 9. Limitations

This study has several limitations that constrain the allowed claims.

First, the internal onset cohort contains only five distinct positive episodes. That is not sufficient for strong independent conclusions about model ranking. The bootstrap reflects uncertainty within the available mapped cohort but cannot manufacture missing independent events.

Second, the public audit uses the event-table semantics of a pinned public repository. The finding that 1,083 stored operational-positive rows do not overlap a reconstructed threshold episode under those semantics is a reproducible dataset-audit result; it does not establish that the authors' final SEPVAL evaluation uses the same mismatch.

Third, ambiguous positive windows cannot be safely assigned to one physical episode. We therefore retain them in full descriptive results but exclude them from the separately labelled mapped-episode bootstrap sensitivity. This reduces inferential sample size but avoids inventing physical identity.

Fourth, the development period has been exposed during project development. It is therefore methodology evidence rather than untouched final evidence. Post-2025 protected outcomes remain sealed and were not used for power rescue or narrative tuning.

Finally, episode normalization answers one scientific question, not every operational question. A continuously updated operational system may legitimately issue multiple forecasts during one storm. Our proposal is that papers should report both window-level operational performance and physical-event/onset performance when making claims about forecasting new events.

## 10. Conclusion

This study developed and independently verified an episode-normalized causal benchmark for 24-hour operational SEP forecasting.

A public benchmark audit showed substantial repeated representation of physical episodes and a large number of already-active persistence windows. In exposed fixed-model development data, a trivial current-proton-active rule scored TSS 0.556 under conventional occurrence evaluation but TSS 0.000 for genuine new onset. Joint XGBoost changed from TSS 0.657 under standard occurrence to 0.443 after episode normalization and 0.043 for new onset. Physical-unit bootstrap intervals supported negative multiplicity and persistence effects for joint XGBoost, while the five-event onset cohort remained too small for a secure joint-versus-XRS model-rank conclusion.

The appropriate conclusion is therefore narrow but meaningful:

> **Conventional window-level SEP occurrence evaluation can partly reward repeated representations of physical events and recognition of already-active storms. Reporting physical-episode-normalized and new-onset-specific skill provides a more causally interpretable measure of pre-onset warning ability.**

Future confirmation should use a preregistered independent cohort or an additional public dataset with sufficient distinct onset events. The existing protected final cohort must remain sealed until its frozen evaluation contract is executed without development-side inspection.

## References

1. NOAA / NWS Space Weather Prediction Center. *NOAA Space Weather Scales*. Solar Radiation Storm scale; operational physical measure based on ≥10 MeV particle flux.
2. Yu, Y., Chen, Y., Zhao, L., Whitman, K., Manchester, W., & Gombosi, T. (2026). *Solar Energetic Particle Forecasting With Multi-Task Deep Learning: SEPNET*. Journal of Geophysical Research: Machine Learning and Computation. DOI: 10.1029/2026JH001247.
3. Yu, Y., Chen, Y., Zhao, L., Whitman, K., Manchester, W., & Gombosi, T. (2026). *SEP-PRISM Data: A multi-source dataset for solar energetic particle forecasting*. arXiv:2607.16160.
4. Whitman, K. (2025). *SEPVAL benchmark resources*. NASA Community Coordinated Modeling Center / Zenodo, as referenced by SEPNET.

## Computational evidence receipt

Authoritative audit-corrected development workflow:

- run: `34371418428`
- benchmark commit: `8b8a1549a7dfaf1745c96a8c7b78d98e564625c5`
- artifact ID: `10112207048`
- artifact SHA-256: `418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777`
- independent V2 verification: PASS
- final evidence-manifest verification: PASS

Pinned public benchmark audit:

- upstream commit: `d0eb54e46b7dd6c760325e123d2ad86f9420fbff`
- workflow run: `34362893938`
- artifact ID: `10108593291`
- artifact SHA-256: `498104f66efb59ebd85795d628d799006f60266a8e3454bd2df3b127b988673d`

## Required final student checks before submission

- Verify every number against the authoritative corrected artifact.
- Replace any wording that does not match the student researchers' own understanding and voice.
- Do not add school name, city or state anywhere in the paper, abstract, slides or video.
- Keep the five-onset-event limitation prominent.
- Do not turn the public target audit into a claim that SEPNET's final published score is wrong.
- Do not describe development evidence as independent prospective validation.
