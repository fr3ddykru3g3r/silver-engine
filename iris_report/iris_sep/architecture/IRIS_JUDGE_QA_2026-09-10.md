# IRIS judge defense Q&A — episode-normalized causal SEP benchmark

**Purpose:** concise oral-defense answers consistent with the verified evidence. Do not strengthen beyond these boundaries.

## 1. What is the project in one sentence?

We test whether ordinary 24-hour SEP forecasting scores partly reward two things that are different from forecasting a new storm: recognizing a storm that is already active and counting one long physical event in multiple positive windows.

## 2. What did you actually invent?

We built an **episode-normalized causal evaluation benchmark**. It separates new onset from already-active persistence and gives each uniquely mapped physical SEP episode total positive statistical weight one, instead of allowing a long event to contribute several full-weight positive windows.

## 3. Isn’t onset versus persistence obvious?

The distinction itself is not new. Our contribution is to **measure how much the evaluation definition changes fixed-model skill when onset/persistence separation and physical-event normalization are applied together under a reproducible benchmark**. We also preserve standard window scoring so the comparison is paired rather than rhetorical.

## 4. What is your cleanest evidence that standard scoring can reward persistence?

A deliberately trivial diagnostic that only checks whether proton flux is already above the operational threshold gets **TSS 0.556** under ordinary occurrence scoring but **TSS 0.000** on genuine new-onset opportunities. It can recognize an ongoing storm; it cannot warn before onset.

## 5. Why use TSS?

SEP events are rare, so raw accuracy can be misleading. TSS combines the hit rate and false-positive rate and is commonly used for imbalanced forecasting tasks. Most importantly here, we compare the **same metric on the same fixed predictions** while changing only the evaluation definition.

## 6. Why not simply delete repeated windows?

Deleting rows would throw away legitimate forecast issue times. Episode normalization keeps the windows but prevents a long physical event from receiving more total positive statistical mass just because it spans more windows. If one episode maps to `m` positive windows, each gets weight `1/m`, so the episode contributes total positive weight one.

## 7. Why not count every window equally if an operator really makes forecasts every day?

That is a valid operational question, which is why we retain conventional window-level results. But if a paper claims it can forecast **new SEP events**, physical-event and onset-specific evaluation answers a different question. Our recommendation is complementary reporting, not replacing all operational window metrics.

## 8. What did the public benchmark audit find?

In the pinned public rolling table there were **11,773 windows**. Among uniquely mapped positives, **610 positive windows represented 256 physical SEP episodes**, giving a multiplicity factor of **2.38**. The reconstruction also identified **411 already-active persistence windows versus 227 new-onset windows**.

## 9. Are you saying the SEPNET paper is wrong?

No. The audit demonstrates a reproducible property of its pinned public rolling data and event-table semantics. SEPNET uses a separate SEPVAL testing construction for its final published benchmark. We do **not** claim its final SEPVAL score is wrong.

## 10. What happened to your own fixed models?

On the exposed 2014–2017 OOF development cohort:

- current-proton-active diagnostic: `0.556 standard -> 0.000 onset`;
- joint XGBoost: `0.657 standard -> 0.443 episode-normalized occurrence -> 0.043 onset`;
- XRS-only XGBoost: `0.320 standard -> 0.216 onset`;
- elastic-net joint: `0.642 standard -> 0.665 onset`.

This is useful because the framework does not mechanically force every model downward.

## 11. Which effects were statistically supported?

On the uniquely mapped physical-unit bootstrap cohort, using 10,000 shared draws:

- joint-XGBoost multiplicity-only TSS shift: median **-0.133**, 95% **[-0.225,-0.041]**;
- joint-XGBoost persistence-exclusion shift: median **-0.400**, 95% **[-0.800,-0.125]**;
- active-proton diagnostic persistence-exclusion shift: median **-0.567**, 95% **[-0.867,-0.267]**.

These intervals exclude zero in the exposed development cohort.

## 12. Did the model ranking change?

The point ranking between joint XGBoost and XRS-only XGBoost reverses from standard occurrence to new onset. But the mapped onset difference has 95% interval **[-0.746,+0.244]**, so the reversal is **not statistically secure**. We report it as a point observation, not a confirmed general result.

## 13. What is the biggest weakness?

The internal causal onset arm contains only **five distinct positive onset episodes**. That makes strong model-ranking claims underpowered. We do not hide this and we do not tune the models to rescue it.

## 14. Why is the public-data arm useful if the internal arm is small?

The public audit has much higher event count and demonstrates that multiplicity and persistence are substantial benchmark-construction phenomena, while the internal fixed-model arm shows that these choices can change measured skill. The two arms answer different parts of the argument.

## 15. How did you avoid changing the experiment after seeing results?

We froze the model families, feature definitions, chronological roles, thresholds and evaluation contracts. After the first successful run, an independent audit found an OOF summary bug and an estimand-labeling ambiguity. The correction changed only the result/evidence layer: **no refit, no feature change, no threshold reselection, no hyperparameter search and no protected-outcome access**. Legacy uncorrected tables were retained.

## 16. How do you know the corrected numbers are reproducible?

The authoritative workflow ran in a pinned environment and persisted prediction rows, attrition, models, shared bootstrap tensors and hashes. A separate verifier that does not import the benchmark runner recomputed the metrics, episode weights, contrasts and file hashes. It passed, hashes were regenerated, and the final manifest was verified a second time.

## 17. What is the difference between “independently verified” and “independent final evidence” here?

The **computation** was independently recomputed from persisted evidence. But the 2014–2017 outcomes are already exposed development data, so the **scientific cohort** is not an untouched final test. Those are different meanings of independence and we state both explicitly.

## 18. Why keep the post-2025 cohort sealed?

Because once we inspect its labels, it can no longer serve as a genuinely untouched evaluation cohort. We therefore forbid development-side queries, counting, stratification, threshold changes or model selection using those outcomes.

## 19. What would falsify your main hypothesis?

If fixed models retained essentially the same skill and ordering after physical-event normalization and causal onset restriction, with paired uncertainty centered near zero, then multiplicity and persistence would not materially affect evaluation. Our framework was built to allow that null result.

## 20. What should the field do differently?

When a study makes a claim about forecasting **new** SEP events, report conventional window-level performance together with: (1) already-active persistence separately, (2) new-onset performance, and (3) a physical-event-normalized sensitivity so one long event cannot dominate simply by spanning more windows.

## 21. Why is this more valuable than building another neural network?

The literature already contains sophisticated neural networks and multi-source forecasting systems. A benchmark that clarifies what their scores actually represent can be applied to many architectures. It targets the measurement problem rather than competing only on another model architecture.

## 22. Are the results operationally deployable?

No operational-certification claim is made. This project is an evaluation-methodology study. Its immediate output is a more interpretable way to measure pre-onset warning skill, not an autonomous spacecraft-control system.

## 23. What is your final claim in one sentence?

**On the exposed cohorts studied here, conventional window-level SEP occurrence scoring can partly reward repeated representations of physical episodes and recognition of already-active storms; physical-event normalization and causal new-onset scoring provide a more interpretable measure of pre-onset forecasting skill.**

## Numbers to memorize

- Public windows: **11,773**
- Public mapped positives / episodes: **610 / 256 = 2.38×**
- Public persistence / onset windows: **411 / 227**
- Internal scored rows: **936**
- Internal standard positives: **27**
- Internal mapped positive episodes: **10**
- Internal new-onset episodes: **5**
- Active diagnostic: **0.556 -> 0.000 TSS**
- Joint XGB: **0.657 -> 0.443 -> 0.043 TSS**
- Joint XGB multiplicity shift: **-0.133 [-0.225,-0.041]**
- Joint XGB persistence shift: **-0.400 [-0.800,-0.125]**
- Authoritative run: **34371418428**
- Artifact SHA-256: **418fd9bd2a70d0e545095a5a8009548aa74c8734d7bb9e2ed09fb19343c92777**
