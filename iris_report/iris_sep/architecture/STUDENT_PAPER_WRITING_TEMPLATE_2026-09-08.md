# Student paper writing template — IRIS-SEP

**Use:** structure and evidence prompts for Kyros Goyal and Lokesh Chaudhary to write the final competition material themselves.

**Do not submit this file as-is.** It is deliberately a template, not finished competition prose. Use `IRIS_EVIDENCE_DOSSIER_2026-09-08.md` and the original receipts to verify every number before writing. Preserve required assistance disclosures.

---

## A. One-sentence project explanation

Write one sentence that a judge with no solar-physics background can understand.

Your sentence should contain:

- the problem: forecast systems can lose measurements;
- the target: a new high-energy proton/radiation threshold crossing in the next 24 hours;
- the idea: distinguish a probability from whether that probability is trustworthy enough to expose;
- the behavior: degrade or abstain when evidence is insufficient.

**Avoid:** “most accurate,” “industry-leading,” “breakthrough,” “physically simulates solar flares,” “operationally ready.”

---

## B. Title

Draft 3–5 titles yourself. Prefer titles that foreground reliability/missing evidence rather than generic AI.

A strong title should communicate at least two of:

- solar energetic particles;
- missing/degraded measurements;
- reliability-aware forecasting;
- abstention;
- 24-hour new-crossing forecast.

Do not use words that imply validated operational superiority.

---

## C. Abstract planning sheet

Write the abstract only after the main paper is complete.

### Background — 1–2 sentences

Explain:

- what solar energetic particles are;
- why >10 MeV proton events matter;
- why forecasting becomes difficult when measurement feeds are missing.

### Question — 1 sentence

State the exact target and reliability question.

Evidence anchor:

- new >10 MeV, >=10 pfu crossing;
- next 24 hours;
- already-active issue times excluded.

### Method — 2–4 sentences

Cover only the essential architecture:

- three evidence families: solar/context, XRS, proton;
- cross-fitted specialist models;
- frozen state-specific missing-feed fallbacks;
- `DEGRADED` / `ABSTAIN` policy;
- no runtime reconstruction of a missing feed.

### Results — 2–4 sentences

Choose only results directly supported by receipts. Options include:

- deterministic package replay;
- development point estimates for V3 missing-feed states;
- negative alert-filter result showing FP/detection trade-off;
- six-of-seven source-family prospective acquisition;
- CEA-NRT missing three required frozen SHARP quantities;
- fail-closed preflight emitted no invalid skill probability.

Do **not** imply independent prospective skill.

### Conclusion — 1–2 sentences

Your conclusion should distinguish:

- what is demonstrated: reliability architecture and fail-closed behavior;
- what is not yet demonstrated: independent untouched prospective superiority.

---

## D. Introduction / objective

### Paragraph 1 — physical problem

In your own words explain:

- solar activity can accelerate energetic protons;
- the target is an SEP proton-threshold event, not a solar-flare prediction target;
- monitoring systems use multiple measurements;
- missing feeds are realistic and create an evidence-quality problem.

### Paragraph 2 — forecasting problem

Explain the key conceptual issue:

> A model can continue producing a plausible-looking probability even when part of its expected evidence disappears.

Then motivate why probability magnitude alone is insufficient.

### Paragraph 3 — objective

State the research question exactly enough that another researcher could test it.

Include:

- new crossing;
- >10 MeV;
- >=10 pfu;
- 24-hour horizon;
- separate forecast probability and exposure/validity decision.

---

## E. Innovation / contribution

Write this section around the **decision architecture**, not around model size.

Evidence-supported contribution candidates:

1. treating availability as an explicit model state;
2. pre-training reduced-input fallbacks instead of fabricating missing feeds at runtime;
3. separating numerical probability from permission (`VALID` / `DEGRADED` / `ABSTAIN`);
4. immutable schema-bound package replay;
5. trusted source receipts and fail-closed prospective preflight;
6. refusing a forecast when the exact frozen live feature interface is unavailable.

### Required nuance

Do not claim each individual component is novel in isolation unless you have literature evidence. Frame novelty at the level you can defend: the controlled reliability workflow and evidence boundaries used in this project.

---

## F. Methodology

### F1. Target construction

Explain:

- definition of new crossing;
- already-active exclusions;
- 24-hour horizon;
- how labels are separated chronologically from issue times.

### F2. Evidence families

Describe the three broad groups without drowning the judge in 259 feature names:

- solar/context evidence;
- X-ray evidence;
- proton-history evidence.

State that the final frozen vector has 259 ordered positions.

### F3. Model structure

Explain:

- specialist XGBoost models;
- five seeds per family;
- median across seeds;
- chronological out-of-fold construction;
- non-negative evidence stack;
- separate calibration and threshold roles.

### F4. Missing-feed policy

Create a four-row table:

| State | Evidence available | Model path | Exposure permission |
|---|---|---|---|
| FULL | solar + XRS + proton | full frozen stack | normal subject to provenance |
| NO_XRS | solar + proton | pre-trained reduced-input model | DEGRADED |
| NO_PROTON | solar + XRS | pre-trained reduced-input model | DEGRADED |
| NO_XRS_OR_PROTON | solar only | diagnostic only | ABSTAIN |

Explicitly state:

- no runtime retraining;
- no runtime feed fabrication;
- `ABSTAIN` cannot emit an alert.

### F5. Evaluation separation

Explain why fit, calibration, threshold, score/monitor and future final evidence are separated.

Mention that previously inspected blocks cannot later be called untouched.

### F6. Metrics

Define in your own words and be ready to derive:

- TSS;
- POD/sensitivity;
- FAR;
- Brier score;
- calibration error if used in your final figures.

Do not present one metric as sufficient by itself.

### F7. Prospective provenance gate

Explain the trusted-source registry and acquisition receipts.

Critical point:

A finite historical aggregate value is not automatically proof that it was natively available at forecast time.

---

## G. Results structure

Do not build this as a list of every experiment ever run. Use a narrative.

### Result 1 — package and replay integrity

Evidence to verify:

- 15 specialists;
- package ZIP SHA;
- independent replay maximum absolute difference ~2.22e-16;
- schema binding;
- solar-only abstain emits no alert.

Interpretation to write yourself:

What does exact replay establish? What does it not establish?

### Result 2 — missing-feed development behavior

Use development point estimates conservatively.

Available evidence:

- `NO_XRS`: TSS about 0.368087 -> 0.430492, but uncertainty crosses zero and Brier is slightly worse;
- `NO_PROTON`: TSS about 0.507371 -> 0.512999; 16/21 detections retained; FP 814 -> 796.

Required wording concept:

- “development point estimate”;
- “promising/inconclusive,” not “proven improvement.”

### Result 3 — why probability preservation is not enough

Use the missingness/outage evidence to show that probability similarity and safe decisions can diverge.

Evidence example:

- at 40% random observed-cell loss, frozen diagnostic-policy TSS dropped by about 0.227 even when causal forward-fill preserved probabilities comparatively well.

### Result 4 — negative false-alarm experiment

Show the monotone-veto table:

| State | V3 TP/FP | Veto TP/FP | Interpretation |
|---|---|---|---|
| NO_XRS | 13 / 603 | 12 / 431 | 28.5% FP reduction but one lost TP |
| NO_PROTON | 16 / 796 | 13 / 594 | 25.4% FP reduction but three lost TPs |

State that the veto was rejected and further tuning stopped because the score block was already inspected.

This is useful scientific evidence of a real trade-off.

### Result 5 — prospective preflight

Use the final run `34217607075`.

Report:

- 7 registered source families attempted;
- 6 authenticated;
- CDAW failed after bounded retries;
- JSOC CEA-NRT missing `CMASKL`, `MEANGBL`, `USFLUXL`;
- 18 frozen vector positions affected;
- prospective skill forecast not admissible;
- no probability emitted.

Interpret the no-forecast outcome as a **fail-closed validation result**, not as forecast skill.

---

## H. Discussion prompts

Write this section yourselves by answering these questions.

### H1. What worked?

Possible evidence-supported themes:

- deterministic packaging and replay;
- explicit availability states;
- runtime abstention mechanics;
- source authentication framework;
- refusal to substitute missing live quantities.

### H2. What did not work?

Must include important negatives:

- high FAR in historical monitor evidence;
- full-stack comparison with late fusion remains inconclusive;
- false-alarm veto loses detections;
- exact prospective frozen interface unavailable from CEA-NRT;
- one external source acquisition failed in the final preflight;
- no independent prospective skill result yet.

### H3. Why are these failures scientifically informative?

Explain how each changed the project design rather than being hidden.

### H4. What would make the result stronger?

Possible future work:

- construct a new model specifically from quantities genuinely available at forecast time;
- pre-register and freeze that interface before final evaluation;
- obtain a genuinely untouched outcome cohort;
- compare against a same-date comparator with aligned target semantics;
- evaluate analyst review burden and alerts-per-detected-event prospectively.

Do not describe future work as already completed.

---

## I. Conclusion checklist

Before finalizing your conclusion, check every sentence:

- Does it distinguish retrospective development from prospective validation?
- Does it avoid saying V3 is operationally superior?
- Does it mention the core reliability contribution?
- Does it acknowledge the live-interface blocker?
- Does it avoid hiding FAR/detection trade-offs?
- Does it state what future evidence is still needed?

---

## J. Figures and tables

### Figure 1 — layman architecture

Draw:

`Solar measurements -> Availability check -> Full/reduced model -> VALID / DEGRADED / ABSTAIN -> Human analyst`

Make `ABSTAIN = no alert` visually obvious.

### Figure 2 — four-state missing-feed diagram

Show which sensor families are present in each state.

### Figure 3 — missingness lesson

Plot a decision metric against missingness level, with probability-similarity information if available from the verified receipt.

Caption must say development stress test.

### Figure 4 — rejected veto trade-off

A simple before/after TP and FP display for `NO_XRS` and `NO_PROTON`.

### Figure 5 — prospective source gate

Seven source boxes -> source authentication -> exact 259-position interface check -> missing three NRT keywords -> **NO FORECAST**.

### Table 1 — frozen architecture

Summarize model families, state models, and permission semantics.

### Table 2 — evidence status

Columns:

- evidence question;
- status;
- receipt/run;
- allowed claim.

Use `config/eight_issue_status_2026-09-08.json` as the source.

---

## K. Judge interview preparation

Each student should independently be able to answer all of the following without reading from a script.

### Fundamentals

1. What is a solar energetic particle event?
2. Why are protons above 10 MeV used here?
3. What does 10 pfu mean?
4. How is an SEP different from a solar flare?
5. What makes a crossing “new”?

### Modeling

6. Why XGBoost rather than a huge neural network?
7. Why use specialist evidence families?
8. Why five seeds?
9. Why chronological cross-fitting?
10. Why non-negative stacking?
11. Why separate calibration and threshold selection?

### Reliability

12. Why is a missing feed different from a low sensor value?
13. Why not just fill a missing sensor value with zero?
14. Why not interpolate everything?
15. Why can probability remain stable while TSS falls?
16. What exactly does `DEGRADED` mean?
17. What exactly does `ABSTAIN` mean?
18. Why can `ABSTAIN` be scientifically preferable to guessing?

### Evaluation

19. Derive TSS from the confusion matrix.
20. Explain FAR in plain language.
21. Why is 95.5% FAR a serious limitation?
22. Why was the veto filter rejected despite cutting FP by ~25–29%?
23. What would constitute an untouched final test?
24. Why can you no longer treat the inspected score block as fresh evidence?

### Provenance

25. Why is the historical aggregate table not automatically causal?
26. What is an acquisition receipt?
27. Why did the preflight refuse a forecast?
28. Which three CEA-NRT fields were missing?
29. Why do those missing fields affect 18 positions rather than only three numbers?
30. Why was CDAW not silently replaced by DONKI?

### Ownership

31. Which design choices did you personally make?
32. Which code paths can you reproduce yourself?
33. Which results did you personally inspect and verify?
34. What external or AI assistance did the project receive?
35. What is the strongest claim you deliberately chose **not** to make, and why?

---

## L. Assistance / ownership log prompts

Maintain a student-authored record with dates and concrete actions.

For each meaningful project step, record:

- question being investigated;
- idea proposed by the students;
- advice received from teachers/mentors/AI/tools;
- code or analysis written/generated with assistance;
- what the students checked independently;
- decision made and why;
- result;
- whether the result changed the next experiment.

Do not write “AI helped with coding” as a single vague line. Be specific enough that a judge can understand the students' intellectual contribution.

---

## M. Final claim audit

Before any abstract/poster/report submission, mark every claim as one of:

- **DIRECTLY VERIFIED** — backed by an immutable receipt or source artifact;
- **DEVELOPMENT EVIDENCE** — observed on an inspected development cohort;
- **INFERENCE** — reasonable interpretation, clearly labelled;
- **FUTURE WORK** — not yet performed;
- **FORBIDDEN/UNSUPPORTED** — remove.

Automatically remove or rewrite any sentence implying:

- independent prospective superiority;
- operational certification;
- guaranteed warning reliability;
- economic savings;
- full-physics solar simulation;
- competition/award certainty.

---

## N. Final pre-submission verification

The student authors should not submit until they can answer **yes** to all of these:

- [ ] I can explain the exact research target from memory.
- [ ] I can distinguish SEP events from solar flares.
- [ ] I understand every model state and why each permission is assigned.
- [ ] I can explain the main metrics mathematically and intuitively.
- [ ] Every numeric result in my final text matches a verified receipt.
- [ ] I explicitly label development-only evidence.
- [ ] I include important negative/inconclusive results.
- [ ] I do not claim live skill from the fail-closed preflight.
- [ ] I can explain the three missing CEA-NRT quantities and the 18 affected positions.
- [ ] I can explain why the project refused to issue a prospective skill forecast.
- [ ] My final abstract/poster/report wording is my own.
- [ ] My assistance/AI disclosure is complete and accurate for the applicable competition.
