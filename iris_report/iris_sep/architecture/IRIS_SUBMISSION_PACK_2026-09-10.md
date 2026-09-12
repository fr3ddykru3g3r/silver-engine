# IRIS 2026 student drafting scaffold — episode-normalized causal SEP benchmark

**Status:** internal evidence-and-rehearsal scaffold, **not final submission prose**.  
**Student requirement:** write the abstract, portal text, poster text and spoken explanation in the students' own words after checking every number and source. Do not paste AI-generated prose into an ISEF-bound research plan, abstract, poster or citation list.

**Do not include:** school name, city, state, protected post-2025 outcomes, state-of-the-art claims, award claims, operational-certification claims, or a claim that SEPNET's published SEPVAL result is wrong.

## Recommended project title

**Are We Forecasting a New Solar Radiation Storm? An Episode-Normalized Causal Benchmark for SEP Prediction**

Short alternative:

**Forecasting New Solar Radiation Storms, Not Just Recognizing Active Ones**

## One-sentence judge hook to learn, not memorize word-for-word

A forecasting score can partly reward a model for recognizing a radiation storm that is already underway and for counting one long storm several times; this project measures how much the score changes when those effects are separated without changing the predictions.

## Evidence blocks for the 250-word abstract

Write the final abstract yourself from these verified blocks. A useful word budget is roughly 30–40 words background/question, 55–65 methods, 80–90 results, 25–35 limitations and 20–25 conclusion.

### Background and question

- Operational event: >10 MeV proton flux crossing 10 pfu.
- Scientific question: does ordinary 24-hour occurrence scoring partly mix genuine new-onset warning, already-active persistence recognition and repeated representation of one physical SEP episode?
- Contribution: an evaluation benchmark, not a new neural-network architecture.

### Method

- Pinned SEP-PRISM table: **14,464** daily 24-hour windows, **650** stored positives, zero reconstructed-target mismatches.
- Physical mapping: **614** uniquely mapped positive windows represent **257** physical episodes; EMF = **2.389**.
- Onset-state audit: **418** persistence windows, **228** onset windows, four eligibility-ambiguous positives; 36 separate positives overlap multiple episodes.
- Fixed replay: **7,558** chronological out-of-fold score issues, six fixed comparators, unchanged alerts.
- Matched inference: **85 onset episodes + 1,080 quiet blocks**, **10,000** shared physical-unit bootstrap draws.

### Primary results

Use the paired matched-population values, not the older five-onset internal experiment, as the headline result:

- joint XGBoost TSS: **0.726 mapped occurrence -> 0.621 episode-normalized -> 0.437 new onset**;
- episode-normalization contrast: **-0.104**, 95% bootstrap interval **[-0.146,-0.063]**;
- persistence-removal contrast: **-0.183**, **[-0.247,-0.125]**;
- past-proton proxy onset-minus-mapped contrast: **-0.506**, **[-0.567,-0.444]**.

All three frozen directional intervals are below zero. The fixed gate therefore returns `STRONG_MODEL_CONFIRMATION` for the historical replay.

### Required limitation sentence

The public source files had already been development-exposed; historical publication-time availability of every predictor is not proven; the replay lacks a purge at every role boundary and does not support an operational-readiness or model-superiority claim. Protected post-2025 outcomes remain sealed.

### Conclusion idea

The evidence supports **complementary reporting**: conventional window occurrence, physical-episode-normalized occurrence and genuine new-onset performance answer different questions and should not be treated as interchangeable.

## Introduction & objective — content checklist

The student-written version should explain, in this order:

1. what an SEP/radiation storm is and why a >10 MeV, 10 pfu crossing matters;
2. how fixed 24-hour supervised-learning windows are constructed;
3. how one physical storm can create several positive windows;
4. how a forecast can be issued after the threshold is already crossed;
5. the falsifiable question: if these effects do not matter, fixed-model skill should stay similar after re-weighting/restriction.

## Innovation — content checklist

The novelty claim must be narrow:

- onset versus persistence is **not** claimed as a new concept;
- event-based validation is **not** claimed as new in general;
- the tested contribution is the **combined reproducible decomposition of the same fixed SEP predictions** into mapped occurrence, episode-normalized occurrence and causal new onset, with shared physical-unit uncertainty and explicit claim boundaries.

## Methodology — content checklist

Mention:

- target reconstruction from the pinned event catalogue;
- episode identity and ambiguity rules;
- 1/m positive weighting within each uniquely mapped episode;
- chronological fit/threshold/score roles and unchanged alerts;
- six comparators, including a deterministic past-proton proxy;
- paired 10,000-draw bootstrap over physical onset episodes and Monday-anchored seven-day quiet blocks;
- independent score/evidence recomputation;
- protected-outcome registry and frozen prospective contract.

Do **not** claim full issue-time causality for all 259 historical predictor variables.

## Results and conclusions — content checklist

Lead with the three-stage joint-XGBoost matched TSS line and the three frozen intervals. Then state two negatives that improve credibility:

- joint-XGBoost minus proton-free XGBoost onset difference is uncertain (about **-0.041**, interval crossing zero);
- joint-XGBoost onset alerting is not operationally ready: TP=41, FN=44, FP=330, TN=6,984, giving 48.24% sensitivity and an 88.95% **false-alarm ratio** (false-positive rate about 4.51%).

## 90-second video rehearsal beats

Do not memorize a generated script. Rehearse these beats until you can explain them naturally in 80–90 seconds:

1. **Question (10–12 s):** are we warning before a new storm, or partly recognizing one already underway?
2. **Mechanism (15–18 s):** one long physical storm can produce multiple positive daily windows; some issue times occur after threshold crossing.
3. **Benchmark (15–18 s):** map windows to physical episodes, give each mapped episode total positive weight one, and score new onset separately while leaving predictions fixed.
4. **Dataset result (10–12 s):** 614 mapped positive windows represented 257 physical episodes; 418 positives were persistence and 228 onset.
5. **Model replay (20–22 s):** matched joint-XGBoost TSS 0.726 -> 0.621 -> 0.437; all three preregistered directional intervals were below zero.
6. **Boundary (10–12 s):** this is exposed historical confirmation, not operational validation; protected prospective evidence remains sealed.
7. **Takeaway (5–8 s):** report window, episode-normalized and new-onset skill separately when the claim is forecasting new storms.

## 30-second fallback beats

Question -> one storm counted several times / already active -> benchmark separates them -> 0.726 -> 0.621 -> 0.437 on matched replay -> conclusion is evaluation sensitivity, not model superiority.

## What to say when a judge asks “what is actually new?”

Answer in your own words using this structure:

- concede the established pieces: SEP onset, persistence, TSS and event-based validation already exist;
- identify the gap: fixed-window SEP benchmarks can mix physical-event multiplicity and already-active state;
- identify the contribution: same fixed predictions, three explicit estimands, episode-level weighting, shared physical-unit uncertainty;
- state the empirical result: the evaluation definition materially changes measured skill in the frozen replay.

## Red-line claims

Never say:

- “We proved SEP papers are inflated.”
- “SEPNET is wrong.”
- “Our model is better than state of the art.”
- “The 85-event replay is an untouched independent test.”
- “The model is operationally deployable.”
- “The 88.95% FAR means 88.95% of quiet days trigger.” It is the false-alarm ratio, not the false-positive rate.
- “This guarantees ISEF qualification or first place.”

## ISEF-bound AI compliance

For an ISEF-bound project, the students must own the final writing and reasoning. Keep AI use as disclosed support/audit assistance, retain a record of what it did, and complete the Student Support Disclosure requirements applicable to the current ISEF rules. Generated text in this repository is a **review scaffold**, not a substitute for student-written abstract, research plan, poster or citations.

## Evidence receipt to retain outside spoken presentation

SEP-PRISM model-free confirmation:

- workflow run `34438057070`
- artifact `10136909164`
- SHA-256 `4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f`

SEP-PRISM fixed-model replay:

- workflow run `34438987251`
- source commit `296f302111371e8d421fb5f2a4569ea2c06bd6e1`
- artifact `10137507101`
- SHA-256 `81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0`
- supplied verifier and independent reconstruction: PASS

Final source-only compatibility verification:

- workflow run `34570209759` at current PR head lineage: PASS
