# IRIS-SEP scientific decision changelog

This file records **scientific and systems-engineering decisions**, including failed ideas and negative results. It is intentionally not a conventional release-note file. A result stays recorded even when it is later superseded.

The objective is fixed unless a separately documented protocol revision says otherwise:

> Forecast the probability of a **NEW >10 MeV, >=10 pfu SEP threshold crossing within the next 24 hours**, while separately deciding whether the available inputs justify exposing that probability as VALID, DEGRADED, or ABSTAIN.

The project does **not** claim operational certification, guaranteed superiority, economic savings, an award outcome, or a breakthrough before independent evidence supports such a statement.

---

## 2026-09-06 — Contiguous outage benchmark v1 failed before scientific scoring

**Decision:** preserve the failed run and replace the placement semantics in a separately preregistered v2; do not weaken the outage-duration requirement or search for a convenient successful interval.

**What was attempted**

- frozen `IRIS_CROSSFIT_EVIDENCE_STACK_V1`;
- complete XRS, proton, and combined XRS+proton synthetic outages;
- durations 24 h, 72 h, 168 h;
- five deterministic, label-blind blocks per scenario;
- no retraining, recalibration, rethresholding, or outage-row meta fitting;
- recovery arms retained from prior work: mask-aware no fill, train-fit median, causal forward-fill.

**Failure:** the runner attempted to place an instrument outage inside the already-filtered eligible score-row sequence and raised `ValueError: no contiguous 24-hour score-row block is available` before producing scientific output.

**Root-cause decision:** this is an experiment-design bug, not evidence against the model. An instrument outage exists on the **source/instrument clock**, whereas eligible forecast rows legitimately contain gaps created by active-event exclusion and purging. V2 therefore places outage intervals on the full hourly source timeline and only then evaluates the eligible forecast rows whose issue times fall inside the outage.

**Immutable failed execution**

- run `33987778788`
- job `101364416536`
- head `4c70b5f089cd826f37425f8f7e87db72b466076b`
- failed artifact digest `sha256:96435328c904eb659ec8233174306287ff7142519c08e825fb6e8c6d56aeb2c`
- source tests before the failed data step: 97 passed
- locked test accessed: false

---

## 2026-09-06 — Random missingness transfer strengthens the validity-layer hypothesis

**Decision:** do not make a universal “fill missing values and forecast normally” claim. Missingness severity must influence VALID / DEGRADED / ABSTAIN status.

The promoted cross-fitted stack was frozen before synthetic hiding. At 5%, 20%, and 40% random loss of genuinely observed score-role cells, all three simple recovery arms were retained.

**Key result:** causal forward-fill minimized mean absolute probability drift at every tested random-loss level, but probability preservation did **not** imply decision-skill preservation.

Under the frozen `POD80_MIN_FAR` policy, causal forward-fill changed TSS by approximately:

- 5% loss: `-0.0454`
- 20% loss: `-0.0959`
- 40% loss: `-0.2265`, with the paired 95% interval entirely below zero.

At 40% loss the Brier degradation was also detectably positive. Therefore a reconstructed probability can look numerically close to the clean probability while the operational decision boundary has become materially less reliable.

**Scientific consequence:** forecast probability and permission to expose that forecast remain separate objects. Severe unsupported input loss should trigger DEGRADED/ABSTAIN logic rather than silent reconstruction.

**Immutable execution:** run `33987312162`, job `101363172765`, preregistered head `4716fe03fa723005f095251575c28a04742b9435`.

---

## 2026-09-06 — Generic missingness reference model rejected as an adequacy surrogate

**Decision:** a weak generic logistic model may test plumbing, but cannot support the final robustness claim. Transfer outage experiments to the promoted forecast architecture itself.

A hash-bound public NEW-crossing missingness package was built with 13,308 rows, 207 positives and 259 causal predictor features. The fixed balanced L2 logistic reference produced, on its score block, TP=0 and POD=0 at its clean threshold, with TSS `-0.1260`.

Because the reference forecast itself was inadequate, its threshold/TSS robustness results were not promoted as evidence that IRIS remains useful under missingness.

The run still established a useful recovery-method signal: causal forward-fill preserved the weak reference probabilities substantially better than no-fill or train-fit median at 5%, 20%, and 40% synthetic hiding. That observation was explicitly disclosed before the promoted-stack transfer and all three arms were retained to avoid cherry-picking.

**Immutable execution:** run `33987054599`, job `101362475080`, head `413249fad92c07c1ecbb5a29b4436320ec845b52`.

---

## 2026-09-06 — Two-state temporal expansion rejected

**Decision:** do not add an LSTM, Transformer, diffusion model, GNN, or deeper sequence model to the current daily aggregate interface.

A deliberately minimal dynamics experiment gave each specialist:

`[current 24 h state, previous 24 h state, current - previous]`

while leaving the specialist family, cross-fitting, evidence fusion, calibration, and threshold protocol fixed.

The temporal representation improved some ranking metrics on the older score block (including AUROC and AUPRC) but degraded the actual frozen operational decision policy: `POD80_MIN_FAR` TSS fell from about `0.5120` to `0.4619`. On the 2023–2025 monitor it was essentially tied with the static cross-fitted stack (`~0.2350` vs `~0.2359`).

**Scientific consequence:** more temporal features increased representational capacity without producing a more robust decision system. Architecture expansion stops on this aggregate table until richer causal observations justify it.

**Immutable execution:** run `33985567816`, head `3f5013bd4723f35b6b090586d03004a2421b25d2`, artifact digest `sha256:546e49353aa0616a3b5183e10b1e7aa0421830482591c8ec7a71262ca2c4bea2`.

---

## 2026-09-06 — Cross-fitted specialist evidence stack promoted

**Decision:** `IRIS_CROSSFIT_EVIDENCE_STACK_V1` is the current development architecture.

**Architecture**

1. train independent five-seed median XGBoost specialists for solar, XRS and historical-proton evidence;
2. make four expanding chronological folds entirely inside the outer fit role;
3. generate out-of-fold specialist probabilities;
4. convert them to prevalence-centred log-odds evidence;
5. learn three nonnegative evidence weights from those out-of-fold fit predictions only;
6. refit specialists on the complete fit role;
7. use the calibration role only for one final calibration intercept;
8. use the threshold role only for the frozen decision threshold;
9. keep validity/admission outside the probability model.

Learned development weights were approximately solar `0.1269`, XRS `0.2447`, proton `0.1823`.

At the operational `POD>=0.8 / minimum-FAR` selection rule, the new stack preserved the older score result (`~0.5120` TSS versus `~0.5126` for previous late fusion) while improving the already-inspected 2023–2025 development monitor (`~0.2359` versus `~0.1894`). Monitor AUROC, Brier, ECE and matched-detection FAR also moved in the desired direction; AUPRC decreased slightly and remains disclosed. The paired monitor advantage over previous late fusion did not have a strictly positive 95% interval, so superiority was **not** established.

**Why promoted anyway:** it uses substantially more appropriate role separation: the fusion rule learns from out-of-fold **fit-era** evidence rather than spending the small calibration block to learn architecture weights. This is a structural improvement even before final independent scoring.

**Immutable execution:** run `33985243824`, head `5d90a177462401d1486891e06c6fa6794ea4fbdb`, artifact digest `sha256:ed04d5b34df094e2eb7f7db3f07b426619494b38987f1993c0faddca90d58f99`.

---

## 2026-09-06 — Hard solar-anchor residual architecture rejected

**Decision:** do not force the solar specialist coefficient to one.

The tested equation was conceptually:

`logit(p) = logit(p_solar) + b + w_xrs * e_xrs + w_proton * e_proton`

with nonnegative bounded context corrections.

The context weights collapsed and the model was materially worse than the existing late-fusion system on the already-inspected monitor; the paired `POD>=0.8` TSS difference was about `-0.255` with the full 95% interval below zero.

**Scientific consequence:** XRS and historical proton evidence are not merely small corrections to a fixed solar baseline in this dataset. Each modality should be allowed to provide independent forecast evidence before fusion.

**Immutable execution:** run `33984943106`, head `8b008f915fa635e602265e52ad68cbddaec3c380`, artifact digest `sha256:1f7e6318bb797525160433c90760bec7692a25932420cefcdfdc5776f070db38`.

---

## 2026-09-05 to 2026-09-06 — Compact neural fusion not promoted

**Decision:** do not scale the compact gated neural architecture merely by adding attention, layers, or parameters.

The compact model separately encoded magnetic, eruption and particle-context inputs but combined branch embeddings through a softmax-weighted representation bottleneck. Earlier train-only rolling diagnostics showed XGBoost around `0.287` TSS, elastic net around `0.276`, and the signed-log compact neural model around `0.258`; paired intervals did not establish a neural advantage. A retained failure also produced only 1,062 finite logits out of 2,120 in one fold/seed before later numerical hardening.

**Scientific consequence:** architecture prestige is not evidence. With rare events and aggregate tabular inputs, simpler specialist models are the stronger engineering prior until richer observations create a genuine spatiotemporal representation problem.

---

## 2026-09-05 onward — Missing data split into structural absence vs transient outage

**Decision:** never treat all NaNs as the same phenomenon.

- **Structural absence:** a measurement did not exist in that instrument/source era or is outside declared source support. It may not be synthetically “reconstructed” and relabelled as observed truth.
- **Transient outage:** a source that should exist becomes temporarily unavailable. This is eligible for controlled recovery experiments.
- **Real alternate source:** if a causally available comparable observation exists, prefer the real observation before any synthetic reconstruction.

Every recovery value must retain provenance. The validity layer must be able to distinguish observed, alternate-source, reconstructed, stale, unsupported and unavailable inputs.

---

## 2026-09-05 onward — Forecast validity separated from forecast skill

**Decision:** the model may output a scientifically plausible probability and the system may still refuse to expose it as a normal forecast.

The validity/admission layer checks properties including source availability, freshness, source revision, source-era support, schema compatibility, extreme/out-of-support inputs, output finiteness, uncertainty completeness, model/policy/calibration/threshold bindings, and evidence-receipt integrity.

The intended states are `VALID`, `DEGRADED`, and `ABSTAIN`.

A deterministic software-safety benchmark already exercised 10,000 trials spanning 299 unique fault combinations and 12 fault-to-recovery sequences with zero status errors and zero unsafe-valid outputs. That is software-safety evidence only, not SEP forecast-skill evidence or operational certification.

---

## Frozen evaluation discipline

These rules are not to be changed because a model score is disappointing:

- target: NEW >10 MeV, >=10 pfu crossing within 24 h;
- already-above-threshold issues excluded;
- chronological episode-disjoint evaluation with a 24 h purge;
- fit / calibration / threshold / score roles kept separate;
- primary metric: TSS;
- matched-detection false-alarm diagnostics reported;
- paired 10,000-replicate bootstrap by complete SEP episode or predeclared quiet block;
- calibration and threshold selection cannot use the score/locked-test outcomes;
- failed and negative experiments remain preserved;
- inspected development score/monitor blocks cannot later be relabelled as fresh final evidence;
- locked test remains untouched during development.

---

## Current engineering priority

Do **not** chase a larger backbone on the current aggregate table.

Priority order:

1. repair and rerun the contiguous whole-modality outage benchmark using the real source clock;
2. turn the missingness result into a preregistered, independently testable DEGRADED/ABSTAIN policy rather than fitting a policy to already-inspected outcomes;
3. reproduce and compare against the strongest same-target external systems on identical cohort/date semantics, including SEPNET/SEPNET-PRISM and NOAA/SWPC where a faithful comparator can be constructed;
4. ingest richer causal high-cadence XRS/proton observations only through a frozen source/latency/provenance contract;
5. run untouched final evaluation only after architecture, source contract, calibration and threshold rules are frozen.

The optimization goal is to outperform strong baselines and operational comparators **if the evidence supports it**. No metric will be tuned, relabelled, hidden, or selectively reported to manufacture that conclusion.
