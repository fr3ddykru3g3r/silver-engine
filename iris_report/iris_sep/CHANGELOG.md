# IRIS-SEP scientific decision changelog

This is a **scientific decision log**, not marketing release notes. Failed ideas, negative results, implementation failures that affect interpretation, and superseded assumptions remain recorded rather than being erased.

Frozen objective unless a separately preregistered revision says otherwise:

> Forecast the probability of a **NEW >10 MeV, >=10 pfu SEP threshold crossing within the next 24 hours**, while separately deciding whether the available inputs justify exposing that probability as `VALID`, `DEGRADED`, or `ABSTAIN`.

The project does **not** claim operational certification, guaranteed superiority, economic savings, an award outcome, or a breakthrough before independent evidence supports it.

---

## 2026-09-06 — Source-clock audit: the model-ready table is daily, not hourly

**Decision:** separate two scientifically different outage questions.

1. **Daily forecast-input outage:** a complete XRS/proton modality is unavailable at one or more daily forecast issue times in `rolling_combinded_seq_24hours.csv`. This can be tested directly on the current model-ready table.
2. **True sensor outage:** the underlying XRS/proton instrument stream is unavailable for 24/72/168 continuous hours. This cannot be faithfully represented by hiding 24/72/168 rows of the model-ready table; it requires the upstream high-cadence source data and reaggregation pipeline.

The public SEPNET-PRISM repository documents `Data/rolling_combinded_seq_24hours.csv` as **one UTC day per row**, with past-24-hour summary features. Its preprocessing documentation also states that GOES proton/XRS streams are fused/resampled on a **5-minute grid** before daily aggregation.

This discovery supersedes the earlier assumption that the model-ready `window_end` clock was hourly. It does **not** change the forecast target or evaluation roles.

**Engineering consequence:**

- daily-table robustness uses 1/3/7 consecutive forecast cycles for 24/72/168-hour spans;
- raw-sensor robustness will be a separate high-cadence experiment that masks the upstream 5-minute measurements and recomputes only causally available aggregates;
- no result from one experiment may be described as evidence for the other.

---

## 2026-09-06 — Contiguous outage V2 rejected before scientific scoring

**Decision:** preserve the failed V2 run; do not reinterpret zero admissible blocks as a model result.

V2 fixed the first benchmark's eligibility-filter mistake by placing outages on the complete model-ready source clock before projecting onto eligible score rows. Unit tests passed and the hash-pinned public inputs were verified. The real-data run then found **zero** admissible 24-hour sequences under the assumed 1-hour source cadence and stopped before any outage scenario was scored.

That failure exposed the daily-cadence fact above. It is therefore an **input-semantics failure**, not evidence for or against IRIS forecast robustness.

Immutable execution:

- run `34043978773`
- job `101515620974`
- head `ebc1b4edd192f5f424ab9472ef92ad21541de844`
- preregistration SHA256 `8178fb99cbee88288756a65801f741dd00052f9769d6f1a1b6b48a4a158700a5`
- runner SHA256 `55ffa253487f381205baf56d020848bbdc9de02ed5a1ad9957740bc23ac75d61`
- 27 source/recovery tests passed before data execution
- locked test accessed: false
- monitor used: false

An earlier V2 CI attempt, run `34043907218`, failed even earlier because a depth-1 checkout could not verify preregistration ancestry. Only workflow plumbing was changed afterward; no data or scientific score had been read.

---

## 2026-09-06 — Contiguous outage V1 failed before scientific scoring

**Decision:** preserve the failed run; never weaken the requested outage duration or search for a convenient interval after failure.

V1 attempted complete XRS, proton, and combined XRS+proton outages for 24/72/168 hours with five deterministic label-blind blocks, frozen model/calibration/thresholds, and all three recovery arms. It tried to place an outage inside the already-filtered eligible score-row sequence and failed because eligibility/purge gaps break row continuity.

The correct lesson was that **instrument/source availability must be defined before target eligibility filtering**. V2 then revealed that the model-ready clock itself had also been misidentified as hourly.

Immutable execution:

- run `33987778788`
- job `101364416536`
- head `4c70b5f089cd826f37425f8f7e87db72b466076b`
- failed artifact digest `sha256:96435328c904eb659ec8233174306287ff7142519c08e825fb6e8c6d56aeb2c`
- 97 tests passed before the failed data step
- locked test accessed: false

---

## 2026-09-06 — Promoted-stack random missingness strengthens the validity hypothesis

**Decision:** do not make a universal “fill missing values and forecast normally” claim. Missingness severity must be visible to the validity layer.

The promoted cross-fitted stack was frozen before hiding 5%, 20%, and 40% of genuinely observed score-role cells. All three simple recovery arms were retained: mask-aware no-fill, train-fit median, and causal forward-fill.

Causal forward-fill minimized mean absolute probability drift at every tested loss level, but probability preservation did **not** imply decision-skill preservation. Under the frozen `POD80_MIN_FAR` policy, causal forward-fill changed TSS by approximately:

- 5% loss: `-0.0454`
- 20% loss: `-0.0959`
- 40% loss: `-0.2265`, with the paired 95% interval entirely below zero.

At 40% loss, Brier degradation was also detectably positive.

**Scientific consequence:** a reconstructed forecast can look numerically similar while becoming materially less reliable at the operational decision boundary. Forecast probability and permission to expose that forecast remain separate scientific objects.

Immutable execution: run `33987312162`, job `101363172765`, preregistered head `4716fe03fa723005f095251575c28a04742b9435`.

---

## 2026-09-06 — Generic missingness reference model rejected as an adequacy surrogate

**Decision:** a weak generic logistic model can test plumbing but cannot support the final robustness claim.

A hash-bound public NEW-crossing missingness package contained 13,308 rows, 207 positives and 259 causal predictor features. The fixed balanced L2 logistic reference produced TP=0, POD=0 and TSS `-0.1260` on its score block.

Its threshold/TSS robustness results were therefore not promoted. It did provide a method signal—causal forward-fill preserved probabilities better than no-fill or train-fit median—which was disclosed before transfer to the promoted stack. All arms were retained to prevent cherry-picking.

Immutable execution: run `33987054599`, job `101362475080`, head `413249fad92c07c1ecbb5a29b4436320ec845b52`.

---

## 2026-09-06 — Two-state temporal expansion rejected

**Decision:** do not add an LSTM, Transformer, diffusion model, GNN, or deeper sequence model to the current daily aggregate interface.

The bounded temporal experiment gave each specialist:

`[current 24 h state, previous 24 h state, current - previous]`

while leaving specialist family, cross-fitting, fusion, calibration and threshold rules fixed.

Ranking improved on the older score block, but the actual frozen operational policy worsened: `POD80_MIN_FAR` TSS fell from about `0.5120` to `0.4619`. On the 2023–2025 monitor it was essentially tied/slightly worse (`~0.2350` vs `~0.2359`).

**Scientific consequence:** extra representation capacity did not produce a more robust decision system. Architecture expansion stops on the daily aggregate table until richer causal observations justify it.

Immutable execution: run `33985567816`, head `3f5013bd4723f35b6b090586d03004a2421b25d2`, artifact digest `sha256:546e49353aa0616a3b5183e10b1e7aa0421830482591c8ec7a71262ca2c4bea2`.

---

## 2026-09-06 — Cross-fitted specialist evidence stack promoted

**Decision:** `IRIS_CROSSFIT_EVIDENCE_STACK_V1` is the current development architecture.

Architecture:

1. independent five-seed median XGBoost specialists for solar, XRS and historical-proton evidence;
2. four expanding chronological folds entirely inside the outer fit role;
3. out-of-fold specialist probabilities converted to prevalence-centred log-odds evidence;
4. three nonnegative fusion weights learned from OOF fit predictions only;
5. specialists refit on the complete fit role;
6. calibration role used only for one final calibration intercept;
7. threshold role used only for the frozen decision threshold;
8. validity/admission stays outside the probability predictor.

Learned development weights were approximately solar `0.1269`, XRS `0.2447`, proton `0.1823`.

At the `POD>=0.8 / minimum-FAR` rule, the cross-fitted stack preserved the older score result (`~0.5120` TSS vs `~0.5126` previous late fusion) while improving the already-inspected 2023–2025 development monitor (`~0.2359` vs `~0.1894`). Monitor AUROC, Brier, ECE and matched-detection FAR also improved; AUPRC decreased slightly and remains disclosed. The paired monitor TSS interval still crossed zero, so superiority was **not** established.

**Why promoted:** fusion architecture is learned from out-of-fold fit-era evidence rather than spending the small calibration block to learn model weights. This is a cleaner statistical architecture independent of the monitor point estimate.

Immutable execution: run `33985243824`, head `5d90a177462401d1486891e06c6fa6794ea4fbdb`, artifact digest `sha256:ed04d5b34df094e2eb7f7db3f07b426619494b38987f1993c0faddca90d58f99`.

---

## 2026-09-06 — Hard solar-anchor residual architecture rejected

**Decision:** do not force the solar specialist coefficient to one.

Tested form:

`logit(p) = logit(p_solar) + b + w_xrs * e_xrs + w_proton * e_proton`

with nonnegative bounded context corrections.

Context weights collapsed, and monitor paired `POD>=0.8` TSS difference versus previous late fusion was about `-0.255` with the full 95% interval below zero.

**Scientific consequence:** XRS and historical proton evidence are not merely small corrections to a hard solar baseline. Each modality should provide independent evidence before fusion.

Immutable execution: run `33984943106`, head `8b008f915fa635e602265e52ad68cbddaec3c380`, artifact digest `sha256:1f7e6318bb797525160433c90760bec7692a25932420cefcdfdc5776f070db38`.

---

## 2026-09-05 to 2026-09-06 — Compact neural fusion not promoted

**Decision:** do not scale the compact gated neural architecture merely by adding attention, layers, or parameters.

The compact model separately encoded magnetic, eruption and particle-context inputs but forced them through a softmax-weighted representation bottleneck. Earlier train-only rolling diagnostics gave approximately XGBoost `0.287` TSS, elastic net `0.276`, and signed-log compact neural `0.258`; paired intervals did not establish neural advantage. One retained failure produced only 1,062 finite logits out of 2,120 before later numerical hardening.

**Scientific consequence:** architecture prestige is not evidence. With rare events and aggregate tabular inputs, specialist models are the stronger engineering prior until a richer observation interface creates a genuine sequence/spatiotemporal problem.

---

## 2026-09-05 onward — Missing data split into structural absence vs transient outage

**Decision:** never treat all missing values as the same phenomenon.

- **Structural absence:** a measurement did not exist in that source era or lies outside declared support. It may not be reconstructed and relabelled as observed truth.
- **Transient outage:** a source that should exist temporarily becomes unavailable. This is eligible for controlled recovery experiments.
- **Real alternate source:** if a causally available comparable observation exists, use the real observation before synthetic reconstruction.

Every recovered value retains provenance. The validity layer must distinguish observed, alternate-source, reconstructed, stale, unsupported and unavailable inputs.

---

## 2026-09-05 onward — Forecast validity separated from forecast skill

**Decision:** a model may output a plausible probability while the system still refuses to expose it as a normal forecast.

The validity/admission layer checks source availability, freshness, revision, era support, schema compatibility, out-of-support magnitude, output finiteness, uncertainty completeness, model/policy/calibration/threshold bindings, and evidence-receipt integrity.

States: `VALID`, `DEGRADED`, `ABSTAIN`.

A deterministic software-safety benchmark exercised 10,000 trials spanning 299 unique fault combinations and 12 fault-to-recovery sequences with zero status errors and zero unsafe-valid outputs. This is software-safety evidence only, not SEP forecast-skill evidence or operational certification.

---

## Frozen evaluation discipline

These rules are not changed because a score is disappointing:

- target: NEW >10 MeV, >=10 pfu crossing within 24 h;
- already-above-threshold issues excluded;
- chronological episode-disjoint evaluation with 24 h purge;
- fit / calibration / threshold / score roles separate;
- primary metric: TSS;
- matched-detection false-alarm diagnostics reported;
- paired 10,000-replicate bootstrap by complete SEP episode or predeclared quiet block;
- calibration and threshold selection cannot use score/locked-test outcomes;
- failed and negative experiments remain preserved;
- inspected development score/monitor blocks cannot be relabelled as fresh final evidence;
- locked test remains untouched during development.

---

## Current engineering priority

Do **not** chase a larger backbone on the current aggregate table.

Priority order:

1. run a preregistered **daily forecast-input outage** benchmark using 1/3/7 consecutive source days for 24/72/168-hour spans;
2. build a separate **high-cadence raw-sensor outage pipeline** from the upstream 5-minute XRS/proton sources and causally recompute the daily summaries;
3. turn missingness/outage findings into a prospective, independently testable `DEGRADED/ABSTAIN` policy rather than fitting a threshold to already-inspected outcomes;
4. reproduce and compare against strong same-target external systems on identical dates/cohorts, especially SEPNET/SEPNET-PRISM and NOAA/SWPC, with UMASEP compared only under correctly aligned horizon/event semantics;
5. run untouched final evaluation only after architecture, source contract, calibration, threshold and admission rules are frozen.

The optimization goal is to outperform strong baselines and operational comparators **if the evidence supports it**. No metric, cohort, threshold, or failure will be altered or hidden to manufacture that conclusion.
