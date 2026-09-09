# IRIS-SEP internal 90+ execution plan

**Date:** 2026-09-07  
**Status:** active internal competition-readiness plan  
**Boundary:** this is an internal target, not a prediction or claim of an IRIS/ISEF score or award.

## Judge-level thesis

> A solar-radiation forecast should not keep producing a normal-looking probability when important sensor evidence disappears. IRIS-SEP tests which reduced-sensor forecasts remain useful, which should be labelled DEGRADED, and when the system should ABSTAIN instead of guessing.

The project must be explainable without assuming the judge knows what an SEP is.

## Internal target score

We use a 100-point ISEF-like internal rubric only as an engineering checklist.

| Dimension | Current target | 90+ requirement |
|---|---:|---|
| Question / scientific design | 10/10 | One fixed NEW-SEP target; causal chronology; no hidden leakage |
| Method / execution | 19/20 | Reproducible model package; explicit provenance; independent audit; all negative results preserved |
| Creativity / impact | 19/20 | Demonstrate a useful VALID/DEGRADED/ABSTAIN rule and fixed analyst-review-budget benefit |
| Analysis / defensibility | 19/20 | Rare-event metrics, uncertainty, baselines, matched identities, untouched final evaluation |
| Presentation / interview | 33/35 | One sentence, one diagram, one headline number, limitation-first answers, student can defend every scientific choice |
| **Target** | **100 possible** | **>=90 only if every scientific gate below is closed** |

## What we have now

Development-only evidence already shows:

- full-input forecast produces nontrivial discrimination but a very high row-level false-alarm ratio, so raw alerting is not a sufficient operator story;
- a pre-trained reduced-sensor fallback can earn DEGRADED status when either XRS or proton-context information is unavailable;
- losing both XRS and proton-context fails the predeclared detection gate and therefore forces ABSTAIN;
- no outage-time imputation, reconstruction or retraining is needed for that fallback decision;
- the locked test remains untouched.

The 24/72/168 terminal fallback rows are identical because the availability-conditioned model depends on information available at the issue time, not on how long the absent feed has already been unavailable. We preserve those receipts for audit continuity but do not present them as three independent findings.

## 90+ scientific gates

### Gate A — human-usefulness metric

Freeze a primary operator review budget before locked-test access.

Primary metric:

- review highest-risk 5% of eligible issue days;
- report fraction of NEW-SEP positive issue days captured;
- report precision and enrichment versus random review at identical workload;
- report approximately equivalent review-days per 365 issue-days;
- repeat for FULL, NO_XRS, NO_PROTON and NO_XRS_OR_PROTON states;
- state-level permission remains independent of ranking: a state that fails the safety gate remains ABSTAIN even if a ranking metric looks nonzero.

**Pass condition for final evidence:** the untouched evaluation must show material enrichment over random review for FULL, with reduced-sensor behavior consistent with the frozen DEGRADED/ABSTAIN policy. No exact numerical superiority threshold will be invented after seeing the test.

### Gate B — comparator fairness

On identical issue identities, report at minimum:

- climatology;
- causally available persistence where defined;
- elastic net;
- ordinary XGBoost;
- current cross-fitted evidence stack;
- aligned external SEP comparator only if its target and available inputs can be made genuinely comparable.

No comparator may receive future information or a different event definition.

### Gate C — provenance / causality

Every headline feature must be either:

1. demonstrated causally available at issue time; or
2. explicitly marked UNKNOWN/reconstructed and excluded from claims that require native observations.

Finite aggregate cells are never treated as proof of native availability.

### Gate D — frozen inference package

Before final evaluation, export and reload without training:

- all specialist models;
- exact feature order;
- stack weights;
- calibration parameters;
- operating thresholds;
- availability-state mapping;
- operator review-budget policy;
- software/environment hashes.

The reloaded package must reproduce saved development predictions within a declared numerical tolerance and identical binary decisions.

### Gate E — untouched evaluation

Before opening any final cohort:

- freeze code SHA;
- freeze package hash;
- freeze target and issue identities through a custodian/overlap attestation;
- freeze comparator definitions;
- freeze primary MAX_TSS policy and 5% operator-review-budget metric;
- freeze DEGRADED/ABSTAIN rules;
- freeze required plots and tables.

Open once. A failure remains a failure.

### Gate F — judge compression

Final board/video/interview must answer only five questions before details:

1. **Problem:** what if a solar-warning sensor disappears?
2. **Experiment:** remove information under controlled chronology and compare frozen forecast states.
3. **Discovery:** one missing feed can sometimes support a degraded warning; too much missing evidence should make the system refuse to guess.
4. **Human value:** can the probability ranking concentrate events into a small fixed set of days worth analyst attention?
5. **Limitation:** rare events make false alarms and uncertainty unavoidable; this is a research prototype, not operational certification.

## Judge-killer questions we must answer in one breath

### "Your false-alarm ratio is huge. Why is this useful?"

Do not dodge the number. Explain that rare daily events make row-level alert precision difficult, which is why the project separately measures ranking under a fixed human review budget and whether missing evidence should trigger DEGRADED or ABSTAIN. Final usefulness depends on untouched review-budget evidence, not hiding FAR.

### "Why not just fill the missing sensor value?"

Because a plausible value can preserve a plausible-looking probability without preserving decision quality. Availability-conditioned fallbacks avoid inventing measurements and let the system explicitly degrade or abstain.

### "Why are 24 h, 72 h and 168 h fallback results identical?"

Because that experiment asks what can be predicted from the sensors available at the current issue time. The fallback uses no history from the missing feed, so outage duration does not alter its inputs. Duration belongs to reconstruction/forward-fill stress tests, not to the state-only fallback claim.

### "Did AI build this for you?"

Answer factually from the student-assistance ledger. The students must be able to derive the target, explain chronology, describe every model family, reproduce the core experiment, interpret TSS/FAR/Brier/ECE/review enrichment, and defend all limitations without relying on generated text.

## What must NOT consume time now

- larger neural networks;
- full MHD simulation;
- synthetic physics added only to sound sophisticated;
- extra metrics with no operator interpretation;
- cosmetic UI before the locked evaluation protocol is frozen;
- award predictions.

## Definition of winner-ready

The project is internally called **90+ ready** only when:

- all Gates A–F are closed;
- the locked test is still untouched until Gate E;
- the main conclusion can be shown in one plot plus one reliability/state diagram;
- every headline number is backed by a hash-pinned receipt;
- the students can defend the project under hostile questioning.

Until then, additional code volume does not increase the score.
