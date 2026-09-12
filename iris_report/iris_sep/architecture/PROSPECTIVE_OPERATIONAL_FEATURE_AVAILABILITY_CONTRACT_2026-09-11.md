# Prospective operational feature-availability contract — 2026-09-11

**Study:** `IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1`  
**Status:** `FROZEN_INPUT_RULES — OUTCOMES SEALED — EXECUTION NOT AUTHORIZED YET`

## Why this exists

The historical SEP-PRISM replay confirms that evaluation definitions matter, but it does not prove that every historical predictor was available at its nominal issue time. The next confirmation therefore treats **forecast-time feature availability as part of the scientific hypothesis**, not as an implementation detail.

The protected post-`2025-09-10T00:00:00Z` outcome pool remains sealed. This contract may inspect public input interfaces and software only. It may not query protected labels, positive counts, event identities, model scores, or power.

## Fail-closed rule

A model may enter the prospective evaluation only after **every input field** has a frozen availability receipt containing:

1. canonical feature name;
2. source and endpoint/archive identity;
3. measurement timestamp semantics;
4. first-seen/retrieval timestamp semantics;
5. units;
6. quality/fill rule;
7. correction/backfill policy;
8. maximum age accepted at issue time;
9. evidence that the value was available before the frozen prediction deadline; and
10. a hash of the feature schema/receipt.

If one required field is missing, late, revised only after issue time, or dependent on a retrospective event catalogue, that model **ABSTAINS**. There is no imputation from future data and no post-outcome feature substitution.

## Current source audit

NOAA SWPC currently exposes a primary-GOES JSON directory containing both `integral-protons-1-day.json` and `xrays-1-day.json`, plus a GOES `instrument-sources.json` resource. The SWPC user guide describes real-time GOES proton plots as five-minute averaged integral flux at >1, >10, >30 and >100 MeV and X-ray measurements in two bands. These establish that relevant real-time streams exist; they do **not** by themselves establish a guaranteed publication latency or a historically identical interface.

Accordingly:

| Candidate input | Current status | Prospective use |
|---|---|---|
| primary GOES >10 MeV integral proton flux | `SOURCE_PRESENT — LATENCY RECEIPT REQUIRED` | may support deterministic past-proton-active proxy after first-seen capture is demonstrated |
| primary GOES XRS | `SOURCE_PRESENT — LATENCY/SCHEMA RECEIPT REQUIRED` | candidate for a future operational joint model only after complete field-level verification |
| SEP-PRISM 259 predictor-side columns | `BLOCKED_AS_A_SET` | historical publication-time availability is not established; cannot be copied into the prospective model wholesale |
| retrospective SEP/event catalogue boundaries | `OUTCOME_ONLY` | may define custodian labels after freeze; never a predictor |

Source directories frozen for the availability audit:

- `https://services.swpc.noaa.gov/json/goes/primary/`
- `https://services.swpc.noaa.gov/json/goes/instrument-sources.json`

Prospective acquisition must preserve the raw response bytes, retrieval time, observation times, source identity and SHA-256 before feature extraction.

## Prediction clock

The scientific issue time is **00:00 UTC daily**. A prospective model must use only values demonstrably available before that instant. To avoid pretending a measurement arriving exactly at 00:00 was known at 00:00, implementation should freeze a pre-issue snapshot and use only measurements present in that captured response.

The append-only prediction record must contain issue time, prediction creation time, model ID, model/rule SHA-256, feature-schema SHA-256, raw-input receipt hashes, probability and frozen binary alert, or explicit `ABSTAIN` if the interface contract is not satisfied.

Prediction creation time must be no later than issue time. Predictions may not be regenerated after the target window is known.

## Minimal causal comparator

The required prospective comparator is `past_proton_active_proxy`. It is intentionally simple: using a frozen pre-issue >10 MeV proton measurement, it alerts only when the input indicates >=10 pfu. Its purpose is not to be a good forecaster; it tests whether ordinary occurrence scoring rewards recognition of an already-active storm.

This rule may not be marked `features_verified_causal=true` until the source-readiness receipt demonstrates that its exact input was captured before issue time. The custodian evaluator rejects an execution manifest without that verification.

## Optional operational learned models

A `joint_operational_model` and `proton_free_operational_model` are optional. They may be added only **before any protected outcome access**, after:

1. their exact causal feature allowlist is verified;
2. training uses development-exposed data only;
3. model objects are serialized and hashed;
4. thresholds are selected without the protected cohort;
5. the environment and feature schema are hashed; and
6. the execution manifest is changed from template status to `EXECUTION_FROZEN`.

If this cannot be achieved, the prospective confirmation proceeds only with the model-free episode statistics and deterministic proton-state comparator; it must not invent a learned-model result after unsealing.

## Outcome construction — custodian only

The custodian constructs the >10 MeV, >=10 pfu target after the analysis is frozen:

- already active at issue: persistence, never an onset positive;
- new crossing: crossing occurs in `(issue, issue+24h]` while not active at issue;
- ambiguous physical mapping: descriptive reporting only, not arbitrarily assigned to a bootstrap episode;
- positive bootstrap unit: distinct qualifying onset episode;
- negative bootstrap unit: frozen Monday-anchored seven-day quiet block.

No protected event timestamp, label, count or episode identity is returned to the development side before conclusion freeze.

## Information floor

Confirmatory inference requires at least **50 distinct onset episodes and 500 quiet blocks**. This floor was frozen without looking at protected counts. It is a minimum-information rule, not a promise of statistical power.

If the floor is not met, the only allowed disposition is `INSUFFICIENT_CONFIRMATORY_INFORMATION`. No threshold, feature set, event definition or model may be changed to escape that outcome.

## Primary prospective contrast

The required comparator contrast is:

`past_proton_active_proxy: NEW_ONSET_CAUSAL TSS - MAPPED_OCCURRENCE TSS`

using 10,000 shared physical-unit bootstrap draws, seed `20260911`.

A prospective confirmation requires the information floor **and** a 95% paired percentile interval strictly below zero. Optional frozen operational-model contrasts must be reported regardless of direction, but cannot be added after outcome access.

## Claim boundary

Even a successful run would support only the narrow statement that, under this frozen prospective protocol, occurrence and causal-onset scoring measure materially different behavior. It would not establish deployment readiness, state-of-the-art forecasting, universal bias in prior work, economic savings or an award outcome.
