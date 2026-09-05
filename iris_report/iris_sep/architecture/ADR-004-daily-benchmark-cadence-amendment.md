# ADR-004: Narrow the first locked benchmark to a daily issue cadence

Status: accepted on 2026-09-04 before SEPVAL identities, outcomes, predictions,
or scores were accessed.

## Evidence and correction

The original v1 contract specified hourly issue times before the model-ready
table was inspected. Header and publisher-source inspection later established
that the comparison dataset represents non-overlapping 24-hour predictor
windows followed by 24-hour forecast windows. The safe V1 publisher training
file also has a daily cadence, with missing-day gaps.

Treating those rows as hourly forecasts would manufacture 23 issue times for
which the benchmark supplies no row. It would also make comparison with
SEPNET-O inequitable. This is a design correction based on cadence metadata,
not on any model result or locked-test outcome.

## Decision

The first IRIS-SEP scientific benchmark issues one forecast per available daily
window and predicts a new operational threshold crossing during the following
24 hours. Every model, including SEPNET-O, must use the identical issue rows.

Hourly updating remains a future prospective-operational study. It cannot be
claimed from this retrospective benchmark. Company-facing material must say
"daily 24-hour hazard forecast" until an independently receipted hourly cohort
and latency audit exist.

The benchmark remains primary-only. AIA, image embeddings, peak flux,
time-to-onset, and >100 MeV heads stay disabled until their own data and
validation contracts pass.

## Claim boundary

This amendment does not authorize the publisher's legacy V1 label as a final
target. Final evaluation still requires an audited new-crossing target,
already-enhanced exclusion, authoritative episode grouping, source-publication
latencies, and the paired locked gate.
