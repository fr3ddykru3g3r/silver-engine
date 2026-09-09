# IRIS-SEP three-minute research video

## 0:00–0:25 — The decision

Visual: one satellite, one incoming radiation-storm timeline, four operator
states. Narration: “A satellite team needs to know whether to keep operating or
prepare protection. The warning must catch a new radiation storm without
creating constant false alarms—and it must never look trustworthy when its data
are stale or broken.”

Evidence on screen: NOAA S1 boundary, >10 MeV proton flux at 10 pfu. Do not show
damage footage as though IRIS prevented a real incident.

## 0:25–0:55 — The hidden failure

Visual: four chronological folds. Three model traces complete; the original
compact trace becomes “nonfinite / rejected” in the latest fold.

Narration: “Our first neural model looked ordinary on aggregate development
metrics. When we moved it chronologically into a later data regime, it emitted
invalid numbers. A conventional pipeline could still pass that output onward.
IRIS abstains.”

## 0:55–1:25 — What IRIS changes

Visual: magnetic, eruption, proton, and XRS sources entering a forecast gate.
Beside the model, show evidence hash, publication time, freshness, schema,
uncertainty, and supported era.

Narration: “IRIS tests both the scientific forecast and whether that forecast is
admissible. Every output is VALID, DEGRADED, or ABSTAIN. An abstention contains
no probability and no protection state.”

On-screen receipt: “10,000 compound synthetic trials; 299 fault combinations;
0 status errors; 0 unsafe-valid outputs; 12/12 recoveries.” Caption throughout:
“software fault benchmark, not SEP prediction accuracy or named-competitor
comparison.”

## 1:25–1:55 — Current measured result

Visual: honest table: XGBoost 0.287, elastic net 0.276, stabilized compact 0.258.
Show fold range for XGBoost, 0.013 to 0.432.

Narration: “The current training-side result is not a victory. XGBoost remains
the strongest complete reference, and performance changes dramatically across
solar and instrument eras. Signed-log preprocessing prevented numerical failure
but did not create better skill.”

## 1:55–2:25 — The scientific hypothesis

Visual: highlighted missing blocks in the historical magnetic table, followed
by forecast-time proton and XRS context.

Narration: “The evidence points to missing physical context and source shift,
not a shortage of neural-network layers. We will test historical proton flux,
then X-ray context, under frozen chronological folds and compare every model on
the same new-crossing identities.”

## 2:25–2:50 — The decisive benchmark

Visual: two gates. Forecast gate: TSS interval, matched-detection FAR,
calibration, lead time. Validity gate: outage, stale feed, schema drift,
unsupported era, corrupted receipt.

Narration: “IRIS succeeds only if it improves the paired forecast benchmark and
survives the operational fault benchmark. A failure remains a failure; we do
not move thresholds after seeing the locked result.”

## 2:50–3:00 — Close

Narration: “The goal is not a model that is confident every day. It is evidence
a satellite operator can audit—and a system that knows when it cannot know.”

End card: Research decision support. No spacecraft control. Final blinded
evaluation pending.
