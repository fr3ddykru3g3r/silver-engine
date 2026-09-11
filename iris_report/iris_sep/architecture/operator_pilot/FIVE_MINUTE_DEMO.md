# Five-minute operator demonstration

**DRAFT — INTERNAL REHEARSAL ONLY; DO NOT SEND WITHOUT REVIEW**

## 0:00–0:45 — the operational question

Show one forecast issue and ask: **before looking at the probability, do we know that the forecast itself is valid to expose?**

Explain that this project separates forecast probability from forecast permission. A record can be numerically formed while still relying on stale, structurally unavailable, ambiguous or causally invalid inputs.

## 0:45–1:45 — validity record

Show an example record with:

```json
{
  "issue_time": "...Z",
  "alert": true,
  "validity_state": "VALID",
  "reason_codes": [],
  "input_age_seconds": {"proton": 60},
  "source_hashes": {"proton": "..."},
  "model_hash": "...",
  "threshold_hash": "...",
  "ledger_previous_hash": "..."
}
```

Then flip one condition at a time:

- short causal recovery -> `DEGRADED`;
- stale feed -> `ABSTAIN`, `alert=null`;
- future observation -> `ABSTAIN`;
- structural unavailability -> `ABSTAIN`;
- broken chain hash -> `ABSTAIN`.

Emphasize that the reason is machine-readable and the raw source/provenance can be audited.

## 1:45–2:45 — why evaluation validity matters too

Show the graphical mechanism: one physical SEP episode can create several positive daily windows, and some issue times can occur after threshold crossing.

Then show the frozen fixed-alert comparison. With the same joint-XGBoost alerts, matched TSS is:

`0.726 mapped occurrence -> 0.621 episode-normalized -> 0.437 causal new onset`.

The point is not that one metric replaces the others. Each answers a different operational/scientific question.

## 2:45–3:30 — model honesty

State the research comparator's onset operating characteristics:

- sensitivity 48.24%;
- false-alarm ratio 88.95%;
- false-positive rate 4.51%.

Therefore this demonstration is **not** “here is our better forecaster.” It is “here is a way to know what a forecast score means and whether the forecast record was valid to expose.”

## 3:30–4:15 — missing feeds

Explain the current evidence boundary. Synthetic random-cell masking suggests causal carry-forward can preserve probability space reasonably under modest loss, but at 40% loss one operational decision policy degraded materially. That test was development-only and does not define a universal outage threshold.

The pilot therefore asks the operator what counts as stale/severe for their actual feed and decision. Source-specific limits must be frozen before prospective use.

## 4:15–5:00 — interview, not sales claim

Ask the reviewer five questions:

1. What decision changes when an SEP alert arrives?
2. What warning time is useful?
3. What false-alert/review burden is tolerable?
4. What should the system do when a required feed is stale or absent?
5. What provenance/audit fields must be retained?

Close with the measurable first milestone: **one documented workflow review and one resulting requirement incorporated into the next versioned prospective protocol.** No company-benefit claim is made before that evidence exists.
