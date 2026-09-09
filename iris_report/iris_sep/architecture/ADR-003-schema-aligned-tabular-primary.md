# ADR-003: Use a schema-aligned tabular primary model

Status: accepted on 2026-09-04 after header-only inspection; no data row,
identity, label value, or locked-test outcome was accessed.

## Evidence

The pinned public SEP-Prediction-V2 model-ready CSV header contains 274 columns:

- 214 `SHARP_*` magnetic columns;
- 11 `Flare_*` columns;
- 26 DONKI/CDAW CME columns;
- 4 `ProtonFlux_*` columns;
- 4 XRS columns;
- 11 `Future_*` target/auxiliary columns;
- `window_begin`, `window_end`, `OSEP_label`, and `GSEP_label`.

The source describes each row as one preceding 24-hour window summarized by
minimum, maximum, and mean statistics. It does not expose 24 hourly timesteps
per modality. Therefore, applying the temporal-convolution prototype directly
would manufacture a sequence interpretation unsupported by the data.

## Decision

The first benchmark uses a compact multibranch tabular model:

1. magnetic aggregate branch;
2. eruption branch containing flare, CME, and XRS aggregates;
3. particle-context branch containing causal current-window proton aggregates.

Each branch receives values and feature-level missingness masks. Small branch
embeddings enter availability-aware gated late fusion and one primary occurrence
head. The default model has no temporal convolution, transformer, image
embedding, AIA branch, or auxiliary prediction head.

Temporal models remain a future experiment only if an independently receipted
dataset supplies real within-window timestamps or sequences. They are not part
of the first locked comparison.

## Leakage exclusions

- Every `Future_*` column is a target or auxiliary outcome and is forbidden as
  an input.
- `window_end` defines the issue time; all source publication times must be no
  later than it.
- Rows already above the operational threshold at issue time are excluded or
  separately reported according to the frozen new-crossing contract.
- Current-window particle context is allowed only as a causal input and must be
  covered by the already-enhanced and source-latency audits.
- SEPVAL rows remain outside tuning and were not inspected to make this
  decision.

## Rationale

This architecture matches the actual information content, has far fewer
parameters, is easier to reproduce on a T4, and makes the ablation question
clear: does physically grouped late fusion reduce false alarms beyond classical
models and reproduced SEPNET-O on identical issue identities?
