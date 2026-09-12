# IRIS-SEP FPR controller V2 future-freeze readiness — 2026-09-12

Status: **READY AS A FROZEN CONTROLLER CANDIDATE; PROSPECTIVE EXECUTION INTENTIONALLY BLOCKED**

Branch: `codex/iris-sep-fpr-controller-v2-20260912`

Freeze-validation workflow run: `34676802192` — **SUCCESS**.

## What is now frozen

The V2 adaptive alert policy is frozen for any future uninspected evaluation:

- causal rolling negative-score quantile controller
- quantile = 0.80
- 120-day history
- minimum 30 previously resolved negative rows
- 24-hour forecast horizon plus one-hour label-availability buffer
- NumPy `higher` quantile rule
- source model frozen fixed threshold is the controller floor

No further controller-parameter selection on the exposed 2017 Phase II rows may support a fresh performance claim.

## Source-only verification completed

CI passed the V2 unit tests, controller boundary tests, and added causal-invariance tests. The tests verify that changing future labels or future scores cannot alter earlier controller outputs, changing the current target label cannot alter the current threshold/alert, and only fully resolved negative rows can enter the controller's negative-score history.

The same CI validates the future-evaluation freeze contracts and confirms that the prospective execution template currently fails closed. The source-only validator performs no protected-outcome, protected-event-count, protected-event-identity, or protected-model-score access.

## Evidence boundary

The 2017 Phase II score cohort, FPR-controller V1 replay, and FPR-controller V2 replay are development-exposed and are registered in `config/inspected_evidence_registry_v3.json`. They may be used for reproducibility and clearly labelled post-hoc engineering, but not as fresh validation or independent confirmation.

The V2 retrospective development result remains contextual only: FPR 0.2392, POD 1.0, TSS 0.7608 on 210 rows with only one positive onset. It is not an operational-performance estimate.

## Current execution blockers

Prospective confirmation must remain blocked until both of the following exist before any protected scoring:

1. a separately frozen issue-time-causal source model package with model/rule hash, feature-schema hash, fixed threshold, training boundary, software-environment receipt, and per-feature causal-availability verification; and
2. an independent adult custodian who completes and freezes the execution manifest without protected-outcome feedback to the development side.

The protected pool beginning 2025-09-10 remains sealed from development-side outcome inspection. The prospective information floor remains at least 50 distinct onset episodes and 500 quiet blocks; otherwise the result must be `INSUFFICIENT_CONFIRMATORY_INFORMATION`.
