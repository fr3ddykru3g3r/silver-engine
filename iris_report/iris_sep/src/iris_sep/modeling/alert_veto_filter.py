"""Monotone alert-veto policy for missing-feed V3 forecasts.

The veto layer can only suppress an alert already produced by the frozen V3
MAX_TSS policy.  It can never create a new alert.  This guarantees that false
positives cannot increase relative to the frozen V3 alert set for a fixed
cohort, although detections can be lost if the veto is too aggressive.

This is development architecture work.  The score cohort was already inspected
before this policy was proposed, so score results are never independent evidence.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .alert_decision_filter import _prob


def select_veto_threshold(
    *,
    labels: Sequence[int],
    decision_score: Sequence[float],
    baseline_alert: Sequence[bool],
    max_true_positive_loss: int = 1,
) -> dict[str, float | int]:
    """Minimise false positives within the existing V3 alert set.

    Threshold selection is intended for the frozen threshold role only.  The
    candidate alert is always ``baseline_alert & (decision_score >= threshold)``.
    """
    y = np.asarray(labels, dtype=int)
    score = _prob(decision_score, "decision_score")
    base = np.asarray(baseline_alert, dtype=bool)
    if y.ndim != 1 or base.ndim != 1 or len(y) != len(score) or len(base) != len(y):
        raise ValueError("veto threshold vectors must be aligned")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("veto threshold labels must be binary")
    if not isinstance(max_true_positive_loss, int) or max_true_positive_loss < 0:
        raise ValueError("max_true_positive_loss must be nonnegative")

    baseline_tp = int(np.sum((y == 1) & base))
    baseline_fp = int(np.sum((y == 0) & base))
    if baseline_tp < 1:
        raise ValueError("baseline alert set needs at least one true positive")
    minimum_tp = max(1, baseline_tp - max_true_positive_loss)

    # Include a threshold below all observed values so the no-veto baseline is
    # always a legal candidate.  The selection therefore cannot be forced to
    # worsen the threshold-role alert set.
    candidates = np.unique(np.concatenate(([np.nextafter(float(np.min(score)), -np.inf)], score)))
    best = None
    for threshold in candidates:
        alert = base & (score >= float(threshold))
        tp = int(np.sum((y == 1) & alert))
        fp = int(np.sum((y == 0) & alert))
        if tp < minimum_tp:
            continue
        # Minimise FP first, retain more TP second, then choose the stricter
        # threshold when otherwise tied.
        key = (fp, -tp, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), tp, fp)
    if best is None:
        raise ValueError("no veto threshold satisfies TP-retention constraint")

    return {
        "threshold": best[1],
        "true_positives": best[2],
        "false_positives": best[3],
        "baseline_true_positives": baseline_tp,
        "baseline_false_positives": baseline_fp,
        "minimum_true_positives": minimum_tp,
        "max_true_positive_loss": int(max_true_positive_loss),
        "new_alerts_outside_v3_allowed": False,
    }


def apply_veto(
    *,
    baseline_alert: Sequence[bool],
    decision_score: Sequence[float],
    threshold: float,
) -> np.ndarray:
    base = np.asarray(baseline_alert, dtype=bool)
    score = _prob(decision_score, "decision_score")
    if len(base) != len(score):
        raise ValueError("veto vectors must be aligned")
    out = base & (score >= float(threshold))
    if np.any(out & ~base):
        raise AssertionError("veto created an alert outside V3")
    return out
