"""Low-capacity alert-decision layer for missing-feed V3 forecasts.

This module does **not** replace or recalibrate the V3 probability.  It learns a
separate decision score from information available in the corresponding
missing-feed state and is used only to decide whether a candidate warning should
be exposed.  The development score role was already inspected before this layer
was selected, so any result produced from that role is development evidence only.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from xgboost import XGBClassifier


STATES = ("NO_XRS", "NO_PROTON")
EPS = 1e-6

# Post-hoc development-selected configurations.  They are frozen here so future
# runs cannot silently tune them.  Selection was informed by the already-
# inspected development score cohort and therefore is NOT preregistration.
STATE_MODEL_PARAMS = {
    "NO_XRS": {
        "max_depth": 2,
        "n_estimators": 20,
        "learning_rate": 0.08,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "reg_alpha": 1.0,
        "scale_pos_weight": 50.0,
    },
    "NO_PROTON": {
        "max_depth": 2,
        "n_estimators": 30,
        "learning_rate": 0.10,
        "min_child_weight": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "reg_alpha": 1.0,
        "scale_pos_weight": 50.0,
    },
}

MAX_THRESHOLD_ROLE_TP_LOSS = 1
RANDOM_STATE = 7


def _as_vector(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be one non-empty finite vector")
    return arr


def _prob(value: Sequence[float], name: str) -> np.ndarray:
    arr = _as_vector(value, name)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return arr


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def meta_features(
    *,
    v3_probability: Sequence[float],
    v1_probability: Sequence[float],
    solar_only_probability: Sequence[float],
    raw_solar_probability: Sequence[float],
    raw_context_probability: Sequence[float],
    solar_seed_range: Sequence[float],
    solar_seed_std: Sequence[float],
    context_seed_range: Sequence[float],
    context_seed_std: Sequence[float],
) -> np.ndarray:
    """Return the frozen 17-column decision feature vector.

    ``raw_context_probability`` is proton for ``NO_XRS`` and XRS for
    ``NO_PROTON``.  Every quantity is available in the missing-feed state; the
    absent family is never used or reconstructed.
    """
    p3 = _prob(v3_probability, "v3_probability")
    p1 = _prob(v1_probability, "v1_probability")
    ps = _prob(solar_only_probability, "solar_only_probability")
    rs = _prob(raw_solar_probability, "raw_solar_probability")
    rc = _prob(raw_context_probability, "raw_context_probability")
    sr = _as_vector(solar_seed_range, "solar_seed_range")
    ss = _as_vector(solar_seed_std, "solar_seed_std")
    cr = _as_vector(context_seed_range, "context_seed_range")
    cs = _as_vector(context_seed_std, "context_seed_std")
    n = len(p3)
    if any(len(arr) != n for arr in (p1, ps, rs, rc, sr, ss, cr, cs)):
        raise ValueError("alert-decision feature vectors are not aligned")
    if np.any(sr < 0) or np.any(ss < 0) or np.any(cr < 0) or np.any(cs < 0):
        raise ValueError("seed-disagreement diagnostics cannot be negative")

    l3, l1, ls, lrs, lrc = map(_logit, (p3, p1, ps, rs, rc))
    return np.column_stack([
        l3, l1, ls, lrs, lrc,
        p3, p1, ps, rs, rc,
        sr, ss, cr, cs,
        l3 - l1,
        l3 - ls,
        l1 - ls,
    ])


def build_model(state: str) -> XGBClassifier:
    if state not in STATES:
        raise ValueError(f"unsupported missing-feed state: {state}")
    params = dict(STATE_MODEL_PARAMS[state])
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
        **params,
    )


def select_decision_threshold(
    *,
    labels: Sequence[int],
    decision_score: Sequence[float],
    baseline_true_positives: int,
    max_true_positive_loss: int = MAX_THRESHOLD_ROLE_TP_LOSS,
) -> dict[str, float | int]:
    """Minimise false positives subject to a frozen TP-retention constraint."""
    y = np.asarray(labels, dtype=int)
    score = _prob(decision_score, "decision_score")
    if y.ndim != 1 or len(y) != len(score) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("threshold labels must be aligned binary values")
    if not isinstance(baseline_true_positives, int) or baseline_true_positives < 1:
        raise ValueError("baseline_true_positives must be positive")
    if not isinstance(max_true_positive_loss, int) or max_true_positive_loss < 0:
        raise ValueError("max_true_positive_loss must be nonnegative")
    minimum_tp = max(1, baseline_true_positives - max_true_positive_loss)

    best = None
    for threshold in np.unique(score):
        pred = score >= float(threshold)
        tp = int(np.sum((y == 1) & pred))
        fp = int(np.sum((y == 0) & pred))
        if tp < minimum_tp:
            continue
        key = (fp, -tp, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), tp, fp)
    if best is None:
        raise ValueError("no decision threshold satisfies the TP-retention constraint")
    return {
        "threshold": best[1],
        "true_positives": best[2],
        "false_positives": best[3],
        "baseline_true_positives": int(baseline_true_positives),
        "minimum_true_positives": int(minimum_tp),
        "max_true_positive_loss": int(max_true_positive_loss),
    }


def binary_metrics(labels: Sequence[int], alert: Sequence[bool]) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(alert, dtype=bool)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred) or len(y) == 0:
        raise ValueError("metric vectors must be aligned and non-empty")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("metric labels must be binary")
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    far = fp / (tp + fp) if tp + fp else math.nan
    tss = pod - fpr if math.isfinite(pod) and math.isfinite(fpr) else math.nan
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "POD": pod,
        "FPR": fpr,
        "FAR": far,
        "TSS": tss,
    }


def improvement_summary(reference: Mapping[str, float | int], candidate: Mapping[str, float | int]) -> dict[str, float | int | bool]:
    ref_fp = int(reference["FP"])
    cand_fp = int(candidate["FP"])
    ref_tp = int(reference["TP"])
    cand_tp = int(candidate["TP"])
    reduction = (ref_fp - cand_fp) / ref_fp if ref_fp else 0.0
    return {
        "false_positives_removed": ref_fp - cand_fp,
        "false_positive_reduction_fraction": float(reduction),
        "true_positive_delta": cand_tp - ref_tp,
        "tss_delta": float(candidate["TSS"] - reference["TSS"]),
        "no_score_detection_loss": cand_tp >= ref_tp,
        "large_development_improvement_gate": bool(cand_tp >= ref_tp and reduction >= 0.20),
    }
