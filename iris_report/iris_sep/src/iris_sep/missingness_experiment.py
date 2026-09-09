"""Executable primitives for the IRIS-SEP hidden-data experiment.

The experiment is intentionally simple:
1. start from train-only observations whose values are known;
2. hide a predeclared subset of *temporarily reconstructable* cells;
3. recover them with simple methods or a separately supplied physics method;
4. evaluate the resulting forecast on identical score identities.

This module never selects on a locked test and never treats structurally absent
historical measurements as recoverable truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .missingness_recovery import (
    apply_train_median_fill,
    causal_forward_fill,
    fit_train_medians,
)


_SCORE_ROLE = "train_only_inner_score"


@dataclass(frozen=True)
class RecoveryOutput:
    method_id: str
    values: np.ndarray
    available_mask: np.ndarray
    reconstructed_mask: np.ndarray
    unresolved_mask: np.ndarray


def _masks(observed_mask, structural_unavailable_mask) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_mask, dtype=bool)
    structural = np.asarray(structural_unavailable_mask, dtype=bool)
    if observed.ndim != 2 or observed.shape != structural.shape or observed.size == 0:
        raise ValueError(
            "observed and structural masks must be matching non-empty 2-D arrays"
        )
    if np.any(observed & structural):
        raise ValueError("a structurally unavailable cell cannot be marked observed")
    return observed, structural


def eligible_transient_cells(observed_mask, structural_unavailable_mask) -> np.ndarray:
    """Cells with real truth that may legitimately be hidden for the experiment."""
    observed, structural = _masks(observed_mask, structural_unavailable_mask)
    return observed & ~structural


def deterministic_transient_random_holdout(
    observed_mask,
    structural_unavailable_mask,
    *,
    missing_fraction: float,
    seed: int,
    row_eligibility=None,
) -> np.ndarray:
    """Hide a deterministic sample of eligible real cells, never structural gaps."""
    eligible = eligible_transient_cells(observed_mask, structural_unavailable_mask)
    if row_eligibility is not None:
        rows = np.asarray(row_eligibility, dtype=bool)
        if rows.shape != (eligible.shape[0],):
            raise ValueError("row_eligibility must have one value per row")
        eligible &= rows[:, None]
    if not math.isfinite(missing_fraction) or not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be in (0,1)")
    locations = np.flatnonzero(eligible.ravel())
    if len(locations) < 2:
        raise ValueError("at least two eligible observed cells are required")
    count = max(
        1,
        min(
            len(locations) - 1,
            int(round(len(locations) * missing_fraction)),
        ),
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(locations, size=count, replace=False)
    mask = np.zeros(eligible.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(eligible.shape)


def transient_block_holdout(
    observed_mask,
    structural_unavailable_mask,
    *,
    start_row: int,
    length_rows: int,
    feature_indices=None,
) -> np.ndarray:
    """Hide one contiguous outage block, clipped to legitimately observed cells."""
    eligible = eligible_transient_cells(observed_mask, structural_unavailable_mask)
    n_rows, n_features = eligible.shape
    if start_row < 0 or length_rows <= 0 or start_row + length_rows > n_rows:
        raise ValueError("outage block is outside the row range")
    if feature_indices is None:
        cols = np.arange(n_features, dtype=int)
    else:
        cols = np.asarray(list(feature_indices), dtype=int)
        if (
            cols.ndim != 1
            or len(cols) == 0
            or np.any(cols < 0)
            or np.any(cols >= n_features)
            or len(np.unique(cols)) != len(cols)
        ):
            raise ValueError(
                "feature_indices must be unique in-range feature indices"
            )
    mask = np.zeros_like(eligible)
    block = slice(start_row, start_row + length_rows)
    mask[block, cols] = eligible[block][:, cols]
    if not mask.any():
        raise ValueError("requested outage contains no eligible real observations")
    return mask


def _experimental_observed(
    observed: np.ndarray,
    structural: np.ndarray,
    holdout: np.ndarray,
) -> np.ndarray:
    if holdout.shape != observed.shape:
        raise ValueError("holdout mask shape mismatch")
    if np.any(holdout & ~observed) or np.any(holdout & structural):
        raise ValueError("holdout may contain only real, nonstructural observations")
    return observed & ~holdout


def recover_train_median(
    values,
    observed_mask,
    structural_unavailable_mask,
    holdout_mask,
    *,
    fit_rows,
) -> RecoveryOutput:
    """Train-fit median recovery. Structural gaps remain unavailable."""
    array = np.asarray(values, dtype=np.float64)
    observed, structural = _masks(observed_mask, structural_unavailable_mask)
    holdout = np.asarray(holdout_mask, dtype=bool)
    if array.shape != observed.shape:
        raise ValueError("values and masks must have identical shape")
    fit = np.asarray(fit_rows, dtype=bool)
    if fit.shape != (array.shape[0],) or not fit.any():
        raise ValueError("fit_rows must select at least one row")
    experimental = _experimental_observed(observed, structural, holdout)
    medians = fit_train_medians(array[fit], experimental[fit])
    filled = apply_train_median_fill(array, experimental, medians)
    # Structural cells are never promoted to available even though the simple
    # filler mathematically produced a number for them.
    available = experimental | holdout
    available[structural] = False
    filled[structural] = np.nan
    unresolved = structural.copy()
    return RecoveryOutput(
        "TRAIN_FIT_MEDIAN",
        filled,
        available,
        holdout.copy(),
        unresolved,
    )


def recover_causal_forward_fill(
    values,
    observed_mask,
    structural_unavailable_mask,
    holdout_mask,
) -> RecoveryOutput:
    """Causal forward-fill recovery; leading and structural gaps stay unresolved."""
    array = np.asarray(values, dtype=np.float64)
    observed, structural = _masks(observed_mask, structural_unavailable_mask)
    holdout = np.asarray(holdout_mask, dtype=bool)
    if array.shape != observed.shape:
        raise ValueError("values and masks must have identical shape")
    experimental = _experimental_observed(observed, structural, holdout)
    filled, unresolved = causal_forward_fill(array, experimental)
    # Never carry through an era where a quantity is authoritatively unavailable.
    filled[structural] = np.nan
    unresolved |= structural
    reconstructed = holdout & ~unresolved
    available = experimental | reconstructed
    return RecoveryOutput(
        "CAUSAL_FORWARD_FILL",
        filled,
        available,
        reconstructed,
        unresolved,
    )


def _binary(values) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a non-empty vector")
    try:
        y = y.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary") from exc
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("labels must be binary")
    return y.astype(np.int8)


def _prob(values, *, allow_nan: bool = False) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a non-empty vector")
    finite = np.isfinite(p)
    if not allow_nan and not finite.all():
        raise ValueError("probabilities must be finite")
    if np.any(finite & ((p < 0) | (p > 1))):
        raise ValueError("finite probabilities must be in [0,1]")
    if allow_nan and np.any(~finite & ~np.isnan(p)):
        raise ValueError("abstained probabilities must use NaN, not infinity")
    return p


def _ratio(numerator: float, denominator: float):
    return None if denominator == 0 else float(numerator / denominator)


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = _ratio(tp, tp + fn)
    fpr = _ratio(fp, fp + tn)
    far = _ratio(fp, tp + fp)
    tss = None if pod is None or fpr is None else float(pod - fpr)
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


def _probability_metrics(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> dict:
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        member = (p >= edges[index]) & (
            p < edges[index + 1]
            if index < bins - 1
            else p <= edges[index + 1]
        )
        if member.any():
            ece += float(
                np.mean(member)
                * abs(np.mean(p[member]) - np.mean(y[member]))
            )
    return {"BRIER": brier, "ECE": float(ece)}


def _minimum_far_at_pod(y: np.ndarray, p: np.ndarray, target_pod: float) -> dict:
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    eligible = []
    for threshold in candidates:
        metrics = _threshold_metrics(y, p, float(threshold))
        if (
            metrics["POD"] is not None
            and metrics["POD"] >= target_pod
            and metrics["FAR"] is not None
        ):
            eligible.append(
                (metrics["FAR"], -float(threshold), metrics["POD"])
            )
    if not eligible:
        return {
            "target_POD": target_pod,
            "FAR": None,
            "threshold": None,
            "achieved_POD": None,
        }
    far, negative_threshold, achieved = min(
        eligible,
        key=lambda row: (row[0], row[1]),
    )
    return {
        "target_POD": target_pod,
        "FAR": float(far),
        "threshold": float(-negative_threshold),
        "achieved_POD": float(achieved),
    }


def _delta(candidate, reference):
    return (
        None
        if candidate is None or reference is None
        else float(candidate - reference)
    )


def evaluate_forecast_preservation(
    *,
    labels,
    reference_probabilities,
    candidate_probabilities,
    reference_threshold: float,
    candidate_threshold: float,
    role: str,
    matched_pod: float = 0.8,
) -> dict:
    """Compare a recovery arm with complete-data forecasts on identical retained rows.

    ``candidate_probabilities`` may contain NaN to mean ABSTAIN. The reference
    is rescored only on those same retained identities, so an arm cannot improve
    its apparent skill merely by dropping difficult rows without paying a
    coverage penalty.
    """
    if role != _SCORE_ROLE:
        raise ValueError(
            "forecast-preservation scoring is restricted to train_only_inner_score"
        )
    y = _binary(labels)
    reference = _prob(reference_probabilities)
    candidate = _prob(candidate_probabilities, allow_nan=True)
    if not (y.shape == reference.shape == candidate.shape):
        raise ValueError("labels and probabilities must align")
    for name, threshold in (
        ("reference_threshold", reference_threshold),
        ("candidate_threshold", candidate_threshold),
    ):
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(f"{name} must be finite in [0,1]")
    if not math.isfinite(matched_pod) or not 0 < matched_pod <= 1:
        raise ValueError("matched_pod must be in (0,1]")

    retained = np.isfinite(candidate)
    if retained.sum() < 2 or len(np.unique(y[retained])) != 2:
        raise ValueError("retained score rows must contain both classes")
    retained_y = y[retained]
    retained_reference = reference[retained]
    retained_candidate = candidate[retained]
    reference_metrics = {
        **_threshold_metrics(
            retained_y,
            retained_reference,
            float(reference_threshold),
        ),
        **_probability_metrics(retained_y, retained_reference),
    }
    candidate_metrics = {
        **_threshold_metrics(
            retained_y,
            retained_candidate,
            float(candidate_threshold),
        ),
        **_probability_metrics(retained_y, retained_candidate),
    }
    reference_matched = _minimum_far_at_pod(
        retained_y,
        retained_reference,
        matched_pod,
    )
    candidate_matched = _minimum_far_at_pod(
        retained_y,
        retained_candidate,
        matched_pod,
    )
    coverage = float(np.mean(retained))
    return {
        "scope": _SCORE_ROLE,
        "score_rows": int(len(y)),
        "retained_rows": int(retained.sum()),
        "coverage": coverage,
        "abstention_rate": float(1.0 - coverage),
        "reference_on_identical_retained_rows": reference_metrics,
        "candidate_on_identical_retained_rows": candidate_metrics,
        "delta_candidate_minus_reference": {
            "TSS": _delta(
                candidate_metrics["TSS"],
                reference_metrics["TSS"],
            ),
            "FAR": _delta(
                candidate_metrics["FAR"],
                reference_metrics["FAR"],
            ),
            "BRIER": _delta(
                candidate_metrics["BRIER"],
                reference_metrics["BRIER"],
            ),
            "ECE": _delta(
                candidate_metrics["ECE"],
                reference_metrics["ECE"],
            ),
        },
        "matched_detection": {
            "reference": reference_matched,
            "candidate": candidate_matched,
            "delta_FAR": _delta(
                candidate_matched["FAR"],
                reference_matched["FAR"],
            ),
        },
        "claim_boundary": (
            "Train-only hidden-data diagnostic; no locked-test, superiority, "
            "operational, or award claim."
        ),
    }
