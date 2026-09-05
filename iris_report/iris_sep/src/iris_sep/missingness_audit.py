"""Train-only missingness audit utilities for IRIS-SEP.

The audit is deliberately descriptive. It is designed to freeze the missing-data
problem before choosing an imputer or physics model. It must not access locked
identities/outcomes and must not turn row-level support into claims of
independent-event significance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureMissingnessSummary:
    feature: str
    observed_count: int
    missing_count: int
    missing_fraction: float
    positive_observed_count: int
    positive_missing_count: int
    negative_observed_count: int
    negative_missing_count: int
    positive_missing_fraction: float | None
    negative_missing_fraction: float | None
    missing_fraction_difference_positive_minus_negative: float | None
    longest_missing_run_rows: int
    longest_missing_run_minutes: float


def _validate_inputs(
    issue_times: Sequence[datetime],
    labels,
    feature_names: Sequence[str],
    observed_mask,
    eras: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    observed = np.asarray(observed_mask, dtype=bool)
    n = len(issue_times)
    if n == 0:
        raise ValueError("at least one train-only issue is required")
    if labels_array.shape != (n,):
        raise ValueError("labels must contain one value per issue")
    if observed.shape != (n, len(feature_names)):
        raise ValueError("observed_mask must be [issues, features]")
    if len(eras) != n:
        raise ValueError("eras must contain one value per issue")
    if len(set(feature_names)) != len(feature_names) or any(not name for name in feature_names):
        raise ValueError("feature_names must be unique non-empty strings")
    if any(not isinstance(t, datetime) or t.tzinfo is None for t in issue_times):
        raise ValueError("issue_times must be timezone-aware datetimes")
    if any(issue_times[i] >= issue_times[i + 1] for i in range(n - 1)):
        raise ValueError("issue_times must be strictly increasing")
    if any(not isinstance(era, str) or not era for era in eras):
        raise ValueError("eras must be non-empty strings")

    # bool labels are accepted, as are exact numeric 0/1. Anything else is a
    # contract error because this audit is tied to a binary train-only target.
    if labels_array.dtype == bool:
        binary = labels_array.astype(np.int8)
    else:
        try:
            numeric = labels_array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must be binary 0/1") from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError("labels must be binary 0/1")
        binary = numeric.astype(np.int8)
    return binary, observed


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _longest_missing_run(
    issue_times: Sequence[datetime],
    missing: np.ndarray,
    *,
    expected_cadence_minutes: float,
    continuity_tolerance: float,
) -> tuple[int, float]:
    """Return longest contiguous missing run in rows and elapsed minutes.

    A run breaks when the issue-time spacing exceeds the declared cadence times
    the tolerance. Duration is measured from the start issue through one expected
    cadence after the last missing issue, so a single missing daily issue has a
    one-day duration rather than zero.
    """
    if not math.isfinite(expected_cadence_minutes) or expected_cadence_minutes <= 0:
        raise ValueError("expected_cadence_minutes must be positive and finite")
    if not math.isfinite(continuity_tolerance) or continuity_tolerance < 1:
        raise ValueError("continuity_tolerance must be finite and >= 1")

    best_rows = 0
    best_minutes = 0.0
    start: int | None = None
    previous: int | None = None
    limit = expected_cadence_minutes * continuity_tolerance

    def close_run(end: int) -> None:
        nonlocal best_rows, best_minutes, start
        if start is None:
            return
        rows = end - start + 1
        elapsed = (issue_times[end] - issue_times[start]).total_seconds() / 60.0
        duration = elapsed + expected_cadence_minutes
        if rows > best_rows or (rows == best_rows and duration > best_minutes):
            best_rows = rows
            best_minutes = float(duration)
        start = None

    for idx, is_missing in enumerate(missing.tolist()):
        if not is_missing:
            if start is not None and previous is not None:
                close_run(previous)
            previous = None
            continue
        if start is None:
            start = idx
        elif previous is not None:
            delta = (issue_times[idx] - issue_times[previous]).total_seconds() / 60.0
            if delta > limit:
                close_run(previous)
                start = idx
        previous = idx
    if start is not None and previous is not None:
        close_run(previous)
    return best_rows, best_minutes


def summarize_missingness(
    *,
    issue_times: Sequence[datetime],
    labels,
    feature_names: Sequence[str],
    observed_mask,
    eras: Sequence[str],
    expected_cadence_minutes: float,
    continuity_tolerance: float = 1.5,
) -> dict:
    """Build a deterministic descriptive train-only missingness manifest.

    The manifest contains global feature support, source-era and calendar-quarter
    support, plus row-level positive/negative missingness differences. Those
    differences are diagnostics only; no independence or p-value claim is made.
    """
    binary, observed = _validate_inputs(issue_times, labels, feature_names, observed_mask, eras)
    n = len(issue_times)
    positives = binary == 1
    negatives = ~positives

    features: list[dict] = []
    for col, name in enumerate(feature_names):
        seen = observed[:, col]
        missing = ~seen
        positive_missing = int((missing & positives).sum())
        negative_missing = int((missing & negatives).sum())
        positive_total = int(positives.sum())
        negative_total = int(negatives.sum())
        pos_fraction = _fraction(positive_missing, positive_total)
        neg_fraction = _fraction(negative_missing, negative_total)
        difference = None if pos_fraction is None or neg_fraction is None else float(pos_fraction - neg_fraction)
        run_rows, run_minutes = _longest_missing_run(
            issue_times, missing,
            expected_cadence_minutes=expected_cadence_minutes,
            continuity_tolerance=continuity_tolerance,
        )
        summary = FeatureMissingnessSummary(
            feature=name,
            observed_count=int(seen.sum()),
            missing_count=int(missing.sum()),
            missing_fraction=float(missing.mean()),
            positive_observed_count=int((seen & positives).sum()),
            positive_missing_count=positive_missing,
            negative_observed_count=int((seen & negatives).sum()),
            negative_missing_count=negative_missing,
            positive_missing_fraction=pos_fraction,
            negative_missing_fraction=neg_fraction,
            missing_fraction_difference_positive_minus_negative=difference,
            longest_missing_run_rows=run_rows,
            longest_missing_run_minutes=run_minutes,
        )
        features.append(summary.__dict__)

    era_rows: list[dict] = []
    for era in sorted(set(eras)):
        row_mask = np.asarray([value == era for value in eras], dtype=bool)
        for col, name in enumerate(feature_names):
            era_observed = observed[row_mask, col]
            era_rows.append({
                "era": era,
                "feature": name,
                "issue_count": int(row_mask.sum()),
                "positive_issue_count": int((row_mask & positives).sum()),
                "observed_count": int(era_observed.sum()),
                "missing_count": int((~era_observed).sum()),
                "missing_fraction": float((~era_observed).mean()),
            })

    quarter_rows: list[dict] = []
    quarters = [f"{t.year}-Q{((t.month - 1) // 3) + 1}" for t in issue_times]
    for quarter in sorted(set(quarters)):
        row_mask = np.asarray([value == quarter for value in quarters], dtype=bool)
        for col, name in enumerate(feature_names):
            q_observed = observed[row_mask, col]
            quarter_rows.append({
                "quarter": quarter,
                "feature": name,
                "issue_count": int(row_mask.sum()),
                "positive_issue_count": int((row_mask & positives).sum()),
                "observed_count": int(q_observed.sum()),
                "missing_count": int((~q_observed).sum()),
                "missing_fraction": float((~q_observed).mean()),
            })

    all_missing_rows = ~observed.any(axis=1)
    complete_rows = observed.all(axis=1)
    return {
        "scope": "TRAIN_ONLY_DESCRIPTIVE_MISSINGNESS_AUDIT",
        "issue_count": n,
        "positive_issue_count": int(positives.sum()),
        "negative_issue_count": int(negatives.sum()),
        "feature_count": len(feature_names),
        "expected_cadence_minutes": float(expected_cadence_minutes),
        "continuity_tolerance": float(continuity_tolerance),
        "complete_issue_count": int(complete_rows.sum()),
        "all_features_missing_issue_count": int(all_missing_rows.sum()),
        "features": features,
        "by_era": era_rows,
        "by_quarter": quarter_rows,
        "claim_boundary": "Descriptive row-level support only; no event-independence, causality, significance, final NEW-crossing skill or operational claim."
    }
