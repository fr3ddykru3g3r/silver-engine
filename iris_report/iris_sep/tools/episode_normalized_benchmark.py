from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


TERMINAL_CODES = {
    "ELIGIBLE_ONSET_POSITIVE",
    "ELIGIBLE_ONSET_NEGATIVE",
    "ALREADY_ACTIVE_PERSISTENCE",
    "UNRESOLVED_GAP",
    "IMMATURE_OUTCOME_WINDOW",
    "DUPLICATE_OR_AMBIGUOUS",
    "SOURCE_UNAVAILABLE",
    "EXCLUDED_BY_FROZEN_BOUNDARY",
}


@dataclass(frozen=True)
class EvaluationRow:
    issue_id: str
    eligibility_code: str
    label: int | None
    probability: float | None
    episode_id: str | None = None
    quiet_block_id: str | None = None

    def validate(self) -> None:
        if self.eligibility_code not in TERMINAL_CODES:
            raise ValueError(f"unknown eligibility code: {self.eligibility_code}")
        if self.label not in (None, 0, 1):
            raise ValueError("label must be None, 0 or 1")
        if self.probability is not None and not (0.0 <= float(self.probability) <= 1.0):
            raise ValueError("probability must be in [0, 1]")
        if self.eligibility_code == "ELIGIBLE_ONSET_POSITIVE":
            if self.label != 1:
                raise ValueError("onset-positive row must have label=1")
            if not self.episode_id:
                raise ValueError("onset-positive row requires episode_id")
        if self.eligibility_code == "ELIGIBLE_ONSET_NEGATIVE" and self.label != 0:
            raise ValueError("onset-negative row must have label=0")
        if self.eligibility_code == "ALREADY_ACTIVE_PERSISTENCE" and self.label == 1:
            # A persistence case can be positive for occurrence, but it is not an onset-positive.
            pass


def episode_normalized_weights(rows: Sequence[EvaluationRow]) -> np.ndarray:
    """Return weights for NEW_ONSET_CAUSAL rows.

    Positive rows belonging to the same physical episode share total weight 1.
    Eligible onset negatives retain unit weight. Non-onset rows receive weight 0.
    """

    for row in rows:
        row.validate()

    episode_counts: dict[str, int] = {}
    for row in rows:
        if row.eligibility_code == "ELIGIBLE_ONSET_POSITIVE":
            assert row.episode_id is not None
            episode_counts[row.episode_id] = episode_counts.get(row.episode_id, 0) + 1

    weights = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        if row.eligibility_code == "ELIGIBLE_ONSET_POSITIVE":
            assert row.episode_id is not None
            weights[idx] = 1.0 / episode_counts[row.episode_id]
        elif row.eligibility_code == "ELIGIBLE_ONSET_NEGATIVE":
            weights[idx] = 1.0
    return weights


def episode_multiplicity_factor(rows: Sequence[EvaluationRow]) -> float:
    positives = [r for r in rows if r.eligibility_code == "ELIGIBLE_ONSET_POSITIVE"]
    episode_ids = {r.episode_id for r in positives}
    if not positives:
        return float("nan")
    if None in episode_ids:
        raise ValueError("positive onset rows require episode ids")
    return len(positives) / len(episode_ids)


def _weighted_confusion(
    labels: np.ndarray,
    alerts: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    tp = float(weights[(labels == 1) & (alerts == 1)].sum())
    fn = float(weights[(labels == 1) & (alerts == 0)].sum())
    fp = float(weights[(labels == 0) & (alerts == 1)].sum())
    tn = float(weights[(labels == 0) & (alerts == 0)].sum())
    return tp, fn, fp, tn


def score_onset_rows(
    rows: Sequence[EvaluationRow],
    threshold: float,
    *,
    episode_normalized: bool,
) -> Mapping[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    eligible = [
        row
        for row in rows
        if row.eligibility_code in {"ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"}
    ]
    if not eligible:
        raise ValueError("no onset-eligible rows")
    if any(row.probability is None for row in eligible):
        raise ValueError("eligible rows require probabilities")

    labels = np.asarray([int(row.label) for row in eligible], dtype=int)
    probs = np.asarray([float(row.probability) for row in eligible], dtype=float)
    alerts = (probs >= threshold).astype(int)
    if episode_normalized:
        weights = episode_normalized_weights(eligible)
    else:
        weights = np.ones(len(eligible), dtype=float)

    tp, fn, fp, tn = _weighted_confusion(labels, alerts, weights)
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    tss = pod - fpr if np.isfinite(pod) and np.isfinite(fpr) else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pod": pod,
        "fpr": fpr,
        "tss": tss,
        "far": far,
        "positive_weight": tp + fn,
        "negative_weight": fp + tn,
    }


def build_shared_bootstrap_draws(
    unit_ids: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 20260909,
) -> np.ndarray:
    """Create one immutable draw tensor to reuse for every paired comparison."""

    units = np.asarray(list(dict.fromkeys(unit_ids)), dtype=object)
    if units.size == 0:
        raise ValueError("at least one bootstrap unit is required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, units.size, size=(replicates, units.size), endpoint=False)
    return units[indices]


def bootstrap_unit_for_row(row: EvaluationRow) -> str:
    if row.eligibility_code == "ELIGIBLE_ONSET_POSITIVE":
        if not row.episode_id:
            raise ValueError("positive onset row requires episode_id")
        return f"EPISODE::{row.episode_id}"
    if row.eligibility_code == "ELIGIBLE_ONSET_NEGATIVE":
        if not row.quiet_block_id:
            raise ValueError("negative onset row requires predeclared quiet_block_id")
        return f"QUIET::{row.quiet_block_id}"
    raise ValueError("row is not onset eligible")


def rank_models(metric_by_model: Mapping[str, float]) -> list[str]:
    if not metric_by_model:
        raise ValueError("no models supplied")
    for name, value in metric_by_model.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite metric for {name}")
    return sorted(metric_by_model, key=lambda name: (-metric_by_model[name], name))


def rank_stability(
    standard_metric_by_model: Mapping[str, float],
    normalized_metric_by_model: Mapping[str, float],
) -> Mapping[str, object]:
    if set(standard_metric_by_model) != set(normalized_metric_by_model):
        raise ValueError("model sets differ across evaluation views")
    standard_rank = rank_models(standard_metric_by_model)
    normalized_rank = rank_models(normalized_metric_by_model)

    names = sorted(standard_metric_by_model)
    reversals: list[tuple[str, str]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            standard_delta = standard_metric_by_model[left] - standard_metric_by_model[right]
            normalized_delta = normalized_metric_by_model[left] - normalized_metric_by_model[right]
            if standard_delta * normalized_delta < 0:
                reversals.append((left, right))

    return {
        "standard_rank": standard_rank,
        "normalized_rank": normalized_rank,
        "rank_changed": standard_rank != normalized_rank,
        "pairwise_reversals": reversals,
    }
