from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SharedBootstrapDraws:
    positive_units: tuple[str, ...]
    negative_units: tuple[str, ...]
    positive_indices: np.ndarray
    negative_indices: np.ndarray

    def validate(self) -> None:
        if self.positive_indices.ndim != 2 or self.negative_indices.ndim != 2:
            raise ValueError("bootstrap index tensors must be two-dimensional")
        if self.positive_indices.shape[0] != self.negative_indices.shape[0]:
            raise ValueError("positive/negative replicate counts differ")
        if self.positive_indices.shape[1] != len(self.positive_units):
            raise ValueError("positive draw width must equal positive unit count")
        if self.negative_indices.shape[1] != len(self.negative_units):
            raise ValueError("negative draw width must equal negative unit count")
        if len(self.positive_units) and np.any((self.positive_indices < 0) | (self.positive_indices >= len(self.positive_units))):
            raise ValueError("positive bootstrap index out of range")
        if len(self.negative_units) and np.any((self.negative_indices < 0) | (self.negative_indices >= len(self.negative_units))):
            raise ValueError("negative bootstrap index out of range")


def build_shared_stratified_bootstrap_draws(
    positive_unit_ids: Sequence[str],
    negative_unit_ids: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 20260909,
) -> SharedBootstrapDraws:
    pos = tuple(sorted(set(map(str, positive_unit_ids))))
    neg = tuple(sorted(set(map(str, negative_unit_ids))))
    if not pos:
        raise ValueError("at least one positive physical episode is required")
    if not neg:
        raise ValueError("at least one negative quiet block is required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    pos_idx = rng.integers(0, len(pos), size=(replicates, len(pos)), endpoint=False, dtype=np.int32)
    neg_idx = rng.integers(0, len(neg), size=(replicates, len(neg)), endpoint=False, dtype=np.int32)
    out = SharedBootstrapDraws(pos, neg, pos_idx, neg_idx)
    out.validate()
    return out


def draw_multiplicity_matrix(indices: np.ndarray, unit_count: int) -> np.ndarray:
    if indices.ndim != 2:
        raise ValueError("indices must be 2D")
    if unit_count <= 0:
        raise ValueError("unit_count must be positive")
    result = np.zeros((indices.shape[0], unit_count), dtype=np.int16)
    for r in range(indices.shape[0]):
        result[r] = np.bincount(indices[r], minlength=unit_count)
    return result


def weighted_confusion(
    labels: Sequence[int],
    alerts: Sequence[int],
    weights: Sequence[float] | None = None,
) -> Mapping[str, float]:
    y = np.asarray(labels, dtype=int)
    a = np.asarray(alerts, dtype=int)
    if y.shape != a.shape:
        raise ValueError("labels/alerts shape mismatch")
    if not np.isin(y, [0, 1]).all() or not np.isin(a, [0, 1]).all():
        raise ValueError("labels and alerts must be binary")
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("invalid weights")
    return {
        "tp": float(w[(y == 1) & (a == 1)].sum()),
        "fn": float(w[(y == 1) & (a == 0)].sum()),
        "fp": float(w[(y == 0) & (a == 1)].sum()),
        "tn": float(w[(y == 0) & (a == 0)].sum()),
    }


def metrics_from_confusion(conf: Mapping[str, float]) -> Mapping[str, float]:
    tp, fn, fp, tn = (float(conf[k]) for k in ("tp", "fn", "fp", "tn"))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr if np.isfinite(pod) and np.isfinite(fpr) else float("nan")
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fn * fp) / denom if denom else float("nan")
    return {**dict(conf), "pod": pod, "fpr": fpr, "far": far, "tss": tss, "hss": hss}


def weighted_binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    weights: Sequence[float] | None = None,
) -> Mapping[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape:
        raise ValueError("labels/probabilities shape mismatch")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite in [0,1]")
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    conf = weighted_confusion(y, (p >= threshold).astype(int), w)
    base = metrics_from_confusion(conf)
    wsum = float(w.sum())
    brier = float(np.sum(w * (p - y) ** 2) / wsum) if wsum else float("nan")
    return {**base, "brier": brier, "weight_sum": wsum}


def unit_confusion_contributions(
    unit_ids: Sequence[str],
    labels: Sequence[int],
    alerts: Sequence[int],
    weights: Sequence[float],
    unit_order: Sequence[str],
) -> np.ndarray:
    if not (len(unit_ids) == len(labels) == len(alerts) == len(weights)):
        raise ValueError("row arrays have inconsistent lengths")
    pos = {str(u): i for i, u in enumerate(unit_order)}
    out = np.zeros((len(unit_order), 4), dtype=float)  # tp, fn, fp, tn
    for unit, y, alert, weight in zip(unit_ids, labels, alerts, weights, strict=True):
        unit = str(unit)
        if unit not in pos:
            continue
        idx = pos[unit]
        y = int(y); alert = int(alert); weight = float(weight)
        if y == 1 and alert == 1:
            out[idx, 0] += weight
        elif y == 1:
            out[idx, 1] += weight
        elif alert == 1:
            out[idx, 2] += weight
        else:
            out[idx, 3] += weight
    return out


def bootstrap_metric_arrays(
    draws: SharedBootstrapDraws,
    positive_contributions: np.ndarray,
    negative_contributions: np.ndarray,
) -> Mapping[str, np.ndarray]:
    draws.validate()
    if positive_contributions.shape != (len(draws.positive_units), 4):
        raise ValueError("positive contribution shape mismatch")
    if negative_contributions.shape != (len(draws.negative_units), 4):
        raise ValueError("negative contribution shape mismatch")
    pos_mult = draw_multiplicity_matrix(draws.positive_indices, len(draws.positive_units)).astype(float)
    neg_mult = draw_multiplicity_matrix(draws.negative_indices, len(draws.negative_units)).astype(float)
    counts = pos_mult @ positive_contributions + neg_mult @ negative_contributions
    tp, fn, fp, tn = counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = tp / (tp + fn)
        fpr = fp / (fp + tn)
        far = fp / (tp + fp)
        tss = pod - fpr
        denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        hss = 2.0 * (tp * tn - fn * fp) / denom
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "pod": pod, "fpr": fpr, "far": far, "tss": tss, "hss": hss}


def percentile_interval(values: Sequence[float], *, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))
