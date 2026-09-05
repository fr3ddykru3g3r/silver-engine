"""Dependency-light evaluation primitives with strict role boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


class EvaluationError(ValueError):
    pass


def _binary(labels: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(labels), dtype=int)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isin(values, [0, 1])):
        raise EvaluationError("labels must be a non-empty binary vector")
    return values


def _probabilities(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise EvaluationError("probabilities must be a non-empty finite vector in [0,1]")
    return result


def _logits(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise EvaluationError("logits must be a non-empty finite vector")
    return result


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    result = np.empty_like(logits)
    nonnegative = logits >= 0
    result[nonnegative] = 1 / (1 + np.exp(-logits[nonnegative]))
    exponential = np.exp(logits[~nonnegative])
    result[~nonnegative] = exponential / (1 + exponential)
    return result


@dataclass(frozen=True)
class InterceptCalibration:
    intercept: float
    fit_role: str
    calibration_id: str


def fit_intercept_calibration(logits: Iterable[float], labels: Iterable[int], *, role: str) -> InterceptCalibration:
    if role != "validation_calibration":
        raise EvaluationError("calibration may be fit only on validation_calibration")
    z, y = _logits(logits), _binary(labels)
    if z.shape != y.shape or len(np.unique(y)) != 2:
        raise EvaluationError("calibration requires aligned logits and both classes")
    intercept = 0.0
    for _ in range(100):
        p = sigmoid(z + intercept)
        gradient = float(np.sum(p - y))
        hessian = float(np.sum(p * (1 - p)))
        if hessian <= 1e-12:
            raise EvaluationError("calibration is numerically degenerate")
        update = gradient / hessian
        intercept -= update
        if abs(update) < 1e-10:
            break
    core = {"method": "LOGIT_INTERCEPT_ONLY", "fit_role": role, "intercept": round(intercept, 15)}
    calibration_id = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return InterceptCalibration(intercept, role, calibration_id)


def apply_calibration(logits: Iterable[float], calibration: InterceptCalibration) -> np.ndarray:
    if calibration.fit_role != "validation_calibration":
        raise EvaluationError("invalid calibration receipt role")
    return sigmoid(_logits(logits) + calibration.intercept)


def confusion(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> tuple[int, int, int, int]:
    y, p = _binary(labels), _probabilities(probabilities)
    if y.shape != p.shape or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise EvaluationError("labels/probabilities/threshold are invalid")
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred)); tn = int(np.sum((y == 0) & ~pred))
    return tp, fn, fp, tn


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def threshold_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float]:
    tp, fn, fp, tn = confusion(labels, probabilities, threshold)
    pod = _ratio(tp, tp + fn); fpr = _ratio(fp, fp + tn); far = _ratio(fp, tp + fp)
    tss = pod - fpr
    hss = _ratio(2 * (tp * tn - fp * fn), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "POD": pod, "FPR": fpr, "FAR": far, "TSS": tss, "HSS": hss}


@dataclass(frozen=True)
class ThresholdReceipt:
    threshold: float
    fit_role: str
    objective: str
    threshold_id: str


def select_tss_threshold(labels: Iterable[int], probabilities: Iterable[float], *, role: str) -> ThresholdReceipt:
    if role != "validation_threshold":
        raise EvaluationError("threshold may be selected only on validation_threshold")
    y, p = _binary(labels), _probabilities(probabilities)
    if y.shape != p.shape or len(np.unique(y)) != 2:
        raise EvaluationError("threshold selection requires aligned probabilities and both classes")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    scored = [(float(threshold_metrics(y, p, float(t))["TSS"]), float(t)) for t in candidates]
    best_score = max(score for score, _ in scored)
    threshold = min(t for score, t in scored if np.isclose(score, best_score, rtol=0, atol=1e-12))
    core = {"objective": "MAXIMIZE_TSS", "fit_role": role, "tie_break": "LOWEST_THRESHOLD", "threshold": threshold}
    threshold_id = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ThresholdReceipt(threshold, role, "MAXIMIZE_TSS", threshold_id)


def probability_metrics(labels: Iterable[int], probabilities: Iterable[float], *, reference_probability: float, bins: int = 10) -> dict[str, float]:
    y, p = _binary(labels), _probabilities(probabilities)
    if y.shape != p.shape or len(np.unique(y)) != 2 or not 0 <= reference_probability <= 1 or bins <= 0:
        raise EvaluationError("probability metric inputs are invalid")
    brier = float(np.mean((p - y) ** 2))
    reference_brier = float(np.mean((reference_probability - y) ** 2))
    bss = 1 - brier / reference_brier if reference_brier > 0 else float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        member = (p >= edges[index]) & (p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1])
        if np.any(member):
            ece += float(np.mean(member) * abs(np.mean(p[member]) - np.mean(y[member])))
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y); fp = np.cumsum(1 - sorted_y)
    distinct = np.r_[np.where(np.diff(p[order]))[0], len(p) - 1]
    grouped_precision = tp[distinct] / (tp[distinct] + fp[distinct])
    grouped_recall = tp[distinct] / int(np.sum(y))
    auprc = float(
        np.sum(
            (grouped_recall - np.concatenate(([0.0], grouped_recall[:-1])))
            * grouped_precision
        )
    )
    roc_tpr = np.r_[0.0, tp[distinct] / int(np.sum(y)), 1.0]
    roc_fpr = np.r_[0.0, fp[distinct] / int(np.sum(1 - y)), 1.0]
    auroc = float(np.trapezoid(roc_tpr, roc_fpr))
    return {"BRIER": brier, "BRIER_SKILL": bss, "ECE": ece, "AUPRC": auprc, "AUROC": auroc}


def minimum_far_at_pod(labels: Iterable[int], probabilities: Iterable[float], target_pod: float) -> dict[str, float]:
    y, p = _binary(labels), _probabilities(probabilities)
    if y.shape != p.shape or not 0 < target_pod <= 1:
        raise EvaluationError("matched-POD inputs are invalid")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    eligible = []
    for threshold in candidates:
        metrics = threshold_metrics(y, p, float(threshold))
        if metrics["POD"] >= target_pod and np.isfinite(metrics["FAR"]):
            eligible.append((metrics["FAR"], -threshold, metrics))
    if not eligible:
        raise EvaluationError("target POD is unattainable")
    far, negative_threshold, metrics = min(eligible, key=lambda row: (row[0], row[1]))
    return {"target_POD": target_pod, "achieved_POD": metrics["POD"], "FAR": far, "threshold": -negative_threshold}


def paired_unit_bootstrap_tss_difference(
    labels: Iterable[int], iris_probabilities: Iterable[float], comparator_probabilities: Iterable[float],
    unit_ids: Iterable[str], *, iris_threshold: float, comparator_threshold: float,
    replicates: int = 10000, seed: int = 20260904,
) -> dict[str, float | int]:
    y = _binary(labels); iris = _probabilities(iris_probabilities); comparator = _probabilities(comparator_probabilities)
    units = np.asarray(list(unit_ids), dtype=str)
    if not (y.shape == iris.shape == comparator.shape == units.shape) or replicates <= 0:
        raise EvaluationError("paired bootstrap inputs are invalid")
    unique_units = np.unique(units)
    if len(unique_units) < 2:
        raise EvaluationError("paired bootstrap requires at least two units")
    rows = {unit: np.flatnonzero(units == unit) for unit in unique_units}
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique_units, size=len(unique_units), replace=True)
        indices = np.concatenate([rows[unit] for unit in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        iris_tss = threshold_metrics(y[indices], iris[indices], iris_threshold)["TSS"]
        comparator_tss = threshold_metrics(y[indices], comparator[indices], comparator_threshold)["TSS"]
        differences.append(float(iris_tss - comparator_tss))
    if len(differences) < int(np.ceil(replicates * 0.95)):
        raise EvaluationError("fewer than 95% valid paired bootstrap replicates")
    values = np.asarray(differences)
    return {
        "valid_replicates": len(values), "median_difference": float(np.median(values)),
        "ci_lower_95": float(np.quantile(values, 0.025)), "ci_upper_95": float(np.quantile(values, 0.975)),
    }
