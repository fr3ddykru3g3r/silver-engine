"""Run the preregistered IRIS Forecast Integrity Gate V1 evaluation.

Input is a long-format prediction table. Selector cutoffs are learned without
labels on the threshold role and then frozen before application to the score role.

Required columns:
  issue_time, role, unit_id, scenario, model, label, probability,
  evidence_integrity, availability_score, decision_threshold

The tool never trains or changes a forecaster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.forecast_integrity import forecast_confidence, shared_cluster_bootstrap_draws

SELECTORS = ("EVIDENCE_INTEGRITY", "FORECAST_CONFIDENCE", "AVAILABILITY_ONLY", "SEEDED_RANDOM")
METRICS_FOR_CONTRAST = ("TSS", "FAR", "POD", "BRIER")


def _stable_random_score(row: pd.Series, seed: int) -> float:
    key = f"{seed}|{row['issue_time']}|{row['unit_id']}|{row['scenario']}|{row['model']}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def _selector_scores(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    p = frame["probability"].to_numpy(dtype=float)
    return {
        "EVIDENCE_INTEGRITY": frame["evidence_integrity"].to_numpy(dtype=float),
        "FORECAST_CONFIDENCE": np.asarray([forecast_confidence(x) for x in p], dtype=float),
        "AVAILABILITY_ONLY": frame["availability_score"].to_numpy(dtype=float),
        "SEEDED_RANDOM": np.asarray([_stable_random_score(row, seed) for _, row in frame.iterrows()], dtype=float),
    }


def _cutoff_for_coverage(scores: np.ndarray, coverage: float) -> float:
    if len(scores) == 0:
        raise ValueError("threshold role is empty")
    keep = max(1, int(np.ceil(len(scores) * coverage)))
    ordered = np.sort(np.asarray(scores, dtype=float))[::-1]
    return float(ordered[keep - 1])


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"rows": 0, "positives": 0, "TP": 0, "FN": 0, "FP": 0, "TN": 0,
                "POD": None, "FPR": None, "FAR": None, "TSS": None, "BRIER": None}
    pred = p >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    far = fp / (tp + fp) if tp + fp else None
    tss = (pod - fpr) if pod is not None and fpr is not None else None
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y)),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "POD": pod, "FPR": fpr, "FAR": far, "TSS": tss,
        "BRIER": float(np.mean((p - y) ** 2)),
    }


def _metric_delta(a, b):
    if a is None or b is None:
        return None
    return float(a - b)


def _percentile_interval(values: list[float]) -> list[float] | None:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(finite) == 0:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def _validate(frame: pd.DataFrame) -> None:
    required = {
        "issue_time", "role", "unit_id", "scenario", "model", "label", "probability",
        "evidence_integrity", "availability_score", "decision_threshold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not set(frame["role"].astype(str)).issubset({"threshold", "score"}):
        raise ValueError("role must contain only threshold/score")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("label must be binary")
    for col in ("probability", "evidence_integrity", "availability_score"):
        values = frame[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{col} must be finite in [0,1]")
    thresholds = frame["decision_threshold"].to_numpy(dtype=float)
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("decision_threshold must be finite in [0,1]")


def evaluate(
    frame: pd.DataFrame,
    *,
    coverages: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    bootstrap_replicates: int = 10_000,
    seed: int = 20260909,
) -> dict[str, object]:
    frame = frame.copy()
    _validate(frame)
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True, errors="raise")
    selector_scores = _selector_scores(frame, seed)
    for name, values in selector_scores.items():
        frame[f"_selector_{name}"] = values

    score_units = tuple(dict.fromkeys(frame.loc[frame["role"] == "score", "unit_id"].astype(str)))
    draws = shared_cluster_bootstrap_draws(score_units, replicates=bootstrap_replicates, seed=seed)
    unit_to_positions = {}
    score_frame_all = frame[frame["role"] == "score"].copy()
    for unit, sub in score_frame_all.groupby(score_frame_all["unit_id"].astype(str), sort=False):
        unit_to_positions[str(unit)] = sub.index.to_numpy()

    results: dict[str, object] = {}
    contrasts: dict[str, object] = {}
    for (scenario, model), group in frame.groupby(["scenario", "model"], sort=True):
        threshold_part = group[group["role"] == "threshold"].copy()
        score_part = group[group["role"] == "score"].copy()
        if threshold_part.empty or score_part.empty:
            raise ValueError(f"{scenario}/{model}: threshold and score roles must both be nonempty")
        unique_thresholds = np.unique(group["decision_threshold"].to_numpy(dtype=float))
        if len(unique_thresholds) != 1:
            raise ValueError(f"{scenario}/{model}: decision_threshold must be frozen and constant")
        decision_threshold = float(unique_thresholds[0])
        cell_key = f"{scenario}::{model}"
        cell = {"decision_threshold": decision_threshold, "selectors": {}}

        masks: dict[tuple[str, float], np.ndarray] = {}
        for selector in SELECTORS:
            selector_cell = {}
            threshold_scores = threshold_part[f"_selector_{selector}"].to_numpy(dtype=float)
            score_scores = score_part[f"_selector_{selector}"].to_numpy(dtype=float)
            for coverage in coverages:
                cutoff = _cutoff_for_coverage(threshold_scores, coverage)
                mask = score_scores >= cutoff
                masks[(selector, coverage)] = mask
                metrics = _metrics(
                    score_part.loc[mask, "label"].to_numpy(dtype=int),
                    score_part.loc[mask, "probability"].to_numpy(dtype=float),
                    decision_threshold,
                )
                metrics["target_coverage"] = float(coverage)
                metrics["realized_coverage"] = float(np.mean(mask))
                metrics["selector_cutoff"] = cutoff
                selector_cell[str(coverage)] = metrics
            cell["selectors"][selector] = selector_cell
        results[cell_key] = cell

        for coverage in coverages:
            a_sel = "EVIDENCE_INTEGRITY"
            b_sel = "FORECAST_CONFIDENCE"
            a_point = cell["selectors"][a_sel][str(coverage)]
            b_point = cell["selectors"][b_sel][str(coverage)]
            boot = {metric: [] for metric in METRICS_FOR_CONTRAST}
            for draw in draws:
                sampled_idx = np.concatenate([unit_to_positions[u] for u in draw if u in unit_to_positions])
                sampled = frame.loc[sampled_idx]
                sampled = sampled[(sampled["scenario"] == scenario) & (sampled["model"] == model)]
                if sampled.empty:
                    continue
                a_scores = sampled[f"_selector_{a_sel}"].to_numpy(dtype=float)
                b_scores = sampled[f"_selector_{b_sel}"].to_numpy(dtype=float)
                a_cutoff = float(a_point["selector_cutoff"])
                b_cutoff = float(b_point["selector_cutoff"])
                ma = a_scores >= a_cutoff
                mb = b_scores >= b_cutoff
                am = _metrics(sampled.loc[ma, "label"].to_numpy(dtype=int),
                              sampled.loc[ma, "probability"].to_numpy(dtype=float),
                              decision_threshold)
                bm = _metrics(sampled.loc[mb, "label"].to_numpy(dtype=int),
                              sampled.loc[mb, "probability"].to_numpy(dtype=float),
                              decision_threshold)
                for metric in METRICS_FOR_CONTRAST:
                    delta = _metric_delta(am[metric], bm[metric])
                    if delta is not None and np.isfinite(delta):
                        boot[metric].append(float(delta))
            contrast_key = f"{cell_key}::coverage={coverage}"
            contrasts[contrast_key] = {
                "candidate": a_sel,
                "comparator": b_sel,
                "point_delta_candidate_minus_comparator": {
                    metric: _metric_delta(a_point[metric], b_point[metric])
                    for metric in METRICS_FOR_CONTRAST
                },
                "paired_shared_cluster_bootstrap_95_interval": {
                    metric: _percentile_interval(boot[metric])
                    for metric in METRICS_FOR_CONTRAST
                },
                "bootstrap_draw_count": int(bootstrap_replicates),
            }

    draw_digest = hashlib.sha256(
        json.dumps(draws, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "format": "IRIS_FORECAST_INTEGRITY_GATE_V1_RESULT",
        "selectors": list(SELECTORS),
        "coverages": list(coverages),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "shared_bootstrap_draw_table_sha256": draw_digest,
        "results": results,
        "primary_contrasts": contrasts,
        "protected_data_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    result = evaluate(frame, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "format": result["format"],
        "cells": len(result["results"]),
        "contrasts": len(result["primary_contrasts"]),
        "shared_bootstrap_draw_table_sha256": result["shared_bootstrap_draw_table_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
