"""Causal adaptive threshold controller for the frozen Phase II onset forecaster.

This is a retrospective development-only controller replay on an already-inspected
2017 score period. It does not alter Phase II model features, training,
hyperparameters, calibration, or probabilities. It only changes the alert
threshold online using previously resolved negative outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import tools.run_onset_forecaster_phase2_v1 as phase2
import tools.run_onset_forecaster_phase2_v1_execution as phase2_execution


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "fpr_controller_v1_preregistration_2026-09-12.json"


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_contract() -> dict[str, Any]:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    if cfg["study_id"] != "IRIS_SEP_FPR_CONTROLLER_V1":
        raise RuntimeError("unexpected FPR-controller study id")
    if cfg["status"] != "FROZEN_BEFORE_CONTROLLER_REPLAY":
        raise RuntimeError("FPR-controller contract is not frozen")
    if cfg["protected_boundary"]["protected_post_2025_outcomes_may_be_accessed"] is not False:
        raise RuntimeError("protected outcome boundary changed")
    return cfg


def metrics_from_alert(y: np.ndarray, alert: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    alert = np.asarray(alert, dtype=bool)
    c = {
        "tp": int(np.sum((y == 1) & alert)),
        "fn": int(np.sum((y == 1) & ~alert)),
        "fp": int(np.sum((y == 0) & alert)),
        "tn": int(np.sum((y == 0) & ~alert)),
    }
    return {**c, **phase2.derived_metrics(c)}


def causal_controller(
    issue_times: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    base_threshold: float,
    *,
    quantile: float,
    history_window_days: int,
    minimum_resolved_negative_history: int,
    label_availability_lag_hours: int,
) -> pd.DataFrame:
    """Generate one threshold per issue using only labels available before it."""
    times = pd.to_datetime(issue_times, utc=True).reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not (len(times) == len(y) == len(p)):
        raise ValueError("controller arrays differ in length")

    thresholds = np.full(len(y), float(base_threshold), dtype=float)
    alerts = np.zeros(len(y), dtype=int)
    history_sizes = np.zeros(len(y), dtype=int)
    quantile_thresholds = np.full(len(y), np.nan, dtype=float)

    lag = pd.Timedelta(hours=int(label_availability_lag_hours))
    window = pd.Timedelta(days=int(history_window_days))

    for i in range(len(y)):
        now = times.iloc[i]
        lower = now - window
        # A prior row's 24 h outcome must be fully resolved, plus the frozen
        # extra lag, before it is eligible to update the controller.
        resolved_cutoff = now - lag
        hist_mask = (
            (times < now)
            & (times >= lower)
            & ((times + pd.Timedelta(hours=24)) <= resolved_cutoff)
            & (y == 0)
        ).to_numpy(dtype=bool)
        hist_scores = p[hist_mask]
        hist_scores = hist_scores[np.isfinite(hist_scores)]
        history_sizes[i] = int(len(hist_scores))

        threshold = float(base_threshold)
        if len(hist_scores) >= int(minimum_resolved_negative_history):
            qth = float(np.quantile(hist_scores, float(quantile), method="higher"))
            quantile_thresholds[i] = qth
            threshold = max(threshold, qth)

        thresholds[i] = threshold
        alerts[i] = int(np.isfinite(p[i]) and p[i] >= threshold)

    return pd.DataFrame(
        {
            "issue_timestamp": times,
            "y_onset": y,
            "probability": p,
            "base_threshold": float(base_threshold),
            "controller_threshold": thresholds,
            "quantile_threshold": quantile_thresholds,
            "resolved_negative_history_size": history_sizes,
            "controller_alert": alerts,
        }
    )


def run(out: Path) -> dict[str, Any]:
    cfg = load_contract()
    out.mkdir(parents=True, exist_ok=True)

    # Reproduce the already-frozen Phase II model exactly. The compatibility
    # adapter changes runtime representation only and keeps the protected pool sealed.
    base_dir = out / "phase2_fixed_replay"
    base_summary = phase2_execution.run(base_dir)
    if base_summary["protected_post_2025_outcomes_accessed"] is not False:
        raise RuntimeError("protected outcomes were accessed")

    pred = pd.read_csv(base_dir / "phase2_score_predictions.csv")
    pred["issue_timestamp"] = pd.to_datetime(pred["issue_timestamp"], utc=True)
    pred = pred.sort_values("issue_timestamp").reset_index(drop=True)

    source_model = str(cfg["source_model"])
    pcol = f"p__{source_model}"
    if pcol not in pred.columns:
        raise RuntimeError(f"missing source-model probability column: {pcol}")

    y = pred["y_onset"].astype(int).to_numpy()
    p = pred[pcol].astype(float).to_numpy()
    original_threshold = float(base_summary["threshold_selection"][source_model]["chosen"]["threshold"])
    original_metrics = phase2.full_metrics(y, p, original_threshold)

    ccfg = cfg["controller"]
    replay = causal_controller(
        pred["issue_timestamp"],
        y,
        p,
        original_threshold,
        quantile=float(ccfg["quantile"]),
        history_window_days=int(ccfg["history_window_days"]),
        minimum_resolved_negative_history=int(ccfg["minimum_resolved_negative_history"]),
        label_availability_lag_hours=int(ccfg["label_availability_lag_hours"]),
    )
    controller_metrics = metrics_from_alert(y, replay["controller_alert"].to_numpy(int))
    requested_cap = float(cfg["requested_operating_goal"]["maximum_fpr"])
    passed = bool(np.isfinite(controller_metrics["FPR"]) and controller_metrics["FPR"] <= requested_cap)

    replay.to_csv(out / "fpr_controller_replay.csv", index=False)
    summary = {
        "study_id": cfg["study_id"],
        "status": "DEVELOPMENT_COMPLETE",
        "protected_post_2025_outcomes_accessed": False,
        "source_model": source_model,
        "score_rows": int(len(replay)),
        "score_positive_onsets": int(np.sum(y == 1)),
        "original_fixed_threshold": original_threshold,
        "original_metrics": {k: float(original_metrics[k]) for k in ("POD", "FPR", "FAR", "TSS", "HSS")},
        "controller_metrics": {k: float(controller_metrics[k]) for k in ("POD", "FPR", "FAR", "TSS", "HSS")},
        "controller_confusion": {k: int(controller_metrics[k]) for k in ("tp", "fn", "fp", "tn")},
        "requested_fpr_cap": requested_cap,
        "fpr_reduction_absolute": float(original_metrics["FPR"] - controller_metrics["FPR"]),
        "fpr_reduction_relative": float((original_metrics["FPR"] - controller_metrics["FPR"]) / original_metrics["FPR"]),
        "passes_requested_fpr_cap": passed,
        "controller_contract": ccfg,
        "claim_boundary": cfg["claim_boundary"],
        "note": "2017 was already inspected in Phase II V1; this is post-hoc causal-controller engineering evidence only."
    }
    dump_json(out / "fpr_controller_summary.json", summary)
    dump_json(out / "contract_snapshot.json", cfg)
    dump_json(
        out / "evidence_hashes.json",
        {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "evidence_hashes.json"},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
