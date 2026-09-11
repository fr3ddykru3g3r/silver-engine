"""Post-hoc causal FPR-controller V2 for the frozen Phase II onset model.

V2 changes only the online threshold controller operating point. The source
Phase II model, features, probabilities and protected-outcome boundary remain
unchanged. The 2017 replay is already inspected and this result is development
engineering only, not held-out validation.
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
from tools.run_fpr_controller_v1 import causal_controller, metrics_from_alert

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "fpr_controller_v2_development_2026-09-12.json"


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
    if cfg["study_id"] != "IRIS_SEP_FPR_CONTROLLER_V2":
        raise RuntimeError("unexpected V2 study id")
    if cfg["status"] != "FROZEN_BEFORE_V2_REPLAY":
        raise RuntimeError("V2 design is not frozen")
    if cfg["protected_boundary"]["protected_post_2025_outcomes_may_be_accessed"] is not False:
        raise RuntimeError("protected outcome boundary changed")
    if cfg["evaluation"]["already_inspected"] is not True:
        raise RuntimeError("V2 must remain explicitly post-hoc")
    return cfg


def run(out: Path) -> dict[str, Any]:
    cfg = load_contract()
    out.mkdir(parents=True, exist_ok=True)

    # Reproduce the frozen Phase II probabilities under the original design.
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
        post_horizon_label_buffer_hours=int(ccfg["post_horizon_label_buffer_hours"]),
    )
    controller_metrics = metrics_from_alert(y, replay["controller_alert"].to_numpy(int))

    maximum_fpr = float(cfg["requested_operating_goal"]["maximum_fpr"])
    minimum_pod = float(cfg["requested_operating_goal"]["minimum_pod"])
    require_tss_above_source = bool(cfg["requested_operating_goal"]["require_tss_above_source"])
    passes = bool(
        np.isfinite(controller_metrics["FPR"])
        and controller_metrics["FPR"] <= maximum_fpr
        and np.isfinite(controller_metrics["POD"])
        and controller_metrics["POD"] >= minimum_pod
        and ((not require_tss_above_source) or controller_metrics["TSS"] > original_metrics["TSS"])
    )

    replay.to_csv(out / "fpr_controller_v2_replay.csv", index=False)
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
        "requested_maximum_fpr": maximum_fpr,
        "requested_minimum_pod": minimum_pod,
        "fpr_reduction_absolute": float(original_metrics["FPR"] - controller_metrics["FPR"]),
        "fpr_reduction_relative": float((original_metrics["FPR"] - controller_metrics["FPR"]) / original_metrics["FPR"]),
        "passes_v2_development_gate": passes,
        "controller_contract": ccfg,
        "claim_boundary": cfg["claim_boundary"],
        "note": "V2 is post-hoc engineering on an already-inspected 2017 replay; it is not independent or held-out evidence."
    }
    dump_json(out / "fpr_controller_v2_summary.json", summary)
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
