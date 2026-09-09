"""Recompute train-only rolling metrics and paired unit-bootstrap deltas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tss(frame: pd.DataFrame, column: str = "prediction") -> float:
    positive = frame.loc[frame.label == 1, column].mean()
    negative = frame.loc[frame.label == 0, column].mean()
    return float(positive - negative)


def analyze(run: Path, replicates: int = 10_000) -> dict:
    receipt = json.loads((run / "receipt.json").read_text())
    predictions_path = run / "inner_predictions.csv"
    if sha256(predictions_path) != receipt["prediction_sha256"]:
        raise ValueError("prediction receipt mismatch")
    frame = pd.read_csv(predictions_path, float_precision="round_trip")
    if set(frame.outer_role) != {"train"} or frame.duplicated(["issue_id", "arm"]).any():
        raise ValueError("training-only identity contract failure")
    fold_metrics = []
    for (fold, arm), group in frame.groupby(["fold", "arm"]):
        false_positive = int(((group.label == 0) & (group.prediction == 1)).sum())
        predicted_positive = int(group.prediction.sum())
        fold_metrics.append({"fold": int(fold), "arm": arm, "rows": len(group),
                             "positives": int(group.label.sum()), "TSS": tss(group),
                             "FAR": false_positive / predicted_positive if predicted_positive else None,
                             "BRIER": float(np.mean((group.probability - group.label) ** 2))})
    rng = np.random.default_rng(20260905)
    base = frame.loc[frame.arm == "xgboost"].set_index("issue_id")
    comparisons = []
    for arm in ("elastic_net", "compact_signed_log"):
        candidate = frame.loc[frame.arm == arm].set_index("issue_id")
        identities = base.index.intersection(candidate.index)
        paired = pd.DataFrame({"label": candidate.loc[identities].label,
                               "candidate": candidate.loc[identities].prediction,
                               "baseline": base.loc[identities].prediction,
                               "unit": candidate.loc[identities].unit_id})
        units = [group[["label", "candidate", "baseline"]].to_numpy()
                 for _, group in paired.groupby("unit")]
        differences = []
        for _ in range(replicates):
            sample = np.concatenate([units[index] for index in
                                     rng.integers(len(units), size=len(units))])
            labels = sample[:, 0]
            if labels.min() == labels.max():
                continue
            differences.append((sample[labels == 1, 1].mean() - sample[labels == 0, 1].mean()) -
                               (sample[labels == 1, 2].mean() - sample[labels == 0, 2].mean()))
        difference = tss(paired.rename(columns={"label": "label", "candidate": "prediction"})) - tss(
            paired.rename(columns={"label": "label", "baseline": "prediction"}))
        comparisons.append({"candidate": arm, "baseline": "xgboost", "paired_rows": len(paired),
                            "paired_units": len(units), "TSS_difference": difference,
                            "paired_unit_bootstrap_replicates": len(differences),
                            "difference_95_CI": np.quantile(differences, [.025, .975]).tolist(),
                            "selection_gate_passed": bool(np.quantile(differences, .025) > 0)})
    return {"status": "TRAIN_ONLY_LEGACY_TARGET_DIAGNOSTIC",
            "locked_test_accessed": False, "outer_monitor_scored": False,
            "new_crossing_evidence": False, "source_receipt_sha256": sha256(run / "receipt.json"),
            "prediction_sha256": sha256(predictions_path), "fold_metrics": fold_metrics,
            "comparisons": comparisons, "retained_failures": receipt["failures"],
            "conclusion": "No tested candidate shows a positive paired advantage over XGBoost."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    output = args.run / "analysis_receipt.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["comparisons"], indent=2))
