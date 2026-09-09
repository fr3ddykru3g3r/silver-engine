"""Paired development-only diagnostic for IRIS-SEP versus XGBoost."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    minimum_far_at_pod,
    paired_unit_bootstrap_tss_difference,
    threshold_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IRIS_RECEIPT = "3e5f59f11a01d30d80557f841542441ec02077ea8581d612b1728e3422bdc1f5"
EXPECTED_BASELINE_ENSEMBLE_RECEIPT = "b3a4228079ed58bf4abcebe431e6babd41f724138d294fa1e27487f6af96ca5d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("comparison directory exists; receipts are immutable")
    iris_root = ROOT / "artifacts" / "local_tabular_v2"
    baseline_root = ROOT / "artifacts" / "local_baseline_ensembles_v1"
    iris_receipt_path = iris_root / "receipt.json"
    baseline_receipt_path = baseline_root / "receipt.json"
    if sha256_file(iris_receipt_path) != EXPECTED_IRIS_RECEIPT or sha256_file(baseline_receipt_path) != EXPECTED_BASELINE_ENSEMBLE_RECEIPT:
        raise ValueError("source receipt hash mismatch")
    iris_receipt = json.loads(iris_receipt_path.read_text())
    baseline_receipt = json.loads(baseline_receipt_path.read_text())
    if iris_receipt["locked_test_accessed"] is not False or baseline_receipt["locked_test_accessed"] is not False:
        raise ValueError("locked-test boundary failure")
    iris_path = iris_root / "development_predictions.csv"
    baseline_path = baseline_root / "development_ensemble_predictions.csv"
    if sha256_file(iris_path) != iris_receipt["predictions_sha256"] or sha256_file(baseline_path) != baseline_receipt["prediction_sha256"]:
        raise ValueError("prediction hash mismatch")
    iris = pd.read_csv(iris_path, float_precision="round_trip")
    baseline = pd.read_csv(baseline_path, float_precision="round_trip")
    baseline = baseline.loc[baseline["model"] == "xgboost"].copy()
    monitor_iris = iris.loc[iris["role"] == "validation_monitor", ["issue_id", "unit_id", "label", "calibrated_probability"]]
    monitor_base = baseline.loc[baseline["role"] == "validation_monitor", ["issue_id", "unit_id", "label", "calibrated_probability"]]
    paired = monitor_iris.merge(monitor_base, on="issue_id", suffixes=("_iris", "_xgboost"), validate="one_to_one")
    if len(paired) != len(monitor_iris) or not (paired["unit_id_iris"] == paired["unit_id_xgboost"]).all() or not (paired["label_iris"] == paired["label_xgboost"]).all():
        raise ValueError("identical-cohort pairing failure")
    xgboost_record = next(row for row in baseline_receipt["models"] if row["model"] == "xgboost")
    iris_threshold = float(iris_receipt["threshold"]["value"])
    xgboost_threshold = float(xgboost_record["threshold"])
    labels = paired["label_iris"]
    iris_probability = paired["calibrated_probability_iris"]
    xgboost_probability = paired["calibrated_probability_xgboost"]
    bootstrap = paired_unit_bootstrap_tss_difference(
        labels, iris_probability, xgboost_probability, paired["unit_id_iris"],
        iris_threshold=iris_threshold, comparator_threshold=xgboost_threshold,
        replicates=10000, seed=20260904,
    )
    iris_metrics = threshold_metrics(labels, iris_probability, iris_threshold)
    xgboost_metrics = threshold_metrics(labels, xgboost_probability, xgboost_threshold)
    matched_pod = {
        "iris_sep": minimum_far_at_pod(labels, iris_probability, 0.8),
        "xgboost": minimum_far_at_pod(labels, xgboost_probability, 0.8),
    }
    passes_positive_interval = bootstrap["ci_lower_95"] > 0
    receipt = {
        "status": "DEVELOPMENT_DIAGNOSTIC_INCONCLUSIVE" if not passes_positive_interval else "DEVELOPMENT_DIAGNOSTIC_POSITIVE_NOT_FINAL",
        "role": "validation_monitor_USED_FOR_EARLY_STOPPING_NOT_UNBIASED",
        "locked_test_accessed": False,
        "identical_issue_count": len(paired),
        "identical_unit_count": int(paired["unit_id_iris"].nunique()),
        "iris_threshold": iris_threshold,
        "xgboost_threshold": xgboost_threshold,
        "iris_metrics": iris_metrics,
        "xgboost_metrics": xgboost_metrics,
        "paired_bootstrap": bootstrap,
        "paired_ci_lower_above_zero": passes_positive_interval,
        "matched_pod_0_8_curve_diagnostic": matched_pod,
        "source_receipts": {
            "iris": EXPECTED_IRIS_RECEIPT,
            "baseline_ensemble": EXPECTED_BASELINE_ENSEMBLE_RECEIPT,
        },
        "claims_forbidden": ["SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION", "SUPERIORITY"],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    result = run(ROOT / "artifacts" / "development_comparison_v2")
    print(json.dumps({"status": result["status"], "paired_bootstrap": result["paired_bootstrap"], "locked_test_accessed": result["locked_test_accessed"]}, sort_keys=True))
