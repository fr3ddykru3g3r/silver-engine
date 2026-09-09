"""Create median calibrated-probability ensembles from fixed seed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    probability_metrics,
    select_tss_threshold,
    threshold_metrics,
)


IRIS_SEP_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "elastic_net": ("artifacts/local_baselines_v4", "3c2ffc72de523cf0292477cf7a6a52b67df29b9d27fca0284212b9a819dcf711"),
    "xgboost": ("artifacts/local_xgboost_v1", "968b9d98412fd0933080ac7fa178d85cc220e4e8ac7d15b4635adc7629729347"),
}
ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("output directory already exists; ensembles are immutable")
    output_dir.mkdir(parents=True)
    all_predictions = []
    source_receipts = {}
    for model, (relative, expected_receipt_hash) in EXPECTED.items():
        root = IRIS_SEP_ROOT / relative
        receipt_path = root / "receipt.json"
        if sha256_file(receipt_path) != expected_receipt_hash:
            raise ValueError(f"{model} receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text())
        prediction_path = root / "development_predictions.csv"
        if sha256_file(prediction_path) != receipt["predictions_sha256"] or receipt["locked_test_accessed"] is not False:
            raise ValueError(f"{model} prediction binding or test boundary failure")
        frame = pd.read_csv(prediction_path, float_precision="round_trip")
        if model == "elastic_net":
            frame = frame.loc[frame["model"] == "elastic_net"].copy()
        if set(frame["seed"].unique()) != {7, 13, 26, 42, 73}:
            raise ValueError(f"{model} must contain exactly five frozen seeds")
        identity = ["issue_id", "role", "unit_id", "label"]
        if frame.groupby(identity, dropna=False)["seed"].nunique().ne(5).any():
            raise ValueError(f"{model} has incomplete seed coverage")
        ensemble = frame.groupby(identity, as_index=False, sort=False)["calibrated_probability"].median()
        ensemble.insert(4, "model", model)
        if ensemble["role"].str.contains("test|sepval|locked", case=False).any():
            raise ValueError("test-like role reached ensemble builder")
        all_predictions.append(ensemble)
        source_receipts[model] = {
            "receipt_path": str(Path(relative) / "receipt.json"),
            "receipt_sha256": expected_receipt_hash,
            "predictions_sha256": receipt["predictions_sha256"],
        }
    output = pd.concat(all_predictions, ignore_index=True)
    output_path = output_dir / "development_ensemble_predictions.csv"
    output.to_csv(output_path, index=False, float_format="%.17g")
    saved = pd.read_csv(output_path, float_precision="round_trip")
    model_receipts = []
    for model in EXPECTED:
        model_rows = saved.loc[saved["model"] == model]
        threshold_rows = model_rows.loc[model_rows["role"] == "validation_threshold"]
        threshold = select_tss_threshold(
            threshold_rows["label"], threshold_rows["calibrated_probability"], role="validation_threshold"
        )
        prevalence = float(model_rows.loc[model_rows["role"] == "train", "label"].mean())
        metrics = {}
        for role in ROLES:
            role_rows = model_rows.loc[model_rows["role"] == role]
            metrics[role] = {
                **threshold_metrics(role_rows["label"], role_rows["calibrated_probability"], threshold.threshold),
                **probability_metrics(role_rows["label"], role_rows["calibrated_probability"], reference_probability=prevalence),
            }
        model_receipts.append({
            "model": model,
            "aggregation": "MEDIAN_OF_FIVE_SEED_CALIBRATED_PROBABILITIES",
            "threshold": threshold.threshold,
            "threshold_id": threshold.threshold_id,
            "metrics": metrics,
        })
    receipt = {
        "status": "PASS_DEVELOPMENT_ONLY",
        "locked_test_accessed": False,
        "source_receipts": source_receipts,
        "prediction_sha256": sha256_file(output_path),
        "models": model_receipts,
        "selection_side_metrics": ["validation_monitor", "validation_calibration", "validation_threshold"],
        "headline_eligible_roles": [],
        "evaluation_policy_sha256": sha256_file(IRIS_SEP_ROOT / "config" / "evaluation_policy_v1.json"),
        "claims_forbidden": ["SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION"],
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    destination = IRIS_SEP_ROOT / "artifacts" / "local_baseline_ensembles_v1"
    result = build(destination)
    print(json.dumps({"status": result["status"], "models": [row["model"] for row in result["models"]], "locked_test_accessed": result["locked_test_accessed"]}, sort_keys=True))
