"""Run a fixed five-seed XGBoost development baseline without test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    apply_calibration,
    fit_intercept_calibration,
    probability_metrics,
    select_tss_threshold,
    threshold_metrics,
)


IRIS_SEP_ROOT = Path(__file__).resolve().parents[1]
TARGET = "future_Operational_SEP_label"
META = {"issue_id", "role", "unit_id", "window_begin", "window_end", TARGET}
ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")
SEEDS = (7, 13, 26, 42, 73)
EXPECTED_SOURCE_SHA256 = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
EXPECTED_MANIFEST_SHA256 = "18c10d4fc76a2ce5e03b9a271951003f274435aa00180fcb90e4f2947eedaebb"
EXPECTED_FEATURE_MANIFEST_SHA256 = "7bca82f223f1be0adbd8afc6e30aed238ed52b3bb2339a98fa9c9cbd944436b5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-8, 1 - 1e-8)
    return np.log(clipped / (1 - clipped))


def run(source: Path, source_manifest: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise ValueError("output directory already exists; runs are immutable")
    if sha256_file(source) != EXPECTED_SOURCE_SHA256 or sha256_file(source_manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source and manifest must match the pinned development-v3 cohort")
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("output_sha256") != EXPECTED_SOURCE_SHA256 or manifest.get("locked_test_rows_present") is not False:
        raise ValueError("source manifest does not bind a non-locked development cohort")
    frame = pd.read_csv(source)
    if not META.issubset(frame.columns) or set(frame["role"].unique()) != set(ROLES):
        raise ValueError("development table contract failure")
    if frame["issue_id"].duplicated().any() or frame.groupby("unit_id")["role"].nunique().max() != 1:
        raise ValueError("identity or unit isolation failure")
    features = [name for name in frame.columns if name not in META]
    feature_hash = hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest()
    if feature_hash != EXPECTED_FEATURE_MANIFEST_SHA256 or any(name.lower().startswith("future_") for name in features):
        raise ValueError("feature allowlist failure")
    roles = {role: frame.loc[frame["role"] == role].copy() for role in ROLES}
    for left, right in zip(ROLES, ROLES[1:]):
        left_end = pd.to_datetime(roles[left]["window_end"], utc=True).max()
        right_start = pd.to_datetime(roles[right]["window_end"], utc=True).min()
        if right_start <= left_end + pd.Timedelta(hours=24):
            raise ValueError(f"inclusive purge failure between {left} and {right}")
    train = roles["train"]
    prevalence = float(train[TARGET].mean())
    positive_weight = float((train[TARGET] == 0).sum() / (train[TARGET] == 1).sum())
    fixed_parameters = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "scale_pos_weight": positive_weight,
        "n_jobs": 4,
    }
    output_dir.mkdir(parents=True)
    predictions: list[pd.DataFrame] = []
    runs: list[dict[str, object]] = []
    for seed in SEEDS:
        model = XGBClassifier(**fixed_parameters, random_state=seed)
        model.fit(
            train[features], train[TARGET],
            eval_set=[(roles["validation_monitor"][features], roles["validation_monitor"][TARGET])],
            verbose=False,
        )
        model_path = output_dir / f"xgboost_seed_{seed}.json"
        model.save_model(model_path)
        logits_by_role = {
            role: _logits(model.predict_proba(group[features])[:, 1])
            for role, group in roles.items()
        }
        calibration = fit_intercept_calibration(
            logits_by_role["validation_calibration"], roles["validation_calibration"][TARGET],
            role="validation_calibration",
        )
        probability_by_role = {role: apply_calibration(logits, calibration) for role, logits in logits_by_role.items()}
        threshold = select_tss_threshold(
            roles["validation_threshold"][TARGET], probability_by_role["validation_threshold"],
            role="validation_threshold",
        )
        role_metrics = {}
        for role, group in roles.items():
            probability = probability_by_role[role]
            role_metrics[role] = {
                **threshold_metrics(group[TARGET], probability, threshold.threshold),
                **probability_metrics(group[TARGET], probability, reference_probability=prevalence),
            }
            predictions.append(pd.DataFrame({
                "issue_id": group["issue_id"], "role": role, "unit_id": group["unit_id"],
                "label": group[TARGET].astype(int), "model": "xgboost", "seed": seed,
                "calibrated_probability": probability,
            }))
        runs.append({
            "seed": seed,
            "best_iteration": int(model.best_iteration),
            "calibration_id": calibration.calibration_id,
            "calibration_intercept": calibration.intercept,
            "threshold_id": threshold.threshold_id,
            "threshold": threshold.threshold,
            "metrics": role_metrics,
            "model_artifact": model_path.name,
            "model_artifact_sha256": sha256_file(model_path),
        })
    prediction_path = output_dir / "development_predictions.csv"
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False, float_format="%.17g")
    reloaded = pd.read_csv(prediction_path, float_precision="round_trip")
    for result in runs:
        for role in ROLES:
            saved = reloaded.loc[(reloaded["seed"] == result["seed"]) & (reloaded["role"] == role)]
            result["metrics"][role] = {
                **threshold_metrics(saved["label"], saved["calibrated_probability"], result["threshold"]),
                **probability_metrics(saved["label"], saved["calibrated_probability"], reference_probability=prevalence),
            }
    receipt = {
        "status": "PASS_DEVELOPMENT_ONLY",
        "source_sha256": sha256_file(source),
        "source_manifest_sha256": sha256_file(source_manifest),
        "feature_manifest_sha256": feature_hash,
        "target_semantics": "PUBLISHER_LEGACY_WINDOW_LABEL_NOT_FINAL_NEW_CROSSING_TARGET",
        "locked_test_accessed": False,
        "fixed_parameters": fixed_parameters,
        "seeds": list(SEEDS),
        "training_prevalence": prevalence,
        "role_prevalence": {role: float(group[TARGET].mean()) for role, group in roles.items()},
        "selection_side_metrics": list(ROLES[1:]),
        "headline_eligible_roles": [],
        "runs": runs,
        "predictions_sha256": sha256_file(prediction_path),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgboost.__version__},
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "evaluation": sha256_file(IRIS_SEP_ROOT / "workstreams" / "luna_i_eval_ops" / "evaluation.py"),
            "benchmark_contract": sha256_file(IRIS_SEP_ROOT / "config" / "benchmark_contract_v2.json"),
            "evaluation_policy": sha256_file(IRIS_SEP_ROOT / "config" / "evaluation_policy_v1.json"),
        },
        "claims_forbidden": ["SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION"],
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.source, args.source_manifest, args.output_dir)
    print(json.dumps({"status": receipt["status"], "seeds": receipt["seeds"], "locked_test_accessed": receipt["locked_test_accessed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
