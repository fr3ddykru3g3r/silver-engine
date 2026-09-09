"""Build the frozen 0.5 IRIS/XGBoost development diagnostic.

This module deliberately has no path to a test or SEPVAL artifact.  It consumes
only the two already sealed development artifacts and writes an immutable
diagnostic bundle under ``artifacts/development_hybrid_v2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    minimum_far_at_pod,
    paired_unit_bootstrap_tss_difference,
    probability_metrics,
    select_tss_threshold,
    threshold_metrics,
)


IRIS_SEP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = IRIS_SEP_ROOT / "artifacts" / "development_hybrid_v2"
IRIS_ROOT = IRIS_SEP_ROOT / "artifacts" / "local_tabular_v2"
ENSEMBLE_ROOT = IRIS_SEP_ROOT / "artifacts" / "local_baseline_ensembles_v1"

ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")
IDENTITY = ("issue_id", "role", "unit_id", "label")
EXPLORATORY_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
FIXED_IRIS_WEIGHT = 0.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260904

# These are the sealed source identities.  A changed source is a new run, not
# something this diagnostic should silently absorb.
EXPECTED_SOURCE_RECEIPT_HASHES = {
    "iris": "3e5f59f11a01d30d80557f841542441ec02077ea8581d612b1728e3422bdc1f5",
    "baseline_ensemble": "b3a4228079ed58bf4abcebe431e6babd41f724138d294fa1e27487f6af96ca5d",
}
EXPECTED_SOURCE_PREDICTION_HASHES = {
    "iris": "97ef474d22fe00b444716f43e41fa2335520ebaecfd8b0df15f020418aa487f7",
    "baseline_ensemble": "9db6efa73007dc5137c8733625458f2c3a4a79fc45602104f171a8f7b5340dc3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON receipt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return value


def _reject_locked_roles(frame: pd.DataFrame, *, name: str) -> None:
    if "role" not in frame.columns:
        raise ValueError(f"{name} is missing role")
    roles = frame["role"].astype("string")
    if roles.isna().any() or roles.str.contains(r"test|sepval|locked", case=False, regex=True).any():
        raise ValueError(f"{name} contains a test/SEPVAL/locked role")


def _validate_receipt(receipt: dict[str, object], *, name: str, expected_prediction_hash: str) -> None:
    if receipt.get("locked_test_accessed") is not False:
        raise ValueError(f"{name} receipt does not prove locked-test exclusion")
    if receipt.get("predictions_sha256", receipt.get("prediction_sha256")) != expected_prediction_hash:
        raise ValueError(f"{name} prediction hash does not match the sealed input")
    forbidden = {str(value).casefold() for value in receipt.get("claims_forbidden", [])}
    if "sepval_score" not in forbidden or "final_new_crossing_score" not in forbidden:
        raise ValueError(f"{name} receipt does not preserve the SEPVAL claim boundary")


def _validate_predictions(frame: pd.DataFrame, *, name: str) -> None:
    required = set(IDENTITY) | {"calibrated_probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} prediction schema is incomplete")
    _reject_locked_roles(frame, name=name)
    if not set(frame["role"].unique()).issubset(ROLES):
        raise ValueError(f"{name} contains an unknown development role")
    if frame[list(IDENTITY)].duplicated().any() or frame["issue_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate issue identities")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{name} labels are not binary")
    probabilities = pd.to_numeric(frame["calibrated_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities.to_numpy()).all() or not probabilities.between(0, 1).all():
        raise ValueError(f"{name} probabilities are not finite values in [0,1]")
    if set(frame["role"].unique()) != set(ROLES):
        raise ValueError(f"{name} is missing one or more required development roles")


def _metric_bundle(labels: Iterable[int], probabilities: Iterable[float], threshold: float, prevalence: float) -> dict[str, float]:
    return {
        **threshold_metrics(labels, probabilities, threshold),
        **probability_metrics(labels, probabilities, reference_probability=prevalence),
    }


def _xgboost_threshold(receipt: dict[str, object], xgboost: pd.DataFrame) -> float:
    """Validate the sealed XGBoost threshold's validation-only provenance.

    The historical ensemble receipt stores the threshold and threshold ID but
    does not include a separate ``fit_role`` field.  Recomputing the threshold
    from the sealed XGBoost ``validation_threshold`` rows and requiring an
    exact receipt match is therefore the fail-closed provenance check.
    """
    models = receipt.get("models")
    if not isinstance(models, list):
        raise ValueError("baseline ensemble receipt has no model records")
    for record in models:
        if isinstance(record, dict) and record.get("model") == "xgboost":
            threshold = record.get("threshold")
            threshold_id = record.get("threshold_id")
            if not isinstance(threshold, (float, int)) or not 0 <= float(threshold) <= 1:
                raise ValueError("invalid sealed XGBoost threshold")
            if not isinstance(threshold_id, str) or not threshold_id:
                raise ValueError("sealed XGBoost threshold provenance is missing threshold_id")
            threshold_rows = xgboost.loc[xgboost["role"] == "validation_threshold"]
            recomputed = select_tss_threshold(
                threshold_rows["label"], threshold_rows["calibrated_probability"], role="validation_threshold"
            )
            if float(threshold) != recomputed.threshold or threshold_id != recomputed.threshold_id:
                raise ValueError("sealed XGBoost threshold was not fit exclusively on validation_threshold")
            return recomputed.threshold
    raise ValueError("sealed baseline ensemble has no XGBoost threshold")


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Create one immutable development-only hybrid diagnostic receipt."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory exists; diagnostic artifacts are immutable")

    iris_receipt_path = IRIS_ROOT / "receipt.json"
    ensemble_receipt_path = ENSEMBLE_ROOT / "receipt.json"
    iris_predictions_path = IRIS_ROOT / "development_predictions.csv"
    ensemble_predictions_path = ENSEMBLE_ROOT / "development_ensemble_predictions.csv"
    source_paths = {
        "iris": (iris_receipt_path, iris_predictions_path),
        "baseline_ensemble": (ensemble_receipt_path, ensemble_predictions_path),
    }
    source_receipts: dict[str, dict[str, object]] = {}
    receipts: dict[str, dict[str, object]] = {}
    for name, (receipt_path, prediction_path) in source_paths.items():
        receipt_hash = sha256_file(receipt_path)
        prediction_hash = sha256_file(prediction_path)
        if receipt_hash != EXPECTED_SOURCE_RECEIPT_HASHES[name]:
            raise ValueError(f"{name} receipt hash mismatch")
        if prediction_hash != EXPECTED_SOURCE_PREDICTION_HASHES[name]:
            raise ValueError(f"{name} prediction hash mismatch")
        receipt = _read_json(receipt_path)
        _validate_receipt(receipt, name=name, expected_prediction_hash=prediction_hash)
        source_receipts[name] = {
            "receipt_path": str(receipt_path.relative_to(IRIS_SEP_ROOT)),
            "receipt_sha256": receipt_hash,
            "predictions_path": str(prediction_path.relative_to(IRIS_SEP_ROOT)),
            "predictions_sha256": prediction_hash,
        }
        receipts[name] = receipt

    # Validate all ensemble roles before selecting its XGBoost rows.  This
    # prevents an unexpected test-like row hidden in the other model's rows.
    iris = pd.read_csv(iris_predictions_path, float_precision="round_trip")
    ensemble_all = pd.read_csv(ensemble_predictions_path, float_precision="round_trip")
    _validate_predictions(iris, name="IRIS")
    _reject_locked_roles(ensemble_all, name="baseline ensemble")
    if "model" not in ensemble_all.columns or set(ensemble_all["model"].unique()) != {"elastic_net", "xgboost"}:
        raise ValueError("baseline ensemble model coverage is not exactly elastic_net and xgboost")
    # Validate both model partitions before selecting the comparator.  The
    # ensemble intentionally duplicates issue IDs across models, so duplicate
    # checks are performed within each model and then cross-model coverage is
    # checked explicitly.
    model_frames: dict[str, pd.DataFrame] = {}
    for model_name in ("elastic_net", "xgboost"):
        model_frame = ensemble_all.loc[ensemble_all["model"] == model_name].copy()
        _validate_predictions(model_frame, name=f"{model_name} ensemble")
        model_frames[model_name] = model_frame
    elastic_net_keys = model_frames["elastic_net"][list(IDENTITY)].sort_values("issue_id", kind="mergesort").reset_index(drop=True)
    xgb_keys = model_frames["xgboost"][list(IDENTITY)].sort_values("issue_id", kind="mergesort").reset_index(drop=True)
    if not elastic_net_keys.equals(xgb_keys):
        raise ValueError("elastic_net and xgboost ensemble identities are not exactly aligned")
    xgboost = model_frames["xgboost"]

    iris_keys = iris[list(IDENTITY)].sort_values("issue_id", kind="mergesort").reset_index(drop=True)
    if not iris_keys.equals(xgb_keys):
        raise ValueError("IRIS and XGBoost issue/role/unit/label identities are not exactly aligned")
    if iris["unit_id"].isna().any() or xgboost["unit_id"].isna().any():
        raise ValueError("unit identities cannot be missing")

    paired = iris[list(IDENTITY) + ["calibrated_probability"]].merge(
        xgboost[list(IDENTITY) + ["calibrated_probability"]],
        on=list(IDENTITY), how="outer", validate="one_to_one", suffixes=("_iris", "_xgboost"), sort=False,
    )
    if len(paired) != len(iris) or paired[["calibrated_probability_iris", "calibrated_probability_xgboost"]].isna().any().any():
        raise ValueError("exact paired probability cohort could not be formed")
    paired["blend_probability"] = (
        FIXED_IRIS_WEIGHT * paired["calibrated_probability_iris"]
        + (1.0 - FIXED_IRIS_WEIGHT) * paired["calibrated_probability_xgboost"]
    )
    paired["blend_weight_iris"] = FIXED_IRIS_WEIGHT
    paired["blend_weight_xgboost"] = 1.0 - FIXED_IRIS_WEIGHT
    predictions = paired[
        list(IDENTITY)
        + ["calibrated_probability_iris", "calibrated_probability_xgboost", "blend_probability", "blend_weight_iris", "blend_weight_xgboost"]
    ]

    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "development_hybrid_predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.17g")
    reloaded = pd.read_csv(predictions_path, float_precision="round_trip")
    if not reloaded[list(IDENTITY)].equals(predictions[list(IDENTITY)]):
        raise ValueError("17-digit output changed identity ordering during reload")
    if not np.allclose(reloaded["blend_probability"], predictions["blend_probability"], rtol=0.0, atol=5e-17):
        raise ValueError("17-digit output did not preserve blend probabilities")

    threshold_rows = reloaded.loc[reloaded["role"] == "validation_threshold"]
    blend_threshold = select_tss_threshold(
        threshold_rows["label"], threshold_rows["blend_probability"], role="validation_threshold"
    )
    xgboost_threshold = _xgboost_threshold(receipts["baseline_ensemble"], xgboost)
    prevalence = float(reloaded.loc[reloaded["role"] == "train", "label"].mean())
    metrics_by_role: dict[str, dict[str, float]] = {}
    for role in ROLES:
        rows = reloaded.loc[reloaded["role"] == role]
        metrics_by_role[role] = _metric_bundle(
            rows["label"], rows["blend_probability"], blend_threshold.threshold, prevalence
        )
    monitor = reloaded.loc[reloaded["role"] == "validation_monitor"]
    xgboost_monitor_metrics = _metric_bundle(
        monitor["label"], monitor["calibrated_probability_xgboost"], xgboost_threshold, prevalence
    )
    bootstrap = paired_unit_bootstrap_tss_difference(
        monitor["label"], monitor["blend_probability"], monitor["calibrated_probability_xgboost"], monitor["unit_id"],
        iris_threshold=blend_threshold.threshold, comparator_threshold=xgboost_threshold,
        replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    matched_pod = {
        "blend": minimum_far_at_pod(monitor["label"], monitor["blend_probability"], 0.8),
        "xgboost": minimum_far_at_pod(monitor["label"], monitor["calibrated_probability_xgboost"], 0.8),
    }

    # This grid is retained as a selection-biased monitor diagnostic only.  It
    # is never used to choose the frozen candidate weight, which is exactly 0.5.
    grid_rows: list[dict[str, object]] = []
    for weight in EXPLORATORY_GRID:
        candidate_probability = weight * reloaded["calibrated_probability_iris"] + (1.0 - weight) * reloaded["calibrated_probability_xgboost"]
        threshold_subset = candidate_probability[reloaded["role"] == "validation_threshold"]
        threshold_labels = reloaded.loc[reloaded["role"] == "validation_threshold", "label"]
        exploratory_threshold = select_tss_threshold(threshold_labels, threshold_subset, role="validation_threshold")
        monitor_probability = candidate_probability[reloaded["role"] == "validation_monitor"]
        monitor_labels = monitor["label"]
        monitor_metrics = threshold_metrics(monitor_labels, monitor_probability, exploratory_threshold.threshold)
        grid_rows.append({
            "iris_weight": weight,
            "xgboost_weight": 1.0 - weight,
            "threshold_fit_role": "validation_threshold",
            "threshold": exploratory_threshold.threshold,
            "validation_monitor_TSS": monitor_metrics["TSS"],
            "selection_biased": True,
            "used_for_weight_selection": False,
            "frozen_evaluated_candidate": bool(weight == FIXED_IRIS_WEIGHT),
        })
    grid = pd.DataFrame(grid_rows)
    grid_path = output_dir / "exploratory_grid_selection_biased.csv"
    grid.to_csv(grid_path, index=False, float_format="%.17g")
    grid_reloaded = pd.read_csv(grid_path, float_precision="round_trip")
    if grid_reloaded["iris_weight"].tolist() != list(EXPLORATORY_GRID):
        raise ValueError("exploratory grid did not round-trip exactly")

    source_script_hash = sha256_file(Path(__file__))
    receipt = {
        "status": "DEVELOPMENT_DIAGNOSTIC_INCONCLUSIVE",
        "diagnostic_scope": "FIXED_0_5_PROBABILITY_BLEND_DEVELOPMENT_ONLY",
        "superiority_claim": False,
        "locked_test_accessed": False,
        "roles_evaluated": list(ROLES),
        "identity_alignment": {
            "join_keys": list(IDENTITY),
            "exact_issue_count": int(len(reloaded)),
            "exact_unit_count": int(reloaded["unit_id"].nunique()),
        },
        "frozen_candidate": {
            "iris_probability_weight": FIXED_IRIS_WEIGHT,
            "xgboost_probability_weight": 1.0 - FIXED_IRIS_WEIGHT,
            "weight_selection": "PREDECLARED_FIXED_CANDIDATE_NO_FURTHER_TUNING",
            "threshold": blend_threshold.threshold,
            "threshold_fit_role": blend_threshold.fit_role,
            "threshold_objective": blend_threshold.objective,
            "threshold_id": blend_threshold.threshold_id,
        },
        "exploratory_grid": {
            "path": grid_path.name,
            "sha256": sha256_file(grid_path),
            "weights": list(EXPLORATORY_GRID),
            "selection_biased": True,
            "selection_role": "validation_monitor",
            "threshold_fit_role": "validation_threshold",
            "used_for_weight_selection": False,
            "note": "Persisted diagnostic sweep; monitor inspection is selection-biased and did not alter the frozen 0.5 candidate.",
        },
        "metrics_recomputed_after_reload": True,
        "prevalence_reference": prevalence,
        "validation_monitor_metrics": metrics_by_role["validation_monitor"],
        "xgboost_validation_monitor_metrics": xgboost_monitor_metrics,
        "metrics_by_role": metrics_by_role,
        "xgboost_comparator": {
            "threshold": xgboost_threshold,
            "threshold_fit_role": "validation_threshold",
            "threshold_provenance_check": "RECOMPUTED_FROM_XGBOOST_VALIDATION_THRESHOLD_ROWS_AND_MATCHED_TO_SOURCE_THRESHOLD_ID",
        },
        "paired_unit_bootstrap_tss_difference": {
            **bootstrap,
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "role": "validation_monitor",
            "comparator": "xgboost",
        },
        # Compatibility aliases use the naming in the prior development
        # comparison receipt while retaining the explicit diagnostic names.
        "paired_bootstrap": bootstrap,
        "matched_pod_0_8_far": matched_pod,
        "matched_pod_0_8_curve_diagnostic": matched_pod,
        "source_receipts": source_receipts,
        "source_sha256": source_script_hash,
        "prediction_sha256": sha256_file(predictions_path),
        "hashes": {
            "source_script_sha256": source_script_hash,
            "inputs": source_receipts,
            "predictions_sha256": sha256_file(predictions_path),
            "exploratory_grid_sha256": sha256_file(grid_path),
        },
        "claims_forbidden": [
            "SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH",
            "OPERATIONAL_CERTIFICATION", "SUPERIORITY", "PRODUCTION_READINESS",
        ],
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "predictions_sha256": result["hashes"]["predictions_sha256"],
        "paired_bootstrap": result["paired_unit_bootstrap_tss_difference"],
        "locked_test_accessed": result["locked_test_accessed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
