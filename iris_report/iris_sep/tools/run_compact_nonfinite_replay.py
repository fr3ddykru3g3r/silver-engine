"""Replay the retained fold-3 original compact failure layer by layer.

The tool never downloads data. It runs only when the exact local train-only
source, diagnostic preregistration, folds, failing logits, preprocessing, and
checkpoint are already present. It also verifies the source implementation
hashes frozen by the original diagnostic before interpreting any replay result.
Otherwise it writes a bounded NOT_RUN receipt rather than guessing a cause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EXPECTED_SOURCE_SHA256 = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
EXPECTED_FAILURE_LOGITS_SHA256 = "68daf463bb3092b22fab09fe65480e5b46a136cf0fe71e9413ebeeb01eeed7d7"
EXPECTED_PREREGISTRATION_SHA256 = "f8099ec7efb2da16672ab7e4cf69beb05c92bdb5b5e7e04ec6e5bac82de3eee8"
TARGET = "future_Operational_SEP_label"
META = {"issue_id", "role", "unit_id", "window_begin", "window_end", TARGET}
MODALITIES = ("magnetic", "eruption", "particle_context")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError("immutable replay receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    result = {name: [] for name in MODALITIES}
    for name in features:
        lowered = name.lower()
        if lowered.startswith(("sharp_", "smarp_")):
            result["magnetic"].append(name)
        elif lowered.startswith(("flare_", "cme_", "xrs_", "goes_xrs_")):
            result["eruption"].append(name)
        elif lowered.startswith(("protonflux_", "proton_flux_", "proton_", "goes_proton_")):
            result["particle_context"].append(name)
        else:
            raise ValueError(f"unmapped predictor column: {name}")
    return result


def audit_preregistered_dependencies(root: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Check the exact implementation files frozen before the diagnostic run."""
    dependencies = preregistration.get("dependency_sha256")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise ValueError("diagnostic preregistration must bind implementation dependencies")
    files: dict[str, Any] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted((str(k), str(v)) for k, v in dependencies.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            files[relative] = {"expected_sha256": expected, "actual_sha256": None, "matches": False}
            continue
        actual = digest(path)
        matches = actual == expected
        files[relative] = {"expected_sha256": expected, "actual_sha256": actual, "matches": matches}
        if not matches:
            mismatched.append(relative)
    return {"files": files, "missing": missing, "mismatched": mismatched,
            "all_match": not missing and not mismatched}


def _model_receipt_audit(receipt: Mapping[str, Any], *, checkpoint_path: Path,
                         preprocessing_path: Path, fold: Mapping[str, Any],
                         predix: Any) -> dict[str, Any]:
    """Bind the retained seed artifact to the exact fold and preprocessing."""
    expected_predict = [int(v) for v in predix.tolist()]
    expected_fit = [int(v) for v in fold["fit"]]
    expected_stop = [int(v) for v in fold["stop"]]
    checks = {
        "seed": receipt.get("seed") == 7,
        "fit_role": receipt.get("fit_role") == "train",
        "stop_role": receipt.get("stop_role") == "train",
        "checkpoint_logits_verified_equal": receipt.get("checkpoint_logits_verified_equal") is True,
        "best_checkpoint_sha256": receipt.get("best_checkpoint_sha256") == digest(checkpoint_path),
        "preprocessing_sha256": receipt.get("preprocessing_sha256") == digest(preprocessing_path),
        "fit_indices": receipt.get("fit_indices") == expected_fit,
        "stop_indices": receipt.get("stop_indices") == expected_stop,
        "predict_indices": receipt.get("predict_indices") == expected_predict,
    }
    return {"checks": checks, "all_match": all(checks.values())}


def run(root: Path, diagnostic_dir: Path, output: Path) -> dict[str, Any]:
    source = root / "data_processed/sepnet_v1_development_v3.csv"
    preregistration_path = diagnostic_dir / "preregistration.json"
    folds_path = diagnostic_dir / "folds.json"
    arm_dir = diagnostic_dir / "fold_3/compact"
    saved_logits_path = arm_dir / "seed_7.npz"
    model_dir = arm_dir / "model_7"
    checkpoint_path = model_dir / "best.pt"
    preprocessing_path = model_dir / "preprocessing.json"
    model_receipt_path = model_dir / "receipt.json"
    required = [source, preregistration_path, folds_path, saved_logits_path,
                checkpoint_path, preprocessing_path, model_receipt_path]
    missing = [str(path) for path in required if not path.exists()]
    base = {
        "scope": "TRAIN_ONLY_LEGACY_TARGET_COMPACT_FAILURE_REPLAY",
        "locked_test_accessed": False,
        "outer_monitor_scored": False,
        "fold": 3,
        "arm": "compact",
        "seed": 7,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "expected_preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "expected_retained_failure_logits_sha256": EXPECTED_FAILURE_LOGITS_SHA256,
    }
    if missing:
        result = {**base, "status": "NOT_RUN_MISSING_LOCAL_ARTIFACTS", "missing_paths": missing,
                  "causal_conclusion": None,
                  "note": "Missing local artifacts are an environment blocker, not a scientific failure. No download was attempted."}
        save(output, result)
        return result

    preregistration_sha256 = digest(preregistration_path)
    if preregistration_sha256 != EXPECTED_PREREGISTRATION_SHA256:
        result = {**base, "status": "NOT_RUN_PREREGISTRATION_HASH_MISMATCH",
                  "preregistration_sha256": preregistration_sha256,
                  "causal_conclusion": None,
                  "note": "The diagnostic preregistration is not the published frozen copy; no replay interpretation is permitted."}
        save(output, result)
        return result
    preregistration = json.loads(preregistration_path.read_text())
    if (preregistration.get("scope") != "TRAIN_ONLY_LEGACY_TARGET_DIAGNOSTIC_NOT_FINAL" or
            preregistration.get("source_sha256") != EXPECTED_SOURCE_SHA256 or
            preregistration.get("locked_test_accessed") is not False):
        raise ValueError("diagnostic preregistration contract mismatch")
    implementation_audit = audit_preregistered_dependencies(root, preregistration)
    if not implementation_audit["all_match"]:
        result = {**base, "status": "NOT_RUN_IMPLEMENTATION_PROVENANCE_MISMATCH",
                  "preregistration_sha256": preregistration_sha256,
                  "implementation_audit": implementation_audit,
                  "causal_conclusion": None,
                  "note": "The replay implementation does not match the source hashes frozen before the retained diagnostic. Restore the hash-matched source before interpreting a mismatch or first-nonfinite stage."}
        save(output, result)
        return result

    import numpy as np
    import pandas as pd
    import torch
    from iris_report.iris_sep.src.iris_sep.modeling.compact_layer_replay import replay_model_layers, transform_from_preprocessing
    from iris_report.iris_sep.src.iris_sep.modeling.tabular_multibranch import BranchInput, IRISSEPTabularModel, TabularModelConfig

    if digest(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("pinned train-only source mismatch")
    if digest(saved_logits_path) != EXPECTED_FAILURE_LOGITS_SHA256:
        raise ValueError("retained failing logits mismatch")
    frame = pd.read_csv(source)
    frame = frame.loc[frame.role == "train"].sort_values("window_end").reset_index(drop=True)
    features = [name for name in frame.columns if name not in META]
    if features != preregistration.get("features"):
        raise ValueError("feature order does not match diagnostic preregistration")
    groups = feature_groups(features)
    folds = json.loads(folds_path.read_text())
    if not isinstance(folds, list) or len(folds) <= 3 or not isinstance(folds[3], Mapping):
        raise ValueError("retained fold definition missing")
    fold = folds[3]["indices"]
    if not isinstance(fold, Mapping) or any(name not in fold for name in ("fit", "stop", "calibration", "threshold", "score")):
        raise ValueError("retained fold roles missing")
    predix = np.concatenate([np.asarray(fold[name], dtype=np.int64) for name in ("calibration", "threshold", "score")])
    if len(predix) != 2120:
        raise ValueError("retained failure row count changed")
    preprocessing = json.loads(preprocessing_path.read_text())
    raw = {name: frame.iloc[predix][columns].to_numpy(dtype=np.float64) for name, columns in groups.items() if columns}
    values, masks, scaling_audit = transform_from_preprocessing(raw, preprocessing)
    for modality in MODALITIES:
        meta = preprocessing["modalities"][modality]
        expected_columns = groups[modality]
        if meta.get("always_unavailable") is True:
            if expected_columns:
                raise ValueError(f"{modality} marked unavailable despite preregistered columns")
        elif list(meta.get("columns", ())) != expected_columns:
            raise ValueError(f"{modality} preprocessing columns do not match preregistered order")
    config = TabularModelConfig(
        magnetic_features=values["magnetic"].shape[1], eruption_features=values["eruption"].shape[1],
        particle_context_features=values["particle_context"].shape[1],
        magnetic_feature_names=tuple(groups["magnetic"]), eruption_feature_names=tuple(groups["eruption"]),
        particle_context_feature_names=tuple(groups["particle_context"] or ["__PARTICLE_CONTEXT_UNAVAILABLE_IN_V1__"]),
    )
    device = torch.device("cpu")
    model = IRISSEPTabularModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model_receipt = json.loads(model_receipt_path.read_text())
    model_receipt_audit = _model_receipt_audit(model_receipt, checkpoint_path=checkpoint_path,
                                               preprocessing_path=preprocessing_path,
                                               fold=fold, predix=predix)
    if not model_receipt_audit["all_match"]:
        raise ValueError("retained model receipt does not bind the checkpoint, preprocessing, and fold")
    inputs = {name: BranchInput(torch.from_numpy(values[name]), torch.from_numpy(masks[name])) for name in MODALITIES}
    replay = replay_model_layers(model, inputs)
    model.eval()
    with torch.no_grad():
        direct = model(inputs, apply_missing_modality_dropout=False).primary_logit.detach().cpu().numpy().astype(np.float64)
    saved_logits = np.load(saved_logits_path)["logits"].astype(np.float64)
    exact = bool(np.array_equal(direct, saved_logits, equal_nan=True))
    if not exact:
        status = "REPLAY_MISMATCH_DO_NOT_DIAGNOSE"
        causal = None
    elif replay["first_nonfinite_stage"] is None:
        status = "FAILURE_NOT_REPRODUCED_DO_NOT_DIAGNOSE"
        causal = None
    else:
        status = "NONFINITE_REPRODUCED_WITH_FIRST_FAILURE_STAGE"
        causal = replay["first_nonfinite_stage"]
    result = {
        **base,
        "status": status,
        "source_sha256": digest(source),
        "preregistration_sha256": preregistration_sha256,
        "implementation_audit": implementation_audit,
        "folds_sha256": digest(folds_path),
        "retained_failure_logits_sha256": digest(saved_logits_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "preprocessing_sha256": digest(preprocessing_path),
        "model_receipt_sha256": digest(model_receipt_path),
        "model_receipt_audit": model_receipt_audit,
        "prediction_rows": int(len(predix)),
        "saved_logits_exactly_reproduced": exact,
        "scaling_audit": scaling_audit,
        "layer_replay": replay,
        "causal_conclusion": causal,
        "causal_boundary": "The first nonfinite tensor location is proven only when the frozen diagnostic preregistration and implementation hashes match, the retained model receipt binds the exact checkpoint/preprocessing/fold, and the saved logits reproduce exactly. Helper mutation, distribution shift, and preprocessing overflow remain unproven unless separately demonstrated by the replay inputs/audit.",
    }
    save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("iris_report/iris_sep"))
    parser.add_argument("--diagnostic-dir", type=Path, default=Path("iris_report/iris_sep/artifacts/train_inner_diagnostic_v4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.diagnostic_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
