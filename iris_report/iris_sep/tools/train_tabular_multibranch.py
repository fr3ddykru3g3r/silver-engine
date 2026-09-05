"""Train the compact IRIS-SEP model on non-locked development roles.

This runner is designed for Colab GPU execution but remains a normal Python
program. It never accepts a test role, fits preprocessing on train only, uses
the monitor role only for early stopping, calibration only for calibration,
and threshold selection only for threshold. The legacy V1 target is not the
final audited new-crossing target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


IRIS_SEP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IRIS_SEP_ROOT / "src"))

from iris_sep.modeling.tabular_multibranch import (  # noqa: E402
    BranchInput,
    IRISSEPTabularModel,
    MODALITIES,
    TabularModelConfig,
)
from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (  # noqa: E402
    apply_calibration,
    fit_intercept_calibration,
    probability_metrics,
    select_tss_threshold,
    sigmoid,
    threshold_metrics,
)


TARGET = "future_Operational_SEP_label"
META = {"issue_id", "role", "unit_id", "window_begin", "window_end", TARGET}
ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")
SEEDS = (7, 13, 26, 42, 73)
EXPECTED_SOURCE_SHA256 = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
EXPECTED_MANIFEST_SHA256 = "18c10d4fc76a2ce5e03b9a271951003f274435aa00180fcb90e4f2947eedaebb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_groups(features: list[str]) -> dict[str, list[str]]:
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
    if not result["magnetic"] or not result["eruption"]:
        raise ValueError("magnetic and eruption predictors are required")
    return result


def _fit_transform(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    groups: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    receipt: dict[str, Any] = {"fit_role": "train", "modalities": {}}
    for modality, columns in groups.items():
        if not columns:
            values[modality] = np.zeros((len(frame), 1), dtype=np.float32)
            masks[modality] = np.zeros((len(frame), 1), dtype=bool)
            receipt["modalities"][modality] = {
                "columns": [],
                "placeholder_width": 1,
                "always_unavailable": True,
            }
            continue
        raw = frame[columns].to_numpy(dtype=np.float64)
        observed = np.isfinite(raw)
        train_raw = raw[train_mask]
        median = np.nanmedian(train_raw, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(observed, raw, median)
        train_filled = filled[train_mask]
        mean = train_filled.mean(axis=0)
        scale = train_filled.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        transformed = (filled - mean) / scale
        if not np.isfinite(transformed).all():
            raise ValueError(f"nonfinite transformed values in {modality}")
        values[modality] = transformed.astype(np.float32)
        masks[modality] = observed
        receipt["modalities"][modality] = {
            "columns": columns,
            "always_unavailable": False,
            "median": median.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }
    return values, masks, receipt


class _Rows(Dataset):
    def __init__(self, indices: np.ndarray, values: dict[str, np.ndarray], masks: dict[str, np.ndarray], labels: np.ndarray) -> None:
        self.indices = indices
        self.values = values
        self.masks = masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor]:
        index = int(self.indices[position])
        return (
            {name: torch.from_numpy(self.values[name][index]) for name in MODALITIES},
            {name: torch.from_numpy(self.masks[name][index]) for name in MODALITIES},
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.int64),
        )


def _inputs(batch_values: dict[str, Tensor], batch_masks: dict[str, Tensor], device: torch.device) -> dict[str, BranchInput]:
    return {
        name: BranchInput(batch_values[name].to(device), batch_masks[name].to(device))
        for name in MODALITIES
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def _predict(model: IRISSEPTabularModel, loader: DataLoader, device: torch.device, total_rows: int) -> np.ndarray:
    model.eval()
    logits = np.full(total_rows, np.nan, dtype=np.float64)
    for batch_values, batch_masks, _, indices in loader:
        output = model(_inputs(batch_values, batch_masks, device), apply_missing_modality_dropout=False)
        logits[indices.numpy()] = output.primary_logit.detach().cpu().numpy()
    return logits


def _save_checkpoint(
    path: Path,
    *,
    model: IRISSEPTabularModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: GradScaler,
    epoch: int,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "rng": _rng_state(),
    }, temporary)
    os.replace(temporary, path)


def train(source: Path, source_manifest: Path, output_dir: Path, *, max_epochs: int, patience: int, batch_size: int, resume: bool) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise ValueError("output directory exists; pass --resume only for the same interrupted run")
    if sha256_file(source) != EXPECTED_SOURCE_SHA256 or sha256_file(source_manifest) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("trainer accepts only the pinned development-v3 cohort and manifest")
    source_manifest_payload = json.loads(source_manifest.read_text())
    if source_manifest_payload.get("output_sha256") != EXPECTED_SOURCE_SHA256 or source_manifest_payload.get("locked_test_rows_present") is not False:
        raise ValueError("source manifest does not bind a non-locked cohort")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if not META.issubset(frame.columns) or set(frame["role"].unique()) != set(ROLES):
        raise ValueError("development role contract failure")
    if frame["role"].str.lower().str.contains("test|sepval|locked").any():
        raise ValueError("test-like roles are forbidden in this trainer")
    features = [name for name in frame.columns if name not in META]
    if any(name.lower().startswith("future_") for name in features):
        raise ValueError("future outcome leaked into predictors")
    groups = _feature_groups(features)
    labels = frame[TARGET].to_numpy(dtype=np.int64)
    train_mask = frame["role"].eq("train").to_numpy()
    values, masks, preprocessing = _fit_transform(frame, train_mask, groups)
    preprocessing_path = output_dir / "preprocessing.json"
    if not preprocessing_path.exists():
        preprocessing_path.write_text(json.dumps(preprocessing, sort_keys=True, indent=2) + "\n")
    elif json.loads(preprocessing_path.read_text()) != preprocessing:
        raise ValueError("resume preprocessing does not match current source")

    indices = {role: np.flatnonzero(frame["role"].eq(role).to_numpy()) for role in ROLES}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        role: DataLoader(
            _Rows(role_indices, values, masks, labels),
            batch_size=batch_size,
            shuffle=(role == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        for role, role_indices in indices.items()
    }
    config = TabularModelConfig(
        magnetic_features=values["magnetic"].shape[1],
        eruption_features=values["eruption"].shape[1],
        particle_context_features=values["particle_context"].shape[1],
        magnetic_feature_names=tuple(groups["magnetic"]),
        eruption_feature_names=tuple(groups["eruption"]),
        particle_context_feature_names=tuple(groups["particle_context"] or ["__PARTICLE_CONTEXT_UNAVAILABLE_IN_V1__"]),
    )
    if not config.schema_bound or config.feature_schema_sha256 is None:
        raise ValueError("real training requires an ordered, fully bound feature schema")
    pos_weight = float(np.sum(labels[train_mask] == 0) / np.sum(labels[train_mask] == 1))
    logits_by_seed: list[np.ndarray] = []
    seed_receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        _seed_everything(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        model = IRISSEPTabularModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        scaler = GradScaler(device.type, enabled=device.type == "cuda")
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        checkpoint_path = seed_dir / "last.pt"
        best_path = seed_dir / "best.pt"
        start_epoch = 0
        best_score = -np.inf
        best_epoch = -1
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        if resume and checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state["best_score"])
            best_epoch = int(state["best_epoch"])
            stale_epochs = int(state["stale_epochs"])
            _restore_rng(state["rng"])
            history_path = seed_dir / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text())
        for epoch in range(start_epoch, max_epochs):
            model.train()
            losses = []
            for batch_values, batch_masks, batch_labels, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                batch_labels = batch_labels.to(device)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(_inputs(batch_values, batch_masks, device))
                    loss = loss_function(output.primary_logit, batch_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            monitor_logits = _predict(model, loaders["validation_monitor"], device, len(frame))[indices["validation_monitor"]]
            monitor_score = probability_metrics(
                labels[indices["validation_monitor"]], sigmoid(monitor_logits),
                reference_probability=float(labels[train_mask].mean()),
            )["AUPRC"]
            scheduler.step(monitor_score)
            improved = monitor_score > best_score + 1e-6
            if improved:
                best_score = monitor_score
                best_epoch = epoch
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "monitor_AUPRC": monitor_score}, best_path)
            else:
                stale_epochs += 1
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "monitor_AUPRC": monitor_score})
            (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            _save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_score=best_score, best_epoch=best_epoch, stale_epochs=stale_epochs,
            )
            if stale_epochs >= patience:
                break
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        all_loader = DataLoader(_Rows(np.arange(len(frame)), values, masks, labels), batch_size=batch_size, shuffle=False)
        seed_logits = _predict(model, all_loader, device, len(frame))
        logits_by_seed.append(seed_logits)
        seed_receipts.append({
            "seed": seed,
            "best_epoch": int(best_state["epoch"]),
            "monitor_AUPRC": float(best_state["monitor_AUPRC"]),
            "best_checkpoint": str(best_path.relative_to(output_dir)),
            "best_checkpoint_sha256": sha256_file(best_path),
        })

    calibration_indices = indices["validation_calibration"]
    calibrations = [
        fit_intercept_calibration(
            seed_logits[calibration_indices], labels[calibration_indices], role="validation_calibration"
        )
        for seed_logits in logits_by_seed
    ]
    seed_probabilities = np.stack([
        apply_calibration(seed_logits, calibration)
        for seed_logits, calibration in zip(logits_by_seed, calibrations)
    ])
    probabilities = np.median(seed_probabilities, axis=0)
    threshold_indices = indices["validation_threshold"]
    threshold = select_tss_threshold(
        labels[threshold_indices], probabilities[threshold_indices], role="validation_threshold"
    )
    prevalence = float(labels[train_mask].mean())
    metrics = {
        role: {
            **threshold_metrics(labels[role_indices], probabilities[role_indices], threshold.threshold),
            **probability_metrics(labels[role_indices], probabilities[role_indices], reference_probability=prevalence),
        }
        for role, role_indices in indices.items()
    }
    predictions = frame[["issue_id", "role", "unit_id"]].copy()
    predictions["label"] = labels
    predictions["calibrated_probability"] = probabilities
    predictions["between_seed_probability_std"] = np.std(seed_probabilities, axis=0)
    prediction_path = output_dir / "development_predictions.csv"
    predictions.to_csv(prediction_path, index=False, float_format="%.17g")
    saved_predictions = pd.read_csv(prediction_path, float_precision="round_trip")
    for role, role_indices in indices.items():
        saved_role = saved_predictions.loc[saved_predictions["role"] == role]
        metrics[role] = {
            **threshold_metrics(saved_role["label"], saved_role["calibrated_probability"], threshold.threshold),
            **probability_metrics(saved_role["label"], saved_role["calibrated_probability"], reference_probability=prevalence),
        }
    receipt = {
        "status": "PASS_DEVELOPMENT_ONLY",
        "source_sha256": sha256_file(source),
        "source_manifest_sha256": sha256_file(source_manifest),
        "target_semantics": "PUBLISHER_LEGACY_WINDOW_LABEL_NOT_FINAL_NEW_CROSSING_TARGET",
        "locked_test_accessed": False,
        "device": str(device),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
        "feature_groups": {name: columns for name, columns in groups.items()},
        "feature_schema_sha256": config.feature_schema_sha256,
        "particle_context_absent_from_v1": not bool(groups["particle_context"]),
        "config": vars(config),
        "train_positive_weight": pos_weight,
        "selection_metric": "validation_monitor_AUPRC",
        "seeds": seed_receipts,
        "ensemble": "MEDIAN_OF_FIVE_SEED_CALIBRATED_PROBABILITIES",
        "calibration": [
            {"seed": seed, "method": "LOGIT_INTERCEPT_ONLY", "fit_role": calibration.fit_role, "id": calibration.calibration_id, "intercept": calibration.intercept}
            for seed, calibration in zip(SEEDS, calibrations)
        ],
        "threshold": {"objective": threshold.objective, "fit_role": threshold.fit_role, "id": threshold.threshold_id, "value": threshold.threshold},
        "metrics": metrics,
        "role_prevalence": {role: float(labels[role_indices].mean()) for role, role_indices in indices.items()},
        "selection_side_metrics": ["validation_monitor", "validation_calibration", "validation_threshold"],
        "headline_eligible_roles": [],
        "training_limits": {"max_epochs": max_epochs, "patience": patience, "batch_size": batch_size},
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "predictions_sha256": sha256_file(prediction_path),
        "source_hashes": {
            "trainer": sha256_file(Path(__file__)),
            "model": sha256_file(IRIS_SEP_ROOT / "src" / "iris_sep" / "modeling" / "tabular_multibranch.py"),
            "evaluation": sha256_file(IRIS_SEP_ROOT / "workstreams" / "luna_i_eval_ops" / "evaluation.py"),
            "benchmark_contract": sha256_file(IRIS_SEP_ROOT / "config" / "benchmark_contract_v2.json"),
            "evaluation_policy": sha256_file(IRIS_SEP_ROOT / "config" / "evaluation_policy_v1.json")
        },
        "claims_forbidden": ["SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION"],
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("training limits must be positive")
    receipt = train(args.source, args.source_manifest, args.output_dir, max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size, resume=args.resume)
    print(json.dumps({"status": receipt["status"], "device": receipt["device"], "locked_test_accessed": receipt["locked_test_accessed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
