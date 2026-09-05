"""Bounded train-only neural fit used by the primary IRIS-SEP diagnostic.

This is intentionally a small adapter around the pinned tabular trainer.  It
fits one seed, selects a checkpoint using only ``stop_indices`` and returns
raw logits in the order requested by ``predict_indices``.  Calibration and
threshold selection belong to the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from iris_sep.modeling.tabular_multibranch import (  # noqa: E402
    IRISSEPTabularModel,
    MODALITIES,
    TabularModelConfig,
)
from iris_report.iris_sep.tools.train_tabular_multibranch import (  # noqa: E402
    TARGET,
    _Rows,
    _feature_groups,
    _fit_transform,
    _inputs,
    _predict,
)
from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (  # noqa: E402
    probability_metrics,
    sigmoid,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _indices(value: Sequence[int] | np.ndarray, name: str, size: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer index vector")
    result = result.astype(np.int64, copy=False)
    if len(result) == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(result < 0) or np.any(result >= size):
        raise IndexError(f"{name} contains an out-of-bounds row")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_predict(
    frame: pd.DataFrame,
    features: Sequence[str],
    fit_indices: Sequence[int] | np.ndarray,
    stop_indices: Sequence[int] | np.ndarray,
    predict_indices: Sequence[int] | np.ndarray,
    seed: int,
    output_dir: str | Path,
    *,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one bounded seed and return ``(raw_logits, metadata)``.

    ``frame`` must already be the caller's train-only subset.  Every returned
    logit corresponds positionally to ``predict_indices``; prediction rows
    may overlap the stopping rows.  Preprocessing is fitted strictly on
    ``fit_indices`` and the stopping set is used only for AUPRC selection.
    """
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    if "role" not in frame or TARGET not in frame:
        raise ValueError(f"frame must contain role and {TARGET}")
    if frame["role"].astype(str).ne("train").any():
        raise ValueError("inner neural fit accepts only role=train rows")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("max_epochs, patience, and batch_size must be positive")
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be a non-empty ordered list of unique columns")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError("feature columns missing from frame: " + ", ".join(missing))
    if any(name.lower().startswith("future_") for name in features):
        raise ValueError("future outcome leaked into predictors")

    n_rows = len(frame)
    fit = _indices(fit_indices, "fit_indices", n_rows)
    stop = _indices(stop_indices, "stop_indices", n_rows)
    predict = _indices(predict_indices, "predict_indices", n_rows)
    if np.intersect1d(fit, stop).size:
        raise ValueError("fit_indices and stop_indices must be disjoint")
    labels = frame[TARGET].to_numpy(dtype=np.int64)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("target must be binary")
    if len(np.unique(labels[fit])) < 2:
        raise ValueError("fit_indices must contain both target classes")
    if len(np.unique(labels[stop])) < 2:
        raise ValueError("stop_indices must contain both target classes for AUPRC")

    groups = _feature_groups(features)
    fit_mask = np.zeros(n_rows, dtype=bool)
    fit_mask[fit] = True
    values, masks, preprocessing = _fit_transform(frame, fit_mask, groups)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preprocessing_path = out / "preprocessing.json"
    preprocessing_path.write_text(json.dumps(preprocessing, sort_keys=True, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TabularModelConfig(
        magnetic_features=values["magnetic"].shape[1],
        eruption_features=values["eruption"].shape[1],
        particle_context_features=values["particle_context"].shape[1],
        magnetic_feature_names=tuple(groups["magnetic"]),
        eruption_feature_names=tuple(groups["eruption"]),
        particle_context_feature_names=tuple(groups["particle_context"] or ["__PARTICLE_CONTEXT_UNAVAILABLE_IN_V1__"]),
    )
    _seed_everything(int(seed))
    model = IRISSEPTabularModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    positive = int(np.sum(labels[fit] == 1))
    pos_weight = float(np.sum(labels[fit] == 0) / positive)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    train_loader = DataLoader(_Rows(fit, values, masks, labels), batch_size=batch_size, shuffle=True, num_workers=0)
    stop_loader = DataLoader(_Rows(stop, values, masks, labels), batch_size=batch_size, shuffle=False, num_workers=0)
    best_path = out / "best.pt"
    best_score = -np.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        losses: list[float] = []
        for batch_values, batch_masks, batch_labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(_inputs(batch_values, batch_masks, device)).primary_logit, batch_labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        stop_logits = _predict(model, stop_loader, device, n_rows)[stop]
        score = probability_metrics(labels[stop], sigmoid(stop_logits), reference_probability=float(labels[fit].mean()))["AUPRC"]
        scheduler.step(score)
        improved = score > best_score + 1e-6
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "stop_AUPRC": score}, best_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "stop_AUPRC": float(score)})
        if stale >= patience:
            break
    if not best_path.exists():
        raise RuntimeError("no best checkpoint was produced")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    prediction_loader = DataLoader(_Rows(predict, values, masks, labels), batch_size=batch_size, shuffle=False, num_workers=0)
    recomputed = _predict(model, prediction_loader, device, n_rows)[predict]
    saved_model = IRISSEPTabularModel(config).to(device)
    saved_model.load_state_dict(best_state["model"])
    saved_model.eval()
    verified = _predict(saved_model, prediction_loader, device, n_rows)[predict]
    np.testing.assert_array_equal(recomputed, verified)
    history_path = out / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    metadata = {
        "seed": int(seed), "fit_indices": fit.tolist(), "stop_indices": stop.tolist(),
        "predict_indices": predict.tolist(), "fit_role": "train", "stop_role": "train",
        "selection_metric": "train_subset_stop_AUPRC", "best_epoch": int(best_state["epoch"]),
        "stop_AUPRC": float(best_state["stop_AUPRC"]), "device": str(device),
        "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=5)",
        "pos_weight": pos_weight, "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "preprocessing": str(preprocessing_path), "preprocessing_sha256": _sha256(preprocessing_path),
        "checkpoint_logits_verified_equal": True,
    }
    (out / "receipt.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return recomputed.astype(np.float64, copy=False), metadata
