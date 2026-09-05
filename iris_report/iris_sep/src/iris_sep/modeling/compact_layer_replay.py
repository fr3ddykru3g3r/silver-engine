"""Controlled layer-by-layer replay for the compact tabular model.

This module is diagnostic only. It reports the first tensor stage containing a
nonfinite value and audits train-fitted scaling/missingness. It does not infer a
scientific cause without the exact checkpoint and exact failing fold arrays.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from iris_report.iris_sep.src.iris_sep.modeling.tabular_multibranch import BranchInput, IRISSEPTabularModel, MODALITIES


def _summary(value: Tensor) -> dict[str, Any]:
    detached = value.detach().cpu()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = int(detached.numel())
    if finite_count:
        max_abs = float(detached[finite].abs().max().item())
    else:
        max_abs = None
    if detached.ndim == 0:
        bad_rows = [0] if not bool(finite.item()) else []
    else:
        row_finite = finite.reshape(detached.shape[0], -1).all(dim=1)
        bad_rows = torch.nonzero(~row_finite, as_tuple=False).flatten().tolist()[:20]
    return {"shape": list(detached.shape), "finite": finite_count, "total": total, "max_abs_finite": max_abs, "first_nonfinite_rows": bad_rows}


def transform_from_preprocessing(raw_features: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Apply the saved train-fitted transform and audit float32 cast behavior."""
    if preprocessing.get("fit_role") != "train" or not isinstance(preprocessing.get("modalities"), Mapping):
        raise ValueError("train-fitted preprocessing receipt required")
    row_count = None
    for value in raw_features.values():
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError("raw feature arrays must be two-dimensional")
        row_count = len(array) if row_count is None else row_count
        if len(array) != row_count:
            raise ValueError("all modalities must share row count")
    if row_count is None:
        raise ValueError("raw feature arrays required")
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {"fit_role": "train", "modalities": {}}
    for modality in MODALITIES:
        meta = preprocessing["modalities"].get(modality)
        if not isinstance(meta, Mapping):
            raise ValueError(f"missing preprocessing for {modality}")
        if meta.get("always_unavailable") is True:
            width = int(meta.get("placeholder_width", 1))
            values[modality] = np.zeros((row_count, width), dtype=np.float32)
            masks[modality] = np.zeros((row_count, width), dtype=bool)
            audit["modalities"][modality] = {"always_unavailable": True, "rows": row_count, "features": width, "observed_per_feature": [0] * width, "pre_cast_nonfinite": 0, "post_cast_nonfinite": 0, "max_abs_float64": 0.0, "max_abs_float32": 0.0}
            continue
        raw = np.asarray(raw_features.get(modality), dtype=np.float64)
        columns = list(meta.get("columns", ()))
        if raw.ndim != 2 or raw.shape[1] != len(columns):
            raise ValueError(f"raw {modality} width does not match preprocessing columns")
        median = np.asarray(meta.get("median"), dtype=np.float64)
        mean = np.asarray(meta.get("mean"), dtype=np.float64)
        scale = np.asarray(meta.get("scale"), dtype=np.float64)
        if median.shape != (raw.shape[1],) or mean.shape != median.shape or scale.shape != median.shape:
            raise ValueError(f"invalid preprocessing vector sizes for {modality}")
        if not np.isfinite(median).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"invalid train-fitted scaling for {modality}")
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, median)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed64 = (filled - mean) / scale
        pre_nonfinite = int((~np.isfinite(transformed64)).sum())
        finite64 = transformed64[np.isfinite(transformed64)]
        max64 = float(np.max(np.abs(finite64))) if finite64.size else None
        with np.errstate(over="ignore", invalid="ignore"):
            transformed32 = transformed64.astype(np.float32)
        post_nonfinite = int((~np.isfinite(transformed32)).sum())
        finite32 = transformed32[np.isfinite(transformed32)]
        max32 = float(np.max(np.abs(finite32))) if finite32.size else None
        values[modality] = transformed32
        masks[modality] = observed
        audit["modalities"][modality] = {
            "always_unavailable": False,
            "rows": row_count,
            "features": raw.shape[1],
            "columns": columns,
            "observed_per_feature": observed.sum(axis=0).astype(int).tolist(),
            "missing_per_feature": (~observed).sum(axis=0).astype(int).tolist(),
            "minimum_train_scale": float(scale.min()),
            "maximum_train_scale": float(scale.max()),
            "pre_cast_nonfinite": pre_nonfinite,
            "post_cast_nonfinite": post_nonfinite,
            "max_abs_float64": max64,
            "max_abs_float32": max32,
        }
    return values, masks, audit


@torch.no_grad()
def replay_model_layers(model: IRISSEPTabularModel, inputs: Mapping[str, BranchInput]) -> dict[str, Any]:
    """Replay the exact inference graph with dropout disabled and stage summaries."""
    if set(inputs) != set(MODALITIES):
        raise ValueError(f"inputs must contain exactly {MODALITIES}")
    was_training = model.training
    model.eval()
    stages: list[tuple[str, Tensor]] = []
    branch_outputs: list[Tensor] = []
    availability: list[Tensor] = []
    feature_support: dict[str, Any] = {}
    try:
        for modality in MODALITIES:
            branch = model.branches[modality]
            parameter = next(branch.parameters())
            values = inputs[modality].values.to(device=parameter.device, dtype=parameter.dtype)
            mask = inputs[modality].observed_mask.to(device=parameter.device)
            if mask.dtype != torch.bool:
                if mask.is_complex() or not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
                    raise ValueError("feature mask must be boolean or exact binary")
            mask = mask.bool()
            if values.ndim != 2 or values.shape[1] != branch.features or mask.shape != values.shape:
                raise ValueError(f"invalid {modality} replay shape")
            if not torch.isfinite(values).all():
                raise ValueError("replay inputs must be finite; inspect transform audit first")
            feature_support[modality] = {
                "rows": int(values.shape[0]),
                "features": int(values.shape[1]),
                "observed_per_feature": mask.sum(dim=0).detach().cpu().to(torch.int64).tolist(),
                "all_missing_rows": int((~mask.any(dim=1)).sum().item()),
                "max_abs_standardized_feature": float(values.abs().max().item()) if values.numel() else None,
            }
            current = torch.cat((values * mask.to(values.dtype), mask.to(values.dtype)), dim=1)
            stages.append((f"{modality}.prepared", current))
            for index, layer in enumerate(branch.network):
                current = layer(current)
                stages.append((f"{modality}.branch.{index}.{layer.__class__.__name__}", current))
            branch_outputs.append(current)
            availability.append(mask.any(dim=1))

        stacked = torch.stack(branch_outputs, dim=1)
        available = torch.stack(availability, dim=1)
        stages.append(("fusion.stacked", stacked))
        effective = available
        masked = stacked * effective.to(stacked.dtype).unsqueeze(-1)
        stages.append(("fusion.masked", masked))
        gate_input = torch.cat((masked.flatten(start_dim=1), effective.to(stacked.dtype)), dim=1)
        stages.append(("gate.input", gate_input))
        gate_current = gate_input
        for index, layer in enumerate(model.gate):
            gate_current = layer(gate_current)
            stages.append((f"gate.{index}.{layer.__class__.__name__}", gate_current))
        masked_gate_logits = gate_current.masked_fill(~effective, -1e4)
        stages.append(("gate.masked_logits", masked_gate_logits))
        weights = torch.softmax(masked_gate_logits, dim=1) * effective.to(stacked.dtype)
        stages.append(("gate.weights_pre_normalize", weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        stages.append(("gate.weights", weights))
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        stages.append(("fusion.fused", fused))
        shared_current = torch.cat((fused, effective.to(fused.dtype)), dim=1)
        stages.append(("shared.input", shared_current))
        for index, layer in enumerate(model.shared):
            shared_current = layer(shared_current)
            stages.append((f"shared.{index}.{layer.__class__.__name__}", shared_current))
        primary = model.primary_head(shared_current).squeeze(-1)
        stages.append(("primary_head.logit", primary))

        direct = model(inputs, apply_missing_modality_dropout=False)
        same = torch.allclose(primary, direct.primary_logit, rtol=0.0, atol=0.0, equal_nan=True)
        stage_payload = [{"stage": name, **_summary(value)} for name, value in stages]
        first = next((entry["stage"] for entry in stage_payload if entry["finite"] != entry["total"]), None)
        return {
            "status": "NONFINITE_REPRODUCED" if first is not None else "FINITE_REPLAY",
            "first_nonfinite_stage": first,
            "feature_support": feature_support,
            "all_missing_rows": int((~effective.any(dim=1)).sum().item()),
            "forward_replay_exact": bool(same),
            "stages": stage_payload,
        }
    finally:
        model.train(was_training)
