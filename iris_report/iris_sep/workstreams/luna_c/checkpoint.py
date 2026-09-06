"""Restart-safe checkpoint utilities for the IRIS-SEP prototype.

Checkpoints are deliberately self-contained and data agnostic.  They capture
model, optimizer, scheduler, AMP scaler, and Python/NumPy/PyTorch RNG state so
Colab pre-emption can resume a run without silently changing its stochastic
trajectory.  No dataset rows, labels, locked-test values, or predictions are
read or written by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, MutableMapping, Optional

import torch
from torch import nn


CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class ResumeState:
    """Non-tensor resume metadata returned after a successful load."""

    step: int
    epoch: int
    best_metric: Optional[float]
    history: tuple[Mapping[str, Any], ...]
    config: Mapping[str, Any]
    extra: Mapping[str, Any]
    checkpoint_schema: int


def _numpy_module() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    return np


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG streams used by the prototype and common trainers."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    numpy_module = _numpy_module()
    if numpy_module is not None:
        state["numpy"] = numpy_module.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`.

    Older checkpoints may omit optional streams; required PyTorch CPU state is
    still enforced so a partial resume cannot look silently reproducible.
    """

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_module = _numpy_module()
        if numpy_module is None:
            raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is unavailable")
        numpy_module.random.set_state(state["numpy"])
    if "torch_cpu" not in state:
        raise ValueError("checkpoint is missing torch_cpu RNG state")
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor):
        cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8)
    torch.set_rng_state(cpu_state.detach().cpu().to(dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _config_metadata(model: nn.Module, config: Optional[Mapping[str, Any] | Any]) -> dict[str, Any]:
    if config is None:
        config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        values = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise TypeError("config must be a mapping or provide to_dict()")
    # Fail early if a trainer attempts to persist a non-JSON value in metadata.
    json.dumps(values, sort_keys=True)
    return dict(values)


def build_checkpoint_payload(
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    history: Optional[list[Mapping[str, Any]]] = None,
    config: Optional[Mapping[str, Any] | Any] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a complete, serializable resume payload."""

    if step < 0 or epoch < 0:
        raise ValueError("step and epoch must be non-negative")
    if best_metric is not None and not isinstance(best_metric, (int, float)):
        raise TypeError("best_metric must be numeric or None")
    history_values = list(history or [])
    extra_values = dict(extra or {})
    # Metadata is JSON checked, while state_dict tensors remain torch-native.
    json.dumps(history_values, sort_keys=True)
    json.dumps(extra_values, sort_keys=True)
    payload: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "history": history_values,
        "config": _config_metadata(model, config),
        "extra": extra_values,
        "torch_version": torch.__version__,
    }
    return payload


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a checkpoint in the target directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(dict(payload), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    history: Optional[list[Mapping[str, Any]]] = None,
    config: Optional[Mapping[str, Any] | Any] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Build and atomically save a complete resume checkpoint."""

    payload = build_checkpoint_payload(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        epoch=epoch,
        best_metric=best_metric,
        history=history,
        config=config,
        extra=extra,
    )
    return atomic_torch_save(payload, path)


def _torch_load(path: Path, map_location: str | torch.device) -> MutableMapping[str, Any]:
    # ``weights_only=False`` is required because RNG state contains Python and
    # NumPy tuples.  The fallback keeps the helper compatible with older torch.
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, MutableMapping):
        raise ValueError("checkpoint root must be a mapping")
    return loaded


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """Load model/trainer state and optionally restore every RNG stream."""

    checkpoint_path = Path(path)
    checkpoint = _torch_load(checkpoint_path, map_location)
    schema = int(checkpoint.get("checkpoint_schema", 0))
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {schema}")
    if "model" not in checkpoint:
        raise ValueError("checkpoint is missing model state")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint is missing RNG state")
        restore_rng_state(rng_state)

    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        raise ValueError("checkpoint history must be a list")
    config = checkpoint.get("config", {})
    extra = checkpoint.get("extra", {})
    if not isinstance(config, Mapping) or not isinstance(extra, Mapping):
        raise ValueError("checkpoint metadata must be mappings")
    return ResumeState(
        step=int(checkpoint.get("step", 0)),
        epoch=int(checkpoint.get("epoch", 0)),
        best_metric=checkpoint.get("best_metric"),
        history=tuple(history),
        config=dict(config),
        extra=dict(extra),
        checkpoint_schema=schema,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "ResumeState",
    "atomic_torch_save",
    "build_checkpoint_payload",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]
