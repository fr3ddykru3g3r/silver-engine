"""Causal missing-data recovery primitives for IRIS-SEP research.

This module does not impute operational data by itself. It provides two things:
(1) provenance checks that keep reconstructed values distinct from observations,
and (2) deterministic held-out-gap metrics for comparing recovery strategies.

A reconstruction must earn admission in train-only experiments before it can be
considered for a frozen forecast model. Full MHD or any other physics model is
therefore treated as a candidate reconstruction method, not as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import re
from typing import Mapping

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_CLASSES = frozenset({
    "CAUSAL_STATISTICAL",
    "MASKED_MODEL",
    "PHYSICS_CONSTRAINED",
    "PHYSICS_ASSIMILATED",
})


@dataclass(frozen=True)
class ReconstructionProvenance:
    """Evidence needed before a reconstructed modality may enter a forecast.

    ``latest_observation_at`` is the latest real observation used to construct
    the replacement. ``generated_at`` is when the reconstruction was produced.
    Both must be no later than forecast issue time for a causal experiment.
    """

    modality: str
    method_id: str
    method_class: str
    fit_role: str
    latest_observation_at: datetime
    generated_at: datetime
    uses_future_information: bool
    physics_constraints: tuple[str, ...]
    normalized_uncertainty: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.modality or not self.method_id:
            raise ValueError("modality and method_id are required")
        if self.method_class not in _METHOD_CLASSES:
            raise ValueError(f"unsupported method_class: {self.method_class}")
        if self.latest_observation_at.tzinfo is None or self.generated_at.tzinfo is None:
            raise ValueError("reconstruction timestamps must be timezone-aware")
        if self.fit_role != "train":
            raise ValueError("reconstruction method must be fit on train role only")
        if not isinstance(self.uses_future_information, bool):
            raise ValueError("uses_future_information must be boolean")
        if (not isinstance(self.normalized_uncertainty, (int, float)) or
                isinstance(self.normalized_uncertainty, bool) or
                not math.isfinite(self.normalized_uncertainty) or
                not 0.0 <= float(self.normalized_uncertainty) <= 1.0):
            raise ValueError("normalized_uncertainty must be finite in [0,1]")
        if _SHA256.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256")


def audit_forecast_time_reconstruction(
    provenance: ReconstructionProvenance,
    *,
    issued_at: datetime,
    allowed_method_ids: Mapping[str, tuple[str, ...]],
    maximum_uncertainty: float,
    require_declared_physics: bool = False,
) -> tuple[str, ...]:
    """Return deterministic fail-closed reasons for one reconstruction.

    The function intentionally does not decide forecast skill. It only decides
    whether the reconstruction satisfies the causal/provenance boundary needed
    to *enter* a train-only missingness experiment or a later frozen model.
    """

    if issued_at.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    if not math.isfinite(maximum_uncertainty) or not 0 <= maximum_uncertainty <= 1:
        raise ValueError("maximum_uncertainty must be finite in [0,1]")

    reasons: list[str] = []
    if provenance.uses_future_information:
        reasons.append("FUTURE_INFORMATION_USED")
    if provenance.latest_observation_at > issued_at:
        reasons.append("OBSERVATION_AFTER_ISSUE_TIME")
    if provenance.generated_at > issued_at:
        reasons.append("RECONSTRUCTION_GENERATED_AFTER_ISSUE_TIME")
    if provenance.method_id not in allowed_method_ids.get(provenance.modality, ()):
        reasons.append("RECONSTRUCTION_METHOD_NOT_ALLOWED")
    if provenance.normalized_uncertainty > maximum_uncertainty:
        reasons.append("RECONSTRUCTION_UNCERTAINTY_TOO_HIGH")
    if require_declared_physics and provenance.method_class.startswith("PHYSICS_") and not provenance.physics_constraints:
        reasons.append("PHYSICS_CONSTRAINTS_UNDECLARED")
    return tuple(sorted(set(reasons)))


def deterministic_random_gap_mask(shape: tuple[int, ...], missing_fraction: float, *, seed: int) -> np.ndarray:
    """Create a deterministic boolean held-out mask without touching source data."""

    if not shape or any(not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError("shape must contain positive dimensions")
    if not math.isfinite(missing_fraction) or not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be in (0,1)")
    total = int(np.prod(shape))
    missing = max(1, min(total - 1, int(round(total * missing_fraction))))
    rng = np.random.default_rng(seed)
    flat = np.zeros(total, dtype=bool)
    flat[rng.choice(total, size=missing, replace=False)] = True
    return flat.reshape(shape)


def contiguous_gap_mask(shape: tuple[int, ...], *, axis: int, start: int, length: int) -> np.ndarray:
    """Create one contiguous missing block along an axis for outage-style tests."""

    if not shape or any(not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError("shape must contain positive dimensions")
    if not 0 <= axis < len(shape):
        raise ValueError("axis out of bounds")
    if start < 0 or length <= 0 or start + length > shape[axis]:
        raise ValueError("gap block outside requested shape")
    mask = np.zeros(shape, dtype=bool)
    index = [slice(None)] * len(shape)
    index[axis] = slice(start, start + length)
    mask[tuple(index)] = True
    if mask.all():
        raise ValueError("benchmark must retain at least one observed value")
    return mask


def reconstruction_metrics(truth, reconstruction, missing_mask) -> dict[str, float | int]:
    """Evaluate reconstruction only on deliberately hidden values.

    Observed cells never contribute to the score, preventing a method from
    looking good simply because most of the original array was never missing.
    """

    target = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(reconstruction, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=bool)
    if target.shape != predicted.shape or target.shape != mask.shape:
        raise ValueError("truth, reconstruction and missing_mask must have identical shapes")
    if target.size == 0 or not mask.any():
        raise ValueError("at least one deliberately hidden value is required")
    if not np.isfinite(target[mask]).all() or not np.isfinite(predicted[mask]).all():
        raise ValueError("held-out truth and reconstruction must be finite")

    error = predicted[mask] - target[mask]
    absolute = np.abs(error)
    return {
        "held_out_count": int(mask.sum()),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(absolute)),
    }


def interval_coverage_metrics(truth, reconstruction, uncertainty, missing_mask, *, z: float = 1.96) -> dict[str, float | int]:
    """Measure uncertainty coverage on hidden values for reconstruction audits."""

    target = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(reconstruction, dtype=np.float64)
    sigma = np.asarray(uncertainty, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=bool)
    if not (target.shape == predicted.shape == sigma.shape == mask.shape):
        raise ValueError("all reconstruction arrays must have identical shapes")
    if not mask.any() or not math.isfinite(z) or z <= 0:
        raise ValueError("held-out values and positive finite z are required")
    if (not np.isfinite(target[mask]).all() or not np.isfinite(predicted[mask]).all() or
            not np.isfinite(sigma[mask]).all() or np.any(sigma[mask] < 0)):
        raise ValueError("held-out values and uncertainties must be finite; uncertainty must be nonnegative")

    lower = predicted[mask] - z * sigma[mask]
    upper = predicted[mask] + z * sigma[mask]
    covered = (target[mask] >= lower) & (target[mask] <= upper)
    width = upper - lower
    return {
        "held_out_count": int(mask.sum()),
        "interval_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(width)),
    }


def reconstruction_payload_sha256(values) -> str:
    """Hash exact float64 reconstruction bytes for receipt binding."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("finite non-empty reconstruction required")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
