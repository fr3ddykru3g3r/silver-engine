"""Simple causal magnetic-map reconstruction for IRIS-SEP.

This is deliberately a *reduced* surface-transport candidate, not a full MHD
solver. It evolves the last observed 2-D radial-field-like map through
longitudinal advection plus diffusive spreading,

    dB/dt + u dB/dx = kappa * Laplacian(B)

on a regular map grid. The source geometry supplies the rotation/drift rate and
map scale; no later observation is consulted. The model is intended only for
held-out transient-gap experiments until real-data validation earns broader use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RotateSpreadConfig:
    """Parameters for the reduced rotate-and-spread reconstruction.

    ``longitude_degrees_per_pixel`` and ``rotation_degrees_per_day`` make the
    advection rate explicit instead of silently assuming a coordinate system.
    ``diffusion_pixels2_per_day`` is a grid-scale transport parameter that must
    later be fixed from source geometry / the train-only reconstruction study;
    it is not claimed to be a universal solar diffusivity.
    """

    longitude_degrees_per_pixel: float
    rotation_degrees_per_day: float
    diffusion_pixels2_per_day: float = 0.0
    max_substep_hours: float = 1.0
    validated_horizon_hours: float = 24.0

    def __post_init__(self) -> None:
        finite_positive = {
            "longitude_degrees_per_pixel": self.longitude_degrees_per_pixel,
            "max_substep_hours": self.max_substep_hours,
            "validated_horizon_hours": self.validated_horizon_hours,
        }
        for name, value in finite_positive.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            not isinstance(self.rotation_degrees_per_day, (int, float))
            or isinstance(self.rotation_degrees_per_day, bool)
            or not math.isfinite(self.rotation_degrees_per_day)
        ):
            raise ValueError("rotation_degrees_per_day must be finite")
        if (
            not isinstance(self.diffusion_pixels2_per_day, (int, float))
            or isinstance(self.diffusion_pixels2_per_day, bool)
            or not math.isfinite(self.diffusion_pixels2_per_day)
            or self.diffusion_pixels2_per_day < 0
        ):
            raise ValueError("diffusion_pixels2_per_day must be finite and nonnegative")


@dataclass(frozen=True)
class PhysicsReconstruction:
    field: np.ndarray
    gap_hours: float
    longitudinal_shift_pixels: float
    mean_before: float
    mean_after: float
    uncertainty_proxy: float
    substeps: int
    method_id: str = "ROTATE_SPREAD_2D_V1"
    method_class: str = "PHYSICS_CONSTRAINED"
    constraints: tuple[str, ...] = (
        "last_observation_only",
        "periodic_longitude_advection",
        "diffusive_spreading",
        "grid_mean_preserved",
    )


def _validate_field(field) -> np.ndarray:
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or array.size == 0:
        raise ValueError(
            "last_observed_field must be a finite 2-D map with both dimensions >= 2"
        )
    if not np.isfinite(array).all():
        raise ValueError("last_observed_field must be finite")
    return np.ascontiguousarray(array)


def _fractional_periodic_shift(field: np.ndarray, shift_pixels: float) -> np.ndarray:
    """Shift longitude periodically with deterministic linear interpolation."""
    if not math.isfinite(shift_pixels):
        raise ValueError("shift_pixels must be finite")
    base = math.floor(shift_pixels)
    fraction = shift_pixels - base
    left = np.roll(field, base, axis=1)
    if fraction == 0.0:
        return left.copy()
    right = np.roll(field, base + 1, axis=1)
    return (1.0 - fraction) * left + fraction * right


def _laplacian_neumann_lat_periodic_lon(field: np.ndarray) -> np.ndarray:
    """Unit-grid Laplacian: periodic longitude, zero-gradient latitude edges."""
    west = np.roll(field, 1, axis=1)
    east = np.roll(field, -1, axis=1)
    north = np.empty_like(field)
    south = np.empty_like(field)
    north[0] = field[0]
    north[1:] = field[:-1]
    south[-1] = field[-1]
    south[:-1] = field[1:]
    return north + south + west + east - 4.0 * field


def _required_substeps(gap_days: float, config: RotateSpreadConfig) -> int:
    by_time = max(1, math.ceil(gap_days * 24.0 / config.max_substep_hours))
    # Explicit 2-D diffusion is stable for alpha <= 1/4. Use a 0.20 margin.
    by_diffusion = max(
        1, math.ceil(config.diffusion_pixels2_per_day * gap_days / 0.20)
    )
    return max(by_time, by_diffusion)


def reconstruct_rotate_and_spread(
    last_observed_field,
    *,
    gap_hours: float,
    config: RotateSpreadConfig,
) -> PhysicsReconstruction:
    """Propagate the last real map causally across a temporary observation gap.

    No future map or label is accepted by this API. A zero-hour gap returns the
    observation exactly. For positive gaps, the field is advected in longitude
    and optionally diffused. The plain grid mean is preserved to floating-point
    precision; this is a numerical constraint, not a claim of spherical-flux
    conservation.
    """
    field0 = _validate_field(last_observed_field)
    if (
        not isinstance(gap_hours, (int, float))
        or isinstance(gap_hours, bool)
        or not math.isfinite(gap_hours)
        or gap_hours < 0
    ):
        raise ValueError("gap_hours must be finite and nonnegative")

    gap_hours = float(gap_hours)
    mean0 = float(np.mean(field0))
    total_shift = (
        float(config.rotation_degrees_per_day)
        * (gap_hours / 24.0)
        / float(config.longitude_degrees_per_pixel)
    )
    if gap_hours == 0.0:
        return PhysicsReconstruction(
            field=field0.copy(),
            gap_hours=0.0,
            longitudinal_shift_pixels=0.0,
            mean_before=mean0,
            mean_after=mean0,
            uncertainty_proxy=0.0,
            substeps=0,
        )

    gap_days = gap_hours / 24.0
    substeps = _required_substeps(gap_days, config)
    dt_days = gap_days / substeps
    shift_per_step = (
        float(config.rotation_degrees_per_day)
        * dt_days
        / float(config.longitude_degrees_per_pixel)
    )
    alpha = float(config.diffusion_pixels2_per_day) * dt_days
    if alpha > 0.200000000001:
        raise RuntimeError("internal diffusion substep stability failure")

    current = field0.copy()
    for _ in range(substeps):
        current = _fractional_periodic_shift(current, shift_per_step)
        if alpha > 0.0:
            current = current + alpha * _laplacian_neumann_lat_periodic_lon(current)
        if not np.isfinite(current).all():
            raise FloatingPointError("nonfinite reduced-physics reconstruction")
        # Remove only roundoff drift in the plain grid mean.
        current += mean0 - float(np.mean(current))

    mean_after = float(np.mean(current))
    proxy = min(1.0, gap_hours / float(config.validated_horizon_hours))
    return PhysicsReconstruction(
        field=np.ascontiguousarray(current),
        gap_hours=gap_hours,
        longitudinal_shift_pixels=float(total_shift),
        mean_before=mean0,
        mean_after=mean_after,
        uncertainty_proxy=float(proxy),
        substeps=substeps,
    )
