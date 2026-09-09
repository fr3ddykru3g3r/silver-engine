"""Train-only hidden magnetic-map benchmark for the simple physics candidate."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .simple_physics import RotateSpreadConfig, reconstruct_rotate_and_spread


@dataclass(frozen=True)
class MapGapResult:
    issue_index: int
    source_index: int
    gap_hours: float
    persistence_mae: float
    persistence_rmse: float
    physics_mae: float
    physics_rmse: float
    uncertainty_proxy: float


def _errors(truth, prediction) -> tuple[float, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth,
        dtype=np.float64,
    )
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error * error))),
    )


def benchmark_hidden_maps(
    *,
    maps,
    map_observed,
    structural_unavailable,
    issue_time_unix_seconds,
    roles,
    holdout_rows,
    config: RotateSpreadConfig,
) -> dict:
    """Compare last-map persistence with simple physics on deliberately hidden maps.

    Each hidden score-role map is reconstructed from the most recent *earlier*
    real map only. No future observation is available to the reconstructor.
    """
    map_values = np.asarray(maps, dtype=np.float64)
    observed = np.asarray(map_observed, dtype=bool)
    structural = np.asarray(structural_unavailable, dtype=bool)
    times = np.asarray(issue_time_unix_seconds, dtype=np.float64)
    role_values = np.asarray(roles, dtype=str)
    holdout = np.asarray(holdout_rows, dtype=bool)
    if (
        map_values.ndim != 3
        or map_values.shape[0] < 2
        or min(map_values.shape[1:]) < 2
    ):
        raise ValueError(
            "maps must be [rows,height,width] with nontrivial map geometry"
        )
    row_count = map_values.shape[0]
    if any(
        value.shape != (row_count,)
        for value in [observed, structural, times, role_values, holdout]
    ):
        raise ValueError("map metadata arrays must align")
    if np.any(observed & structural):
        raise ValueError("structural map cannot be observed")
    if (
        not np.isfinite(map_values[observed]).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "observed maps and issue times must be finite and chronological"
        )
    if np.any(holdout & (~observed | structural)):
        raise ValueError(
            "holdout rows must be real nonstructural observations"
        )
    if np.any(holdout & (role_values != "score")):
        raise ValueError(
            "artificial map holdouts are restricted to train-only score role"
        )
    if not holdout.any():
        raise ValueError("at least one map must be deliberately hidden")

    experimental_observed = observed & ~holdout
    results: list[MapGapResult] = []
    abstained: list[int] = []
    for issue_index in np.flatnonzero(holdout):
        prior = np.flatnonzero(
            experimental_observed[:issue_index] & ~structural[:issue_index]
        )
        if len(prior) == 0:
            abstained.append(int(issue_index))
            continue
        source_index = int(prior[-1])
        gap_hours = float(
            (times[issue_index] - times[source_index]) / 3600.0
        )
        if gap_hours <= 0 or not math.isfinite(gap_hours):
            raise ValueError("hidden-map gap must be positive")
        truth = map_values[issue_index]
        persistence = map_values[source_index]
        physics = reconstruct_rotate_and_spread(
            map_values[source_index],
            gap_hours=gap_hours,
            config=config,
        )
        persistence_mae, persistence_rmse = _errors(truth, persistence)
        physics_mae, physics_rmse = _errors(truth, physics.field)
        results.append(
            MapGapResult(
                issue_index=int(issue_index),
                source_index=source_index,
                gap_hours=gap_hours,
                persistence_mae=persistence_mae,
                persistence_rmse=persistence_rmse,
                physics_mae=physics_mae,
                physics_rmse=physics_rmse,
                uncertainty_proxy=float(physics.uncertainty_proxy),
            )
        )

    if not results:
        raise ValueError("no hidden map had a causal prior observation")
    return {
        "scope": "TRAIN_ONLY_HIDDEN_MAGNETIC_MAP_DIAGNOSTIC",
        "held_out_maps": int(holdout.sum()),
        "scored_maps": len(results),
        "abstained_no_prior_map": len(abstained),
        "coverage": float(len(results) / holdout.sum()),
        "mean_persistence_mae": float(
            np.mean([row.persistence_mae for row in results])
        ),
        "mean_physics_mae": float(
            np.mean([row.physics_mae for row in results])
        ),
        "mean_persistence_rmse": float(
            np.mean([row.persistence_rmse for row in results])
        ),
        "mean_physics_rmse": float(
            np.mean([row.physics_rmse for row in results])
        ),
        "delta_physics_minus_persistence_mae": float(
            np.mean(
                [
                    row.physics_mae - row.persistence_mae
                    for row in results
                ]
            )
        ),
        "delta_physics_minus_persistence_rmse": float(
            np.mean(
                [
                    row.physics_rmse - row.persistence_rmse
                    for row in results
                ]
            )
        ),
        "rows": [row.__dict__ for row in results],
        "abstained_indices": abstained,
        "claim_boundary": (
            "Pixel-space train-only reconstruction diagnostic only; no "
            "downstream SEP, MHD, operational, superiority or award claim."
        ),
    }
