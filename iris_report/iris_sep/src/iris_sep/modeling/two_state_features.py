"""Causal two-state feature transform for daily IRIS-SEP specialist models.

For each issue row the transform exposes only:

- the current 24 h aggregate state;
- the nearest state ending 24 h earlier (within a fixed tolerance and strictly
  earlier than the issue state);
- the current-minus-previous change where both values are finite;
- two low-dimensional provenance features describing lag availability/age.

It is intentionally not a sequence model. The transform gives a tabular expert
one explicit state transition without pretending aggregate min/max/mean columns
are temporal tokens.

Timestamp arithmetic deliberately uses pandas ``Timestamp``/``Timedelta``
objects instead of the integer storage representation. Pandas 2.x commonly
stores parsed UTC timestamps at nanosecond resolution while pandas 3.x may
preserve microsecond resolution; comparing raw ``astype('int64')`` values to
``Timedelta.value`` therefore creates a silent 1000x unit mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TwoStateConfig:
    lag_hours: float = 24.0
    tolerance_hours: float = 3.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lag_hours) or self.lag_hours <= 0:
            raise ValueError("lag_hours must be positive and finite")
        if not np.isfinite(self.tolerance_hours) or self.tolerance_hours < 0:
            raise ValueError("tolerance_hours must be finite and nonnegative")


def build_two_state_features(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    *,
    time_column: str = "window_end",
    config: TwoStateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return causal current/lag/delta features aligned one-to-one with ``frame``."""
    cfg = config or TwoStateConfig()
    names = list(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be nonempty and unique")
    missing = [name for name in [time_column, *names] if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    times = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("issue times must be unique and monotonically increasing")
    time_index = pd.DatetimeIndex(times)

    current = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n, width = current.shape
    lagged = np.full((n, width), np.nan, dtype=np.float64)
    delta = np.full((n, width), np.nan, dtype=np.float64)
    available = np.zeros(n, dtype=np.float64)
    age_hours = np.full(n, np.nan, dtype=np.float64)
    lag_index = np.full(n, -1, dtype=np.int64)

    lag = pd.Timedelta(hours=float(cfg.lag_hours))
    tolerance = pd.Timedelta(hours=float(cfg.tolerance_hours))
    one_hour = pd.Timedelta(hours=1)

    for i, issue_time in enumerate(time_index):
        target = issue_time - lag
        pos = int(time_index.searchsorted(target, side="left"))
        best = -1
        best_distance: pd.Timedelta | None = None
        for candidate in (pos - 1, pos):
            if 0 <= candidate < i:  # strict causality: lag row must be earlier
                distance = abs(time_index[candidate] - target)
                if distance <= tolerance and (best_distance is None or distance < best_distance):
                    best = int(candidate)
                    best_distance = distance
        if best < 0:
            continue

        lag_index[i] = best
        lagged[i] = current[best]
        finite_pair = np.isfinite(current[i]) & np.isfinite(current[best])
        delta[i, finite_pair] = current[i, finite_pair] - current[best, finite_pair]
        available[i] = 1.0
        age_hours[i] = float((issue_time - time_index[best]) / one_hour)

    columns: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        columns[f"cur__{name}"] = current[:, j]
        columns[f"lag24__{name}"] = lagged[:, j]
        columns[f"delta24__{name}"] = delta[:, j]
    columns["state__lag24_available"] = available
    columns["state__lag_age_hours"] = age_hours
    result = pd.DataFrame(columns, index=frame.index)

    receipt = {
        "lag_hours": float(cfg.lag_hours),
        "tolerance_hours": float(cfg.tolerance_hours),
        "source_feature_count": int(width),
        "output_feature_count": int(result.shape[1]),
        "rows": int(n),
        "lag_available_rows": int(np.sum(available == 1.0)),
        "lag_missing_rows": int(np.sum(available == 0.0)),
        "strictly_earlier_lag_rows": bool(np.all((lag_index < 0) | (lag_index < np.arange(n)))),
        "exact_24h_lag_rows": int(np.sum(np.isclose(age_hours, cfg.lag_hours, equal_nan=False))),
        "timestamp_storage_unit_independent": true,
    }
    return result, receipt
