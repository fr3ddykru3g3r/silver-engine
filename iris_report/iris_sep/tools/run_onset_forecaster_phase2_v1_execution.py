"""Execution adapter for the preregistered Phase II onset forecaster.

This file contains runtime-compatibility corrections only. The first correction
resolved a pre-execution XRS feature-name alias mismatch. The second installs
the already-reviewed pandas timezone/index compatibility semantics used by the
frozen episode benchmark after the first Phase II execution attempt terminated
with zero modelable rows before fitting or scoring any Phase II model.

The timezone conversion is cached once per source DataFrame. This changes only
runtime cost: the cached nanosecond timestamp array is byte-for-byte the same
array that the frozen compatibility helper would reconstruct on every call.

Neither correction changes the preregistered model set, splits, target,
hyperparameters, calibration, blend grid, threshold policy, success gate, or
protected-outcome boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import tools.run_episode_normalized_development_benchmark_v1_compat as benchmark_compat
import tools.run_freshness_crossover_study_v1 as fresh
import tools.run_onset_forecaster_phase2_v1 as phase2


_TIME_NS_CACHE: dict[int, tuple[int, np.ndarray]] = {}


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"none of the required feature aliases exist: {candidates}")


def _nanosecond_times(frame: pd.DataFrame) -> np.ndarray:
    key = id(frame)
    cached = _TIME_NS_CACHE.get(key)
    if cached is not None and cached[0] == len(frame):
        return cached[1]
    ns = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    _TIME_NS_CACHE[key] = (len(frame), ns)
    return ns


def family_features_cached(df: pd.DataFrame, issue: pd.Timestamp, delay: int, family: str):
    """Exact frozen compatibility feature windows with cached timestamp conversion."""
    ns = _nanosecond_times(df)
    ins = int(issue.value)
    lo = ins - int(24 * 3600 * 1e9)
    cutoff = min(ins - 1, ins - int(delay * 60 * 1e9))
    a = int(np.searchsorted(ns, lo, side="left"))
    b = int(np.searchsorted(ns, cutoff, side="right"))
    sub = df.iloc[a:b]
    tns = ns[a:b]

    if family == "proton":
        v = sub["proton"].to_numpy(float)
        out = fresh.stream_features(tns, v, ins, 288, "p", True)
        good = v[np.isfinite(v)]
        for q in fresh.P_THRESH:
            out[f"p_count_ge_{q:g}"] = float(np.sum(good >= q))
        return out

    if family != "xrs":
        raise ValueError(f"unknown family: {family}")
    out = fresh.stream_features(tns, sub["A"].to_numpy(float), ins, 1440, "a")
    out.update(fresh.stream_features(tns, sub["B"].to_numpy(float), ins, 1440, "b"))
    good = sub["B"].to_numpy(float)
    good = good[np.isfinite(good)]
    for q in fresh.XRS_THRESH:
        out[f"b_count_ge_{q:.0e}"] = float(np.sum(good >= q))
    return out


def corrected_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the frozen engineered feature family with robust source aliases."""
    out = frame.copy()
    created: list[str] = []

    def add(name: str, values: Iterable[float] | pd.Series) -> None:
        arr = np.asarray(values, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        out[name] = arr
        created.append(name)

    ratio = phase2.safe_ratio
    add("eng_log1p_p_last", np.log1p(np.maximum(pd.to_numeric(out["p_last"], errors="coerce"), 0)))
    add("eng_log1p_p_max", np.log1p(np.maximum(pd.to_numeric(out["p_max"], errors="coerce"), 0)))
    add("eng_p_last_over_mean", ratio(out["p_last"], out["p_mean"]))
    add("eng_p_max_over_mean", ratio(out["p_max"], out["p_mean"]))
    add("eng_p_last_minus_median", pd.to_numeric(out["p_last"], errors="coerce") - pd.to_numeric(out["p_median"], errors="coerce"))
    add("eng_p_ge5_fraction", ratio(out["p_count_ge_5"], out["p_n"]))
    add("eng_p_ge8_fraction", ratio(out["p_count_ge_8"], out["p_n"]))
    add("eng_b_last_over_mean", ratio(out["b_last"], out["b_mean"]))
    add("eng_b_max_over_mean", ratio(out["b_max"], out["b_mean"]))
    add("eng_a_last_over_mean", ratio(out["a_last"], out["a_mean"]))
    add("eng_b_over_a_last", ratio(out["b_last"], out["a_last"]))
    add("eng_b_over_a_max", ratio(out["b_max"], out["a_max"]))
    add("eng_b_ge1e5_fraction", ratio(_column(out, "b_count_ge_1e-05", "b_count_ge_1e-5"), out["b_n"]))
    add("eng_b_ge1e4_fraction", ratio(_column(out, "b_count_ge_1e-04", "b_count_ge_1e-4", "b_count_ge_0.0001"), out["b_n"]))
    add(
        "eng_joint_impulsiveness",
        pd.to_numeric(out["b_max_pos_log_change"], errors="coerce")
        * (1.0 + pd.to_numeric(out["p_max_pos_log_change"], errors="coerce")),
    )
    return out, created


def install_runtime_compatibility() -> None:
    """Install representation-only compatibility fixes before acquisition."""
    benchmark_compat.install_compatibility_patch()
    fresh.family_features = family_features_cached
    phase2.add_engineered_features = corrected_engineered_features


def run(output: Path) -> dict:
    install_runtime_compatibility()
    return phase2.run(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
