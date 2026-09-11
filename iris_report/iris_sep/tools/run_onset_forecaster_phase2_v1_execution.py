"""Execution adapter for the preregistered Phase II onset forecaster.

This file exists to correct one pre-execution feature-name mismatch in the
initial implementation. No Phase II real-data score had been executed or
inspected when this adapter was added. It does not change the preregistered
model set, splits, target, hyperparameters, calibration, blend grid, threshold
policy, success gate, or protected-outcome boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import tools.run_onset_forecaster_phase2_v1 as phase2


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"none of the required feature aliases exist: {candidates}")


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


def run(output: Path) -> dict:
    # Pre-execution compatibility correction only. The imported run() resolves
    # this global at execution time, so all scientific logic remains in the
    # preregistered implementation.
    phase2.add_engineered_features = corrected_engineered_features
    return phase2.run(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
