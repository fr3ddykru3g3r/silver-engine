"""Runtime compatibility wrapper for the frozen episode benchmark V1.

Scientific definitions are unchanged. This wrapper replaces only the legacy
Pandas timezone indexing path used by the reused freshness feature helper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_freshness_crossover_study_v1 as fresh
import tools.run_episode_normalized_development_benchmark_v1 as benchmark


def family_features_compat(df: pd.DataFrame, issue: pd.Timestamp, delay: int, family: str):
    ns = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True)).asi8
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


def install_compatibility_patch() -> None:
    fresh.family_features = family_features_compat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install_compatibility_patch()
    benchmark.run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
