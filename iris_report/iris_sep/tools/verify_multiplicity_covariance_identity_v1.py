"""Exact mechanism identity for row-vs-episode-normalized SEP sensitivity.

For episode i let n_i be the number of mapped positive windows and r_i the
fraction of those windows alerted by a fixed classifier. Then

    POD_row - POD_episode = Cov(n_i, r_i) / E[n_i].

When the negative cohort and its weights are unchanged, FPR is identical in the
two views, so the same identity holds exactly for TSS:

    TSS_row - TSS_episode = Cov(n_i, r_i) / E[n_i].

This script verifies the identity against the immutable fixed-replay artifact.
It does not fit models, change thresholds, or touch protected post-2025 data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def mechanism_for_model(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    use = frame[frame["model"].eq(model)].copy()
    pos = use[use["standard_label"].eq(1)].copy()
    if pos.empty:
        raise RuntimeError(f"no positive rows for {model}")
    if pos["episode_id"].isna().any():
        raise RuntimeError(f"positive mapped rows missing episode_id for {model}")

    episodes = (
        pos.groupby("episode_id", sort=True)["alert"]
        .agg(mapped_windows="size", detected_fraction="mean", detected_windows="sum")
        .reset_index()
    )
    n = episodes["mapped_windows"].to_numpy(float)
    r = episodes["detected_fraction"].to_numpy(float)
    mean_n = float(np.mean(n))
    mean_r = float(np.mean(r))
    mean_nr = float(np.mean(n * r))
    covariance = float(mean_nr - mean_n * mean_r)

    row_pod = float(pos["alert"].mean())
    episode_pod = mean_r
    observed_gap = float(row_pod - episode_pod)
    identity_gap = float(covariance / mean_n)

    neg = use[use["standard_label"].eq(0)]
    fpr = float(neg["alert"].mean())
    row_tss = float(row_pod - fpr)
    episode_tss = float(episode_pod - fpr)

    corr = float(np.corrcoef(n, r)[0, 1]) if np.std(n) > 0 and np.std(r) > 0 else float("nan")
    return {
        "model": model,
        "episode_units": int(len(episodes)),
        "mapped_positive_windows": int(len(pos)),
        "mean_positive_window_multiplicity": mean_n,
        "row_weighted_pod": row_pod,
        "episode_normalized_pod": episode_pod,
        "row_minus_episode_pod": observed_gap,
        "covariance_multiplicity_detection_fraction": covariance,
        "covariance_over_mean_multiplicity": identity_gap,
        "identity_absolute_error": float(abs(observed_gap - identity_gap)),
        "multiplicity_detection_fraction_correlation": corr,
        "negative_fpr_common_to_both_views": fpr,
        "row_weighted_tss": row_tss,
        "episode_normalized_tss": episode_tss,
        "row_minus_episode_tss": float(row_tss - episode_tss),
    }


def run(predictions: Path, point_results: Path | None, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(predictions)
    required = {"model", "alert", "standard_label", "episode_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    models = sorted(frame["model"].dropna().astype(str).unique())
    rows = [mechanism_for_model(frame, model) for model in models]

    point_check: list[dict[str, Any]] = []
    if point_results is not None:
        point = pd.read_csv(point_results)
        for row in rows:
            model = row["model"]
            mapped = point[(point["model"].eq(model)) & (point["view"].eq("MAPPED_STANDARD"))]
            normalized = point[(point["model"].eq(model)) & (point["view"].eq("EPISODE_NORMALIZED_OCCURRENCE"))]
            if len(mapped) != 1 or len(normalized) != 1:
                raise RuntimeError(f"point-result view missing for {model}")
            external_gap = float(mapped.iloc[0]["tss"] - normalized.iloc[0]["tss"])
            identity_gap = float(row["covariance_over_mean_multiplicity"])
            point_check.append({
                "model": model,
                "artifact_mapped_minus_episode_tss": external_gap,
                "identity_gap": identity_gap,
                "absolute_error": abs(external_gap - identity_gap),
            })

    max_identity_error = max(row["identity_absolute_error"] for row in rows)
    max_point_error = max((row["absolute_error"] for row in point_check), default=0.0)
    result = {
        "format": "IRIS_SEP_MULTIPLICITY_COVARIANCE_IDENTITY_V1",
        "status": "PASS" if max(max_identity_error, max_point_error) < 1e-10 else "FAIL",
        "protected_post_2025_outcomes_accessed": False,
        "model_fitting_performed": False,
        "threshold_changes_performed": False,
        "identity": "POD_row - POD_episode = Cov(n_i, r_i) / E[n_i]; with identical negative weighting, the same equality holds for TSS.",
        "models": rows,
        "artifact_point_result_checks": point_check,
        "max_internal_identity_absolute_error": max_identity_error,
        "max_artifact_tss_absolute_error": max_point_error,
        "interpretation": "Positive covariance means episodes represented by more positive windows are also easier for the fixed alert rule, so row-weighted scoring is mechanically higher than episode-normalized scoring. Negative covariance reverses the direction.",
        "claim_boundary": "Exact algebraic mechanism verification on already-frozen historical predictions; not a new independent forecast-performance result."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--point-results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.predictions, args.point_results, args.output), indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
