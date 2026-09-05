"""Bounded development diagnostic for source-era support and late context fusion.

This experiment is preregistered in
config/context_stability_preregistration_2026-09-05.json.  The 2023-2025
monitor has already been inspected and is development-only.  The locked test
is never accessed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark_v2 as v2

EXPECTED_FEATURE_SHA256 = v1.EXPECTED_FEATURE_SHA256
EXPECTED_EVENT_SHA256 = v1.EXPECTED_EVENT_SHA256
TARGET = v1.TARGET
MONITOR_START = pd.Timestamp("2023-07-31T00:00:00Z")
SCOPES = {
    "ALL_HISTORY": None,
    "SDO_HMI_ERA": pd.Timestamp("2010-05-01T00:00:00Z"),
}
ARMS = (
    "BASE_SOLAR",
    "BASE_PLUS_XRS",
    "BASE_PLUS_PROTON",
    "LATE_FUSION_SOLAR_XRS",
    "LATE_FUSION_SOLAR_XRS_PROTON",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value):
    if isinstance(value, dict):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return [finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_frame(features: Path, events: Path):
    if digest(features) != EXPECTED_FEATURE_SHA256:
        raise ValueError("feature-table hash mismatch")
    if digest(events) != EXPECTED_EVENT_SHA256:
        raise ValueError("event-catalogue hash mismatch")
    frame = pd.read_csv(features)
    event_frame = pd.read_csv(events)
    frame["window_begin"] = pd.to_datetime(frame["window_begin"], utc=True, errors="raise")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="raise")
    frame = frame.sort_values(["window_end", "window_begin"]).reset_index(drop=True)
    if frame.duplicated(["window_begin", "window_end"]).any():
        raise ValueError("duplicate issue windows")
    y, active, event_ids, _ = v1.derive_target(frame, event_frame)
    frame = frame.loc[~active].reset_index(drop=True)
    y = np.asarray(y)[~active]
    event_ids = np.asarray(event_ids)[~active]
    if len(frame) != len(y):
        raise ValueError("target/frame alignment failure")
    feature_sets, dropped = v2.feature_sets(frame)
    base = feature_sets["BASE_SOLAR"]
    xrs = [c for c in feature_sets["BASE_PLUS_XRS"] if c not in set(base)]
    proton = [c for c in feature_sets["BASE_PLUS_PROTON"] if c not in set(base)]
    if not xrs or not proton:
        raise ValueError("missing context feature family")
    return frame, y.astype(np.int8), event_ids.astype(str), base, xrs, proton, dropped


def build_scope_roles(frame, y, event_ids, start):
    pre_monitor = frame["window_end"] < MONITOR_START
    if start is not None:
        pre_monitor &= frame["window_end"] >= start
    idx = np.flatnonzero(pre_monitor.to_numpy())
    if len(idx) == 0:
        raise ValueError("empty pre-monitor scope")
    units_sub, roles_sub, purged, positive_units = v2.build_units(
        frame.loc[idx, "window_end"].reset_index(drop=True),
        y[idx],
        event_ids[idx],
    )
    roles = np.full(len(frame), "outside", dtype="U16")
    units = np.full(len(frame), "", dtype="U64")
    roles[idx] = roles_sub
    units[idx] = units_sub

    monitor = frame["window_end"] >= MONITOR_START
    mon_idx = np.flatnonzero(monitor.to_numpy())
    mon_units, _mon_roles, _mon_purged, _ = v2.build_units(
        frame.loc[mon_idx, "window_end"].reset_index(drop=True),
        y[mon_idx],
        event_ids[mon_idx],
    )
    roles[mon_idx] = "monitor"
    units[mon_idx] = mon_units
    return roles, units, purged, positive_units


def fit_xgb_family(frame, names, y, fit_mask):
    predictions = []
    for seed in v1.SEEDS:
        predictions.append(v1.fit_xgb(frame.loc[fit_mask, names], y[fit_mask], frame[names], seed))
    return np.median(np.stack(predictions, axis=0), axis=0)


def calibrate_direct(raw, y, roles):
    cal = roles == "calibration"
    threshold = roles == "threshold"
    intercept = v1.fit_intercept(raw[cal], y[cal])
    p = v1.sigmoid(v1.logit(raw) + intercept)
    t = v1.select_threshold(y[threshold], p[threshold])
    return p, float(t), {"calibration_intercept": float(intercept)}


def fit_late_stack(component_probabilities, y, roles):
    cal = roles == "calibration"
    threshold = roles == "threshold"
    matrix = np.column_stack([v1.logit(p) for p in component_probabilities])
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000)
    model.fit(matrix[cal], y[cal])
    p = model.predict_proba(matrix)[:, 1]
    t = v1.select_threshold(y[threshold], p[threshold])
    return p, float(t), {
        "stack_intercept": float(model.intercept_[0]),
        "stack_coefficients": [float(v) for v in model.coef_[0]],
        "component_count": int(matrix.shape[1]),
    }


def evaluate_role(y, p, threshold, roles, role, reference_probability):
    mask = roles == role
    if int(mask.sum()) == 0 or len(np.unique(y[mask])) != 2:
        raise ValueError(f"role {role} lacks both classes")
    return {
        **v1.threshold_metrics(y[mask], p[mask], threshold),
        **v1.probability_metrics(y[mask], p[mask], reference_probability),
        "matched_detection": {
            str(pod): v1.minimum_far_at_pod(y[mask], p[mask], pod)
            for pod in (0.6, 0.7, 0.8, 0.9)
        },
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
    }


def bootstrap_difference(y, pa, ta, pb, tb, units, roles, role, seed=20260905, replicates=10000):
    mask = roles == role
    y = y[mask]
    pa = pa[mask]
    pb = pb[mask]
    units = units[mask]
    unique = np.unique(units)
    row_map = {u: np.flatnonzero(units == u) for u in unique}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([row_map[u] for u in sampled])
        if len(np.unique(y[ix])) != 2:
            continue
        a = v1.threshold_metrics(y[ix], pa[ix], ta)["TSS"]
        b = v1.threshold_metrics(y[ix], pb[ix], tb)["TSS"]
        diffs.append(float(a - b))
    values = np.asarray(diffs, dtype=float)
    if len(values) == 0:
        raise ValueError("no valid bootstrap replicates")
    return {
        "valid_replicates": int(len(values)),
        "median_TSS_difference": float(np.median(values)),
        "ci_lower_95": float(np.quantile(values, 0.025)),
        "ci_upper_95": float(np.quantile(values, 0.975)),
        "probability_difference_positive": float(np.mean(values > 0)),
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = prepare_frame(features, events)
    all_rows = []
    summary = {
        "status": "COMPLETED_CONTEXT_STABILITY_DEVELOPMENT_DIAGNOSTIC",
        "target": TARGET,
        "locked_test_accessed": False,
        "fixed_monitor_start": MONITOR_START.isoformat(),
        "monitor_is_already_inspected_development_only": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "dropped_non_numeric_columns": dropped,
        "feature_families": {"base": len(base), "xrs_only": len(xrs), "proton_only": len(proton)},
        "scopes": {},
    }
    predictions_by_scope = {}

    for scope_name, start in SCOPES.items():
        roles, units, purged, positive_units = build_scope_roles(frame, y, event_ids, start)
        fit_mask = roles == "fit"
        if len(np.unique(y[fit_mask])) != 2:
            raise ValueError(f"{scope_name} fit lacks both classes")
        prevalence = float(np.mean(y[fit_mask]))

        raw_base = fit_xgb_family(frame, base, y, fit_mask)
        raw_xrs_concat = fit_xgb_family(frame, base + xrs, y, fit_mask)
        raw_proton_concat = fit_xgb_family(frame, base + proton, y, fit_mask)
        raw_xrs_only = fit_xgb_family(frame, xrs, y, fit_mask)
        raw_proton_only = fit_xgb_family(frame, proton, y, fit_mask)

        probability = {}
        thresholds = {}
        metadata = {}
        probability["BASE_SOLAR"], thresholds["BASE_SOLAR"], metadata["BASE_SOLAR"] = calibrate_direct(raw_base, y, roles)
        probability["BASE_PLUS_XRS"], thresholds["BASE_PLUS_XRS"], metadata["BASE_PLUS_XRS"] = calibrate_direct(raw_xrs_concat, y, roles)
        probability["BASE_PLUS_PROTON"], thresholds["BASE_PLUS_PROTON"], metadata["BASE_PLUS_PROTON"] = calibrate_direct(raw_proton_concat, y, roles)
        probability["LATE_FUSION_SOLAR_XRS"], thresholds["LATE_FUSION_SOLAR_XRS"], metadata["LATE_FUSION_SOLAR_XRS"] = fit_late_stack([raw_base, raw_xrs_only], y, roles)
        probability["LATE_FUSION_SOLAR_XRS_PROTON"], thresholds["LATE_FUSION_SOLAR_XRS_PROTON"], metadata["LATE_FUSION_SOLAR_XRS_PROTON"] = fit_late_stack([raw_base, raw_xrs_only, raw_proton_only], y, roles)

        scope_result = {
            "start": None if start is None else start.isoformat(),
            "pre_monitor_positive_units": int(positive_units),
            "purged_units": purged,
            "role_support": {},
            "arms": {},
            "paired_comparisons": {},
        }
        for role in ("fit", "calibration", "threshold", "score", "monitor"):
            mask = roles == role
            if mask.any():
                scope_result["role_support"][role] = {
                    "rows": int(mask.sum()),
                    "positives": int(y[mask].sum()),
                    "from": frame.loc[mask, "window_end"].min().isoformat(),
                    "to": frame.loc[mask, "window_end"].max().isoformat(),
                    "units": int(len(np.unique(units[mask]))),
                }
        for arm in ARMS:
            scope_result["arms"][arm] = {
                "threshold": float(thresholds[arm]),
                "model_metadata": metadata[arm],
                "inner_score": evaluate_role(y, probability[arm], thresholds[arm], roles, "score", prevalence),
                "monitor": evaluate_role(y, probability[arm], thresholds[arm], roles, "monitor", prevalence),
            }

        for candidate in ("BASE_PLUS_XRS", "LATE_FUSION_SOLAR_XRS", "LATE_FUSION_SOLAR_XRS_PROTON"):
            scope_result["paired_comparisons"][f"{candidate}_minus_BASE_SOLAR_inner_score"] = bootstrap_difference(
                y, probability[candidate], thresholds[candidate], probability["BASE_SOLAR"], thresholds["BASE_SOLAR"], units, roles, "score"
            )
            scope_result["paired_comparisons"][f"{candidate}_minus_BASE_SOLAR_monitor"] = bootstrap_difference(
                y, probability[candidate], thresholds[candidate], probability["BASE_SOLAR"], thresholds["BASE_SOLAR"], units, roles, "monitor"
            )

        summary["scopes"][scope_name] = scope_result
        predictions_by_scope[scope_name] = (roles, units, probability, thresholds)

        rows = pd.DataFrame({
            "scope": scope_name,
            "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "role": roles,
            "unit_id": units,
            "label": y,
        })
        for arm in ARMS:
            rows[arm] = probability[arm]
        all_rows.append(rows)

    all_roles, all_units, all_probs, all_thresholds = predictions_by_scope["ALL_HISTORY"]
    hmi_roles, hmi_units, hmi_probs, hmi_thresholds = predictions_by_scope["SDO_HMI_ERA"]
    common_monitor = (all_roles == "monitor") & (hmi_roles == "monitor")
    if not np.array_equal(all_units[common_monitor], hmi_units[common_monitor]):
        raise ValueError("monitor unit identities differ across scopes")
    bootstrap_roles = np.where(common_monitor, "monitor", "outside")
    summary["cross_scope_monitor"] = {
        "SDO_HMI_LATE_FUSION_SOLAR_XRS_minus_ALL_HISTORY_BASE_SOLAR": bootstrap_difference(
            y,
            hmi_probs["LATE_FUSION_SOLAR_XRS"], hmi_thresholds["LATE_FUSION_SOLAR_XRS"],
            all_probs["BASE_SOLAR"], all_thresholds["BASE_SOLAR"],
            all_units, bootstrap_roles, "monitor",
        ),
        "SDO_HMI_BASE_PLUS_XRS_minus_ALL_HISTORY_BASE_PLUS_XRS": bootstrap_difference(
            y,
            hmi_probs["BASE_PLUS_XRS"], hmi_thresholds["BASE_PLUS_XRS"],
            all_probs["BASE_PLUS_XRS"], all_thresholds["BASE_PLUS_XRS"],
            all_units, bootstrap_roles, "monitor",
        ),
    }

    predictions = pd.concat(all_rows, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    receipt = {
        "status": "DEVELOPMENT_ONLY_CONTEXT_STABILITY_RUN",
        "preregistration": "config/context_stability_preregistration_2026-09-05.json",
        "target": TARGET,
        "locked_test_accessed": False,
        "monitor_prior_inspection_disclosed": True,
        "model_hyperparameters_changed_after_result": False,
        "source_files_hash_verified": True,
    }
    save_json(output / "receipt.json", receipt)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
