"""Four-fold rolling-origin replication for the bounded IRIS-SEP late-fusion candidate.

The experiment is preregistered in
config/late_fusion_rolling_preregistration_2026-09-05.json. It uses only rows
strictly before the already-inspected 2023-2025 monitor and never touches the
locked test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark_v2 as v2
from iris_report.iris_sep.tools import run_context_stability_diagnostic as ctx

MONITOR_START = pd.Timestamp("2023-07-31T00:00:00Z")
FOLD_COUNT = 4
SCORE_POSITIVES_PER_FOLD = 20
CALIBRATION_POSITIVES = 10
THRESHOLD_POSITIVES = 10
ARMS = (
    "BASE_SOLAR",
    "BASE_PLUS_XRS",
    "LATE_FUSION_SOLAR_XRS",
    "LATE_FUSION_SOLAR_XRS_PROTON",
)


def segment_premonitor_units(frame, y, event_ids):
    mask = frame["window_end"] < MONITOR_START
    idx = np.flatnonzero(mask.to_numpy())
    units, _roles, _purged, _positive_units = v2.build_units(
        frame.loc[idx, "window_end"].reset_index(drop=True), y[idx], event_ids[idx]
    )
    table = pd.DataFrame({
        "local_row": np.arange(len(idx)),
        "global_row": idx,
        "unit_id": units,
        "time": frame.loc[idx, "window_end"].to_numpy(),
        "label": y[idx],
    })
    if (table.groupby("unit_id")["label"].nunique() > 1).any():
        raise ValueError("hardened unit mixes labels")
    summaries = (
        table.groupby("unit_id", sort=False)
        .agg(start=("time", "min"), end=("time", "max"), label=("label", "max"))
        .sort_values(["start", "end"])
    )
    positive_ids = summaries.index[summaries["label"] == 1].tolist()
    minimum = FOLD_COUNT * SCORE_POSITIVES_PER_FOLD + CALIBRATION_POSITIVES + THRESHOLD_POSITIVES + 20
    if len(positive_ids) < minimum:
        raise ValueError(f"insufficient positive units for frozen folds: {len(positive_ids)} < {minimum}")
    return idx, table, summaries, positive_ids


def frame_time_from_roles(roles, table, role):
    selected = [int(g) for g in table["global_row"] if roles[int(g)] == role]
    return table.set_index("global_row").loc[selected, "time"]


def build_fold_roles(frame_length, idx, table, summaries, positive_ids, fold_index):
    first_score_positive = len(positive_ids) - FOLD_COUNT * SCORE_POSITIVES_PER_FOLD + fold_index * SCORE_POSITIVES_PER_FOLD
    calibration_positive = first_score_positive - CALIBRATION_POSITIVES - THRESHOLD_POSITIVES
    threshold_positive = first_score_positive - THRESHOLD_POSITIVES
    after_score_positive = first_score_positive + SCORE_POSITIVES_PER_FOLD

    cal_start = summaries.loc[positive_ids[calibration_positive], "start"]
    threshold_start = summaries.loc[positive_ids[threshold_positive], "start"]
    score_start = summaries.loc[positive_ids[first_score_positive], "start"]
    score_end_boundary = summaries.loc[positive_ids[after_score_positive], "start"] if after_score_positive < len(positive_ids) else MONITOR_START

    unit_roles = {}
    for uid, row in summaries.iterrows():
        start = row["start"]
        if start < cal_start:
            role = "fit"
        elif start < threshold_start:
            role = "calibration"
        elif start < score_start:
            role = "threshold"
        elif start < score_end_boundary:
            role = "score"
        else:
            role = "outside"
        unit_roles[str(uid)] = role

    purged = set()
    for left, right in zip(("fit", "calibration", "threshold"), ("calibration", "threshold", "score")):
        left_units = [u for u, role in unit_roles.items() if role == left and u not in purged]
        if not left_units:
            raise ValueError(f"fold {fold_index}: empty {left} before purge")
        left_end = summaries.loc[left_units, "end"].max()
        for uid in [u for u, role in unit_roles.items() if role == right]:
            if summaries.loc[uid, "start"] <= left_end + pd.Timedelta(hours=v1.PURGE_HOURS):
                purged.add(uid)

    roles = np.full(frame_length, "outside", dtype="U16")
    units = np.full(frame_length, "", dtype="U64")
    for row in table.itertuples(index=False):
        uid = str(row.unit_id)
        units[int(row.global_row)] = uid
        role = "purged" if uid in purged else unit_roles[uid]
        roles[int(row.global_row)] = role

    for role in ("fit", "calibration", "threshold", "score"):
        labels = np.asarray([int(row.label) for row in table.itertuples(index=False) if roles[int(row.global_row)] == role])
        if len(labels) == 0 or len(np.unique(labels)) != 2:
            raise ValueError(f"fold {fold_index}: {role} lacks both classes")
    for left, right in zip(("fit", "calibration", "threshold"), ("calibration", "threshold", "score")):
        left_times = pd.to_datetime(frame_time_from_roles(roles, table, left), utc=True)
        right_times = pd.to_datetime(frame_time_from_roles(roles, table, right), utc=True)
        gap = right_times.min() - left_times.max()
        if not gap > pd.Timedelta(hours=v1.PURGE_HOURS):
            raise ValueError(f"fold {fold_index}: strict purge failed {left}->{right}: {gap}")
    return roles, units, sorted(purged), {
        "calibration_start": pd.Timestamp(cal_start).isoformat(),
        "threshold_start": pd.Timestamp(threshold_start).isoformat(),
        "score_start": pd.Timestamp(score_start).isoformat(),
        "score_end_exclusive": pd.Timestamp(score_end_boundary).isoformat(),
        "planned_score_positive_index_range": [int(first_score_positive), int(after_score_positive - 1)],
    }


def decision_metrics(y, pred):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred)); tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / denom if denom else float("nan")
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "POD": pod, "FPR": fpr, "FAR": far, "TSS": pod - fpr, "HSS": hss}


def pooled_probability_metrics(y, p, reference):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); reference = np.asarray(reference, dtype=float)
    brier = float(np.mean((p - y) ** 2)); ref_brier = float(np.mean((reference - y) ** 2))
    edges = np.linspace(0, 1, 11); ece = 0.0
    for i in range(10):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask])))
    return {
        "BRIER": brier,
        "BRIER_SKILL": float(1 - brier / ref_brier) if ref_brier > 0 else None,
        "ECE": ece,
        "AUPRC": float(average_precision_score(y, p)),
        "AUROC": float(roc_auc_score(y, p)),
    }


def bootstrap_binary_difference(y, pred_a, pred_b, units, replicates=10000, seed=20260905):
    y = np.asarray(y, dtype=int); pred_a = np.asarray(pred_a, dtype=bool); pred_b = np.asarray(pred_b, dtype=bool); units = np.asarray(units, dtype=str)
    unique = np.unique(units); rows = {u: np.flatnonzero(units == u) for u in unique}
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([rows[u] for u in sampled])
        if len(np.unique(y[ix])) != 2:
            continue
        diffs.append(decision_metrics(y[ix], pred_a[ix])["TSS"] - decision_metrics(y[ix], pred_b[ix])["TSS"])
    values = np.asarray(diffs, dtype=float)
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
        raise ValueError("output directory must be new")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = ctx.prepare_frame(features, events)
    idx, table, summaries, positive_ids = segment_premonitor_units(frame, y, event_ids)
    fold_results = []
    pooled = {arm: {"y": [], "p": [], "pred": [], "units": [], "reference": [], "fold": []} for arm in ARMS}

    for fold in range(FOLD_COUNT):
        roles, units, purged, boundaries = build_fold_roles(len(frame), idx, table, summaries, positive_ids, fold)
        fit = roles == "fit"
        prevalence = float(np.mean(y[fit]))
        raw_base = ctx.fit_xgb_family(frame, base, y, fit)
        raw_xrs_concat = ctx.fit_xgb_family(frame, base + xrs, y, fit)
        raw_xrs_only = ctx.fit_xgb_family(frame, xrs, y, fit)
        raw_proton_only = ctx.fit_xgb_family(frame, proton, y, fit)

        probabilities = {}; thresholds = {}; metadata = {}
        probabilities["BASE_SOLAR"], thresholds["BASE_SOLAR"], metadata["BASE_SOLAR"] = ctx.calibrate_direct(raw_base, y, roles)
        probabilities["BASE_PLUS_XRS"], thresholds["BASE_PLUS_XRS"], metadata["BASE_PLUS_XRS"] = ctx.calibrate_direct(raw_xrs_concat, y, roles)
        probabilities["LATE_FUSION_SOLAR_XRS"], thresholds["LATE_FUSION_SOLAR_XRS"], metadata["LATE_FUSION_SOLAR_XRS"] = ctx.fit_late_stack([raw_base, raw_xrs_only], y, roles)
        probabilities["LATE_FUSION_SOLAR_XRS_PROTON"], thresholds["LATE_FUSION_SOLAR_XRS_PROTON"], metadata["LATE_FUSION_SOLAR_XRS_PROTON"] = ctx.fit_late_stack([raw_base, raw_xrs_only, raw_proton_only], y, roles)

        support = {}
        for role in ("fit", "calibration", "threshold", "score"):
            mask = roles == role
            support[role] = {
                "rows": int(mask.sum()), "positives": int(y[mask].sum()),
                "from": frame.loc[mask, "window_end"].min().isoformat(),
                "to": frame.loc[mask, "window_end"].max().isoformat(),
                "units": int(len(np.unique(units[mask]))),
            }
        arms = {}; score_mask = roles == "score"
        for arm in ARMS:
            metrics = ctx.evaluate_role(y, probabilities[arm], thresholds[arm], roles, "score", prevalence)
            arms[arm] = {"threshold": float(thresholds[arm]), "model_metadata": metadata[arm], "score": metrics}
            pooled[arm]["y"].append(y[score_mask]); pooled[arm]["p"].append(probabilities[arm][score_mask])
            pooled[arm]["pred"].append(probabilities[arm][score_mask] >= thresholds[arm]); pooled[arm]["units"].append(units[score_mask])
            pooled[arm]["reference"].append(np.full(int(score_mask.sum()), prevalence)); pooled[arm]["fold"].append(np.full(int(score_mask.sum()), fold))
        fold_results.append({"fold": fold, "boundaries": boundaries, "purged_units": purged, "support": support, "arms": arms})

    pooled_results = {}
    for arm in ARMS:
        values = {key: np.concatenate(chunks) for key, chunks in pooled[arm].items()}
        pooled_results[arm] = {
            **decision_metrics(values["y"], values["pred"]),
            **pooled_probability_metrics(values["y"], values["p"], values["reference"]),
            "rows": int(len(values["y"])), "positives": int(values["y"].sum()), "folds": FOLD_COUNT,
            "matched_detection_on_pooled_calibrated_probabilities": {str(pod): v1.minimum_far_at_pod(values["y"], values["p"], pod) for pod in (0.6, 0.7, 0.8, 0.9)},
        }

    base_values = {key: np.concatenate(chunks) for key, chunks in pooled["BASE_SOLAR"].items()}
    comparisons = {}; fold_differences = {}
    for candidate in ("BASE_PLUS_XRS", "LATE_FUSION_SOLAR_XRS", "LATE_FUSION_SOLAR_XRS_PROTON"):
        candidate_values = {key: np.concatenate(chunks) for key, chunks in pooled[candidate].items()}
        if not np.array_equal(base_values["y"], candidate_values["y"]) or not np.array_equal(base_values["units"], candidate_values["units"]):
            raise ValueError("pooled candidate/base alignment failure")
        comparisons[f"{candidate}_minus_BASE_SOLAR"] = bootstrap_binary_difference(base_values["y"], candidate_values["pred"], base_values["pred"], base_values["units"])
        diffs = [fold_results[i]["arms"][candidate]["score"]["TSS"] - fold_results[i]["arms"]["BASE_SOLAR"]["score"]["TSS"] for i in range(FOLD_COUNT)]
        fold_differences[candidate] = {"differences": [float(x) for x in diffs], "median": float(np.median(diffs)), "positive_folds": int(np.sum(np.asarray(diffs) > 0))}

    late = comparisons["LATE_FUSION_SOLAR_XRS_minus_BASE_SOLAR"]
    late_fold = fold_differences["LATE_FUSION_SOLAR_XRS"]
    pooled_delta = pooled_results["LATE_FUSION_SOLAR_XRS"]["TSS"] - pooled_results["BASE_SOLAR"]["TSS"]
    candidate_rule = {
        "median_fold_difference_positive": bool(late_fold["median"] > 0),
        "pooled_TSS_difference": float(pooled_delta),
        "pooled_TSS_difference_positive": bool(pooled_delta > 0),
        "pooled_bootstrap_ci_lower": float(late["ci_lower_95"]),
        "pooled_bootstrap_ci_upper": float(late["ci_upper_95"]),
        "pooled_bootstrap_does_not_show_clear_harm": bool(late["ci_upper_95"] > 0),
    }
    candidate_rule["eligible_for_candidate_freeze_under_preregistered_rule"] = bool(candidate_rule["median_fold_difference_positive"] and candidate_rule["pooled_TSS_difference_positive"] and candidate_rule["pooled_bootstrap_does_not_show_clear_harm"])

    summary = {
        "status": "COMPLETED_PREMONITOR_ROLLING_LATE_FUSION_REPLICATION",
        "target": v1.TARGET, "locked_test_accessed": False, "monitor_rows_accessed": False,
        "feature_table_sha256": ctx.digest(features), "event_catalogue_sha256": ctx.digest(events),
        "dropped_non_numeric_columns": dropped, "premonitor_positive_units": int(len(positive_ids)),
        "fold_contract": {"folds": FOLD_COUNT, "score_positive_units_per_fold_planned": SCORE_POSITIVES_PER_FOLD, "calibration_positive_units_planned": CALIBRATION_POSITIVES, "threshold_positive_units_planned": THRESHOLD_POSITIVES},
        "folds": fold_results, "pooled": pooled_results, "paired_comparisons": comparisons,
        "fold_TSS_differences": fold_differences, "candidate_rule": candidate_rule,
    }
    ctx.save_json(output / "summary.json", summary)
    ctx.save_json(output / "receipt.json", {
        "status": "DEVELOPMENT_ONLY_PREMONITOR_ROLLING_REPLICATION",
        "preregistration": "config/late_fusion_rolling_preregistration_2026-09-05.json",
        "locked_test_accessed": False, "monitor_rows_accessed": False,
        "unfavorable_folds_dropped": False, "post_result_hyperparameter_change": False,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
