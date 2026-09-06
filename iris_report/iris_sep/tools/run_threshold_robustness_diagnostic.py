"""Threshold-stability diagnostic on the frozen pre-monitor rolling folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as ctx
from iris_report.iris_sep.tools import run_late_fusion_rolling_replication as roll

POLICIES = ("MAX_TSS", "POD80_MIN_FAR", "BOOTSTRAP_MEDIAN_MAX_TSS")
MODELS = ("BASE_SOLAR", "LATE_FUSION_SOLAR_XRS", "LATE_FUSION_SOLAR_XRS_PROTON")
THRESHOLD_BOOTSTRAPS = 1000
THRESHOLD_SEED = 20260905


def pod80_threshold(y, p):
    result = v1.minimum_far_at_pod(np.asarray(y), np.asarray(p), 0.8)
    if result is None:
        raise ValueError("POD80 threshold unavailable")
    return float(result["threshold"])


def bootstrap_median_threshold(y, p, units, replicates=THRESHOLD_BOOTSTRAPS, seed=THRESHOLD_SEED):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); units = np.asarray(units, dtype=str)
    unique = np.unique(units); rows = {u: np.flatnonzero(units == u) for u in unique}
    rng = np.random.default_rng(seed); thresholds = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([rows[u] for u in sampled])
        if len(np.unique(y[ix])) != 2:
            continue
        thresholds.append(float(v1.select_threshold(y[ix], p[ix])))
    if not thresholds:
        raise ValueError("no valid threshold bootstrap replicates")
    values = np.asarray(thresholds, dtype=float)
    return float(np.median(values)), {
        "valid_replicates": int(len(values)),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "q975": float(np.quantile(values, 0.975)),
    }


def selected_thresholds(y, p, units, threshold_mask):
    yt = y[threshold_mask]; pt = p[threshold_mask]; ut = units[threshold_mask]
    boot, receipt = bootstrap_median_threshold(yt, pt, ut)
    return {
        "MAX_TSS": float(v1.select_threshold(yt, pt)),
        "POD80_MIN_FAR": pod80_threshold(yt, pt),
        "BOOTSTRAP_MEDIAN_MAX_TSS": boot,
    }, receipt


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be immutable/new")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = ctx.prepare_frame(features, events)
    idx, table, summaries, positive_ids = roll.segment_premonitor_units(frame, y, event_ids)
    fold_results = []
    pooled = {
        model: {
            policy: {"y": [], "p": [], "pred": [], "units": [], "reference": []}
            for policy in POLICIES
        }
        for model in MODELS
    }

    for fold in range(roll.FOLD_COUNT):
        roles, units, purged, boundaries = roll.build_fold_roles(len(frame), idx, table, summaries, positive_ids, fold)
        fit = roles == "fit"; threshold_mask = roles == "threshold"; score = roles == "score"
        prevalence = float(np.mean(y[fit]))

        raw_base = ctx.fit_xgb_family(frame, base, y, fit)
        raw_xrs_only = ctx.fit_xgb_family(frame, xrs, y, fit)
        raw_proton_only = ctx.fit_xgb_family(frame, proton, y, fit)

        probabilities = {}; model_metadata = {}
        probabilities["BASE_SOLAR"], _unused_t, model_metadata["BASE_SOLAR"] = ctx.calibrate_direct(raw_base, y, roles)
        probabilities["LATE_FUSION_SOLAR_XRS"], _unused_t, model_metadata["LATE_FUSION_SOLAR_XRS"] = ctx.fit_late_stack([raw_base, raw_xrs_only], y, roles)
        probabilities["LATE_FUSION_SOLAR_XRS_PROTON"], _unused_t, model_metadata["LATE_FUSION_SOLAR_XRS_PROTON"] = ctx.fit_late_stack([raw_base, raw_xrs_only, raw_proton_only], y, roles)

        models = {}
        for model in MODELS:
            thresholds, boot_receipt = selected_thresholds(y, probabilities[model], units, threshold_mask)
            policy_results = {}
            for policy in POLICIES:
                t = thresholds[policy]
                metrics = ctx.evaluate_role(y, probabilities[model], t, roles, "score", prevalence)
                policy_results[policy] = {"threshold": float(t), "score": metrics}
                pooled[model][policy]["y"].append(y[score])
                pooled[model][policy]["p"].append(probabilities[model][score])
                pooled[model][policy]["pred"].append(probabilities[model][score] >= t)
                pooled[model][policy]["units"].append(units[score])
                pooled[model][policy]["reference"].append(np.full(int(score.sum()), prevalence))
            models[model] = {
                "model_metadata": model_metadata[model],
                "bootstrap_threshold_receipt": boot_receipt,
                "policies": policy_results,
            }
        fold_results.append({
            "fold": fold,
            "boundaries": boundaries,
            "purged_units": purged,
            "models": models,
        })

    pooled_results = {}; policy_comparisons = {}; model_comparisons = {}; arrays = {}
    for model in MODELS:
        pooled_results[model] = {}; arrays[model] = {}
        for policy in POLICIES:
            values = {k: np.concatenate(v) for k, v in pooled[model][policy].items()}
            arrays[model][policy] = values
            pooled_results[model][policy] = {
                **roll.decision_metrics(values["y"], values["pred"]),
                **roll.pooled_probability_metrics(values["y"], values["p"], values["reference"]),
                "rows": int(len(values["y"])),
                "positives": int(values["y"].sum()),
                "matched_detection_on_pooled_calibrated_probabilities": {
                    str(pod): v1.minimum_far_at_pod(values["y"], values["p"], pod)
                    for pod in (0.6, 0.7, 0.8, 0.9)
                },
            }
        base_policy = arrays[model]["MAX_TSS"]
        policy_comparisons[model] = {}
        for policy in ("POD80_MIN_FAR", "BOOTSTRAP_MEDIAN_MAX_TSS"):
            candidate = arrays[model][policy]
            if not np.array_equal(candidate["y"], base_policy["y"]) or not np.array_equal(candidate["units"], base_policy["units"]):
                raise ValueError("within-model policy alignment failure")
            policy_comparisons[model][f"{policy}_minus_MAX_TSS"] = roll.bootstrap_binary_difference(
                candidate["y"], candidate["pred"], base_policy["pred"], candidate["units"]
            )

    for policy in POLICIES:
        base_values = arrays["BASE_SOLAR"][policy]
        model_comparisons[policy] = {}
        for candidate_model in ("LATE_FUSION_SOLAR_XRS", "LATE_FUSION_SOLAR_XRS_PROTON"):
            candidate = arrays[candidate_model][policy]
            if not np.array_equal(candidate["y"], base_values["y"]) or not np.array_equal(candidate["units"], base_values["units"]):
                raise ValueError("between-model policy alignment failure")
            model_comparisons[policy][f"{candidate_model}_minus_BASE_SOLAR"] = roll.bootstrap_binary_difference(
                candidate["y"], candidate["pred"], base_values["pred"], candidate["units"]
            )

    fold_policy_differences = {}
    for model in MODELS:
        fold_policy_differences[model] = {}
        for policy in ("POD80_MIN_FAR", "BOOTSTRAP_MEDIAN_MAX_TSS"):
            diffs = [
                fold_results[i]["models"][model]["policies"][policy]["score"]["TSS"]
                - fold_results[i]["models"][model]["policies"]["MAX_TSS"]["score"]["TSS"]
                for i in range(roll.FOLD_COUNT)
            ]
            fold_policy_differences[model][policy] = {
                "differences": [float(x) for x in diffs],
                "median": float(np.median(diffs)),
                "positive_folds": int(np.sum(np.asarray(diffs) > 0)),
            }

    summary = {
        "status": "COMPLETED_PREMONITOR_THRESHOLD_ROBUSTNESS_DIAGNOSTIC",
        "target": v1.TARGET,
        "locked_test_accessed": False,
        "monitor_rows_accessed": False,
        "feature_table_sha256": ctx.digest(features),
        "event_catalogue_sha256": ctx.digest(events),
        "dropped_non_numeric_columns": dropped,
        "policies": list(POLICIES),
        "threshold_bootstrap_replicates": THRESHOLD_BOOTSTRAPS,
        "folds": fold_results,
        "pooled": pooled_results,
        "within_model_policy_comparisons": policy_comparisons,
        "between_model_same_policy_comparisons": model_comparisons,
        "fold_policy_TSS_differences": fold_policy_differences,
    }
    ctx.save_json(output / "summary.json", summary)
    ctx.save_json(output / "receipt.json", {
        "status": "DEVELOPMENT_ONLY_THRESHOLD_ROBUSTNESS",
        "preregistration": "config/threshold_robustness_preregistration_2026-09-05.json",
        "monitor_rows_accessed": False,
        "locked_test_accessed": False,
        "policies_added_after_result": False,
        "unfavorable_folds_dropped": False,
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
