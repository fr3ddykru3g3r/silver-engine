"""Development-only V3 alert-decision layer for false-alarm reduction.

The V3 probability remains unchanged.  This experiment trains one tiny
state-specific XGBoost decision model on the calibration role, selects its alert
threshold on the threshold role with at most one TP loss relative to the frozen
V3 MAX_TSS policy, and evaluates on the already-inspected score role.

The exact model configurations were selected after development inspection and
are frozen in ``modeling.alert_decision_filter``.  Therefore this is architecture
selection evidence only, never an untouched or preregistered skill result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.alert_decision_filter import (
    MAX_THRESHOLD_ROLE_TP_LOSS,
    STATE_MODEL_PARAMS,
    binary_metrics,
    build_model,
    improvement_summary,
    meta_features,
    select_decision_threshold,
)
from iris_report.iris_sep.tools import run_availability_conditioned_fallback as v1_fallback
from iris_report.iris_sep.tools import run_availability_distilled_fallback_v3 as v3
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs


STATES = ("NO_XRS", "NO_PROTON")
PRIMARY_POLICY = "MAX_TSS"
FULL_REPRO_TOLERANCE = 1e-6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_finite(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(_finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _seed_stats(models, frame: pd.DataFrame, names) -> dict[str, np.ndarray]:
    values = np.stack(
        [model.predict_proba(frame.loc[:, names])[:, 1] for model in models],
        axis=0,
    )
    return {
        "median": np.median(values, axis=0),
        "range": np.max(values, axis=0) - np.min(values, axis=0),
        "std": np.std(values, axis=0),
    }


def _paired_unit_bootstrap(
    *,
    labels: np.ndarray,
    reference_alert: np.ndarray,
    candidate_alert: np.ndarray,
    units: np.ndarray,
    replicates: int = 10000,
    seed: int = 20260908,
) -> dict:
    unique = np.unique(units)
    row_map = {unit: np.flatnonzero(units == unit) for unit in unique}
    rng = np.random.default_rng(seed)
    tss_delta = []
    fp_delta = []
    tp_delta = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([row_map[u] for u in sampled])
        ref = binary_metrics(labels[ix], reference_alert[ix])
        cand = binary_metrics(labels[ix], candidate_alert[ix])
        if math.isfinite(float(ref["TSS"])) and math.isfinite(float(cand["TSS"])):
            tss_delta.append(float(cand["TSS"] - ref["TSS"]))
        fp_delta.append(int(cand["FP"]) - int(ref["FP"]))
        tp_delta.append(int(cand["TP"]) - int(ref["TP"]))
    return {
        "replicates": int(replicates),
        "tss_delta_median": float(np.median(tss_delta)),
        "tss_delta_ci95": [float(np.quantile(tss_delta, 0.025)), float(np.quantile(tss_delta, 0.975))],
        "fp_delta_median": float(np.median(fp_delta)),
        "fp_delta_ci95": [float(np.quantile(fp_delta, 0.025)), float(np.quantile(fp_delta, 0.975))],
        "tp_delta_median": float(np.median(tp_delta)),
        "tp_delta_ci95": [float(np.quantile(tp_delta, 0.025)), float(np.quantile(tp_delta, 0.975))],
    }


def run(features: Path, events: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("output must be a new immutable directory")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)

    monitor = roles == "monitor"
    monitor_rows_excluded = int(np.sum(monitor))
    if monitor_rows_excluded:
        keep = ~monitor
        frame = frame.loc[keep].reset_index(drop=True)
        y = np.asarray(y)[keep]
        roles = np.asarray(roles)[keep]
        units = np.asarray(units)[keep]
    if np.any(roles == "monitor"):
        raise ValueError("monitor exclusion failed")

    for required_role in ("calibration", "threshold", "score"):
        mask = roles == required_role
        if not np.any(mask) or int(np.sum(np.asarray(y)[mask])) == 0:
            raise ValueError(f"event-bearing {required_role} role required")

    v1_model = v1_fallback._fit_all_states(frame, y, roles, units, base, xrs, proton)
    v3_model = v3._fit_distilled_states(frame, y, roles, units, base, xrs, proton)
    v1_probability = v1_model["probability"]
    v3_probability = v3_model["probability"]

    full_diff = float(np.max(np.abs(v1_probability["FULL"] - v3_probability["FULL"])))
    if full_diff > FULL_REPRO_TOLERANCE:
        raise ValueError(f"V3 changed FULL probability by {full_diff}")

    family_stats = {
        "solar": _seed_stats(v1_model["family_models"]["solar"], frame, base),
        "xrs": _seed_stats(v1_model["family_models"]["xrs"], frame, xrs),
        "proton": _seed_stats(v1_model["family_models"]["proton"], frame, proton),
    }

    calibration = roles == "calibration"
    threshold_role = roles == "threshold"
    score = roles == "score"
    predictions = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
    })

    state_results = {}
    all_large = True
    for state in STATES:
        context = "proton" if state == "NO_XRS" else "xrs"
        x = meta_features(
            v3_probability=v3_probability[state],
            v1_probability=v1_probability[state],
            solar_only_probability=v3_probability["NO_XRS_OR_PROTON"],
            raw_solar_probability=family_stats["solar"]["median"],
            raw_context_probability=family_stats[context]["median"],
            solar_seed_range=family_stats["solar"]["range"],
            solar_seed_std=family_stats["solar"]["std"],
            context_seed_range=family_stats[context]["range"],
            context_seed_std=family_stats[context]["std"],
        )
        model = build_model(state)
        model.fit(x[calibration], np.asarray(y)[calibration], verbose=False)
        decision_score = model.predict_proba(x)[:, 1]

        v3_threshold = float(v3_model["thresholds"][state][PRIMARY_POLICY])
        baseline_threshold_alert = np.asarray(v3_probability[state])[threshold_role] >= v3_threshold
        baseline_threshold_metrics = binary_metrics(np.asarray(y)[threshold_role], baseline_threshold_alert)
        threshold_selection = select_decision_threshold(
            labels=np.asarray(y)[threshold_role],
            decision_score=decision_score[threshold_role],
            baseline_true_positives=int(baseline_threshold_metrics["TP"]),
            max_true_positive_loss=MAX_THRESHOLD_ROLE_TP_LOSS,
        )
        alert_threshold = float(threshold_selection["threshold"])

        reference_alert = np.asarray(v3_probability[state])[score] >= v3_threshold
        candidate_alert = decision_score[score] >= alert_threshold
        reference_metrics = binary_metrics(np.asarray(y)[score], reference_alert)
        candidate_metrics = binary_metrics(np.asarray(y)[score], candidate_alert)
        improvement = improvement_summary(reference_metrics, candidate_metrics)
        bootstrap = _paired_unit_bootstrap(
            labels=np.asarray(y)[score],
            reference_alert=reference_alert,
            candidate_alert=candidate_alert,
            units=np.asarray(units)[score],
        )
        all_large = all_large and bool(improvement["large_development_improvement_gate"])

        state_results[state] = {
            "available_context_family": context.upper(),
            "absent_family_used_by_filter": False,
            "meta_feature_count": int(x.shape[1]),
            "model_parameters": STATE_MODEL_PARAMS[state],
            "training_role": {
                "name": "calibration",
                "rows": int(np.sum(calibration)),
                "positives": int(np.sum(np.asarray(y)[calibration])),
            },
            "threshold_role": {
                "name": "threshold",
                "rows": int(np.sum(threshold_role)),
                "positives": int(np.sum(np.asarray(y)[threshold_role])),
                "v3_reference_threshold": v3_threshold,
                "v3_reference_metrics": baseline_threshold_metrics,
                "decision_threshold_selection": threshold_selection,
            },
            "score_role": {
                "rows": int(np.sum(score)),
                "positives": int(np.sum(np.asarray(y)[score])),
                "v3_reference": reference_metrics,
                "decision_layer": candidate_metrics,
                "improvement": improvement,
                "paired_unit_bootstrap": bootstrap,
            },
            "probability_replaced": False,
            "probability_for_brier_remains_v3": True,
        }
        predictions[f"p_v3_{state}"] = v3_probability[state]
        predictions[f"alert_score_{state}"] = decision_score
        predictions[f"v3_alert_{state}"] = np.asarray(v3_probability[state]) >= v3_threshold
        predictions[f"decision_alert_{state}"] = decision_score >= alert_threshold

    predictions_path = output / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.17g")

    summary = {
        "status": "COMPLETED_V3_ALERT_DECISION_FILTER_DEVELOPMENT_ONLY",
        "architecture": "IRIS_V3_ALERT_DECISION_FILTER_V1",
        "target": "new_sep_10mev_10pfu_within_24h",
        "selection_status": "POST_HOC_DEVELOPMENT_SELECTION_SCORE_ALREADY_INSPECTED",
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "development_score_already_inspected": True,
        "candidate_probability_changed": False,
        "runtime_missing_family_reconstruction": False,
        "max_threshold_role_tp_loss": MAX_THRESHOLD_ROLE_TP_LOSS,
        "large_improvement_definition": "score-role TP must not decrease and false positives must fall by at least 20 percent versus frozen V3 MAX_TSS alerts",
        "full_state_max_abs_difference_v1_vs_v3": full_diff,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric": dropped,
        "states": state_results,
        "both_states_pass_large_development_improvement_gate": bool(all_large),
        "candidate_for_future_prospective_comparison": bool(all_large),
        "claim_boundary": (
            "Development-selected alert policy. The score cohort was already inspected and the exact configurations were selected after development analysis. "
            "It must be frozen before and validated on future witnessed outcomes before any independent skill claim."
        ),
        "source_hashes": {
            "features": digest(features),
            "events": digest(events),
            "runner": digest(Path(__file__)),
        },
    }
    summary_path = output / "summary.json"
    save_json(summary_path, summary)
    save_json(output / "receipt.json", {
        "format": "IRIS_V3_ALERT_DECISION_FILTER_RECEIPT_V1",
        "summary_sha256": digest(summary_path),
        "predictions_sha256": digest(predictions_path),
        "locked_test_accessed": False,
        "monitor_used": False,
        "independent_validation": False,
        "both_states_large_gate": bool(all_large),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.features, args.events, args.output)
    print(json.dumps({
        "status": result["status"],
        "both_states_large_gate": result["both_states_pass_large_development_improvement_gate"],
        "states": {
            state: result["states"][state]["score_role"]
            for state in STATES
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
