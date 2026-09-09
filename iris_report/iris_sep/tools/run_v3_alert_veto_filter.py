"""Development-only monotone veto on frozen V3 missing-feed alerts.

The V3 probability and MAX_TSS threshold remain unchanged.  A tiny state-specific
meta-model is trained on the calibration role.  A veto threshold is selected on
the threshold role with at most one TP loss.  On score, the layer may only
suppress alerts already emitted by frozen V3; it cannot create a new alert.

The score block is already inspected.  Results from this runner are development
architecture evidence only and cannot establish independent skill.
"""
from __future__ import annotations

import argparse
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
)
from iris_report.iris_sep.src.iris_sep.modeling.alert_veto_filter import (
    apply_veto,
    select_veto_threshold,
)
from iris_report.iris_sep.tools import run_availability_conditioned_fallback as v1_fallback
from iris_report.iris_sep.tools import run_availability_distilled_fallback_v3 as v3
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_v3_alert_decision_filter as prior


STATES = ("NO_XRS", "NO_PROTON")
PRIMARY_POLICY = "MAX_TSS"
FULL_REPRO_TOLERANCE = 1e-6


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

    for required in ("calibration", "threshold", "score"):
        mask = roles == required
        if not np.any(mask) or int(np.sum(np.asarray(y)[mask])) == 0:
            raise ValueError(f"event-bearing {required} role required")

    v1_model = v1_fallback._fit_all_states(frame, y, roles, units, base, xrs, proton)
    v3_model = v3._fit_distilled_states(frame, y, roles, units, base, xrs, proton)
    p1 = v1_model["probability"]
    p3 = v3_model["probability"]
    full_diff = float(np.max(np.abs(p1["FULL"] - p3["FULL"])))
    if full_diff > FULL_REPRO_TOLERANCE:
        raise ValueError(f"V3 changed FULL probability by {full_diff}")

    family_stats = {
        "solar": prior._seed_stats(v1_model["family_models"]["solar"], frame, base),
        "xrs": prior._seed_stats(v1_model["family_models"]["xrs"], frame, xrs),
        "proton": prior._seed_stats(v1_model["family_models"]["proton"], frame, proton),
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
            v3_probability=p3[state],
            v1_probability=p1[state],
            solar_only_probability=p3["NO_XRS_OR_PROTON"],
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
        threshold_v3_alert = np.asarray(p3[state])[threshold_role] >= v3_threshold
        selection = select_veto_threshold(
            labels=np.asarray(y)[threshold_role],
            decision_score=decision_score[threshold_role],
            baseline_alert=threshold_v3_alert,
            max_true_positive_loss=MAX_THRESHOLD_ROLE_TP_LOSS,
        )
        veto_threshold = float(selection["threshold"])

        score_v3_alert = np.asarray(p3[state])[score] >= v3_threshold
        score_veto_alert = apply_veto(
            baseline_alert=score_v3_alert,
            decision_score=decision_score[score],
            threshold=veto_threshold,
        )
        if np.any(score_veto_alert & ~score_v3_alert):
            raise AssertionError("veto created a new score alert")

        reference = binary_metrics(np.asarray(y)[score], score_v3_alert)
        candidate = binary_metrics(np.asarray(y)[score], score_veto_alert)
        improvement = improvement_summary(reference, candidate)
        bootstrap = prior._paired_unit_bootstrap(
            labels=np.asarray(y)[score],
            reference_alert=score_v3_alert,
            candidate_alert=score_veto_alert,
            units=np.asarray(units)[score],
        )
        all_large = all_large and bool(improvement["large_development_improvement_gate"])

        state_results[state] = {
            "available_context_family": context.upper(),
            "absent_family_used_by_filter": False,
            "veto_only": True,
            "new_alerts_outside_v3_allowed": False,
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
                "veto_threshold_selection": selection,
            },
            "score_role": {
                "rows": int(np.sum(score)),
                "positives": int(np.sum(np.asarray(y)[score])),
                "v3_reference": reference,
                "veto_layer": candidate,
                "improvement": improvement,
                "paired_unit_bootstrap": bootstrap,
            },
            "probability_replaced": False,
            "probability_for_brier_remains_v3": True,
        }
        predictions[f"p_v3_{state}"] = p3[state]
        predictions[f"veto_score_{state}"] = decision_score
        predictions[f"v3_alert_{state}"] = np.asarray(p3[state]) >= v3_threshold
        predictions[f"veto_alert_{state}"] = (np.asarray(p3[state]) >= v3_threshold) & (decision_score >= veto_threshold)

    predictions_path = output / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.17g")
    summary = {
        "status": "COMPLETED_V3_ALERT_VETO_FILTER_DEVELOPMENT_ONLY",
        "architecture": "IRIS_V3_ALERT_VETO_FILTER_V2",
        "predecessor_failed_run": 34206898291,
        "predecessor_failure_mechanism": "meta-model could create alerts outside frozen V3 alert set",
        "structural_fix": "candidate alert is V3_MAX_TSS_alert AND veto_score>=frozen_veto_threshold",
        "target": "new_sep_10mev_10pfu_within_24h",
        "selection_status": "POST_HOC_DEVELOPMENT_SELECTION_SCORE_ALREADY_INSPECTED",
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "development_score_already_inspected": True,
        "candidate_probability_changed": False,
        "runtime_missing_family_reconstruction": False,
        "new_alerts_outside_v3_allowed": False,
        "max_threshold_role_tp_loss": MAX_THRESHOLD_ROLE_TP_LOSS,
        "large_improvement_definition": "score-role TP must not decrease and false positives must fall by at least 20 percent versus frozen V3 MAX_TSS alerts",
        "full_state_max_abs_difference_v1_vs_v3": full_diff,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric": dropped,
        "states": state_results,
        "both_states_pass_large_development_improvement_gate": bool(all_large),
        "candidate_for_future_prospective_comparison": bool(all_large),
        "claim_boundary": "Development-selected pure-veto alert policy. Score was already inspected. Future witnessed outcomes are required for independent evidence.",
        "source_hashes": {
            "features": prior.digest(features),
            "events": prior.digest(events),
            "runner": prior.digest(Path(__file__)),
        },
    }
    summary_path = output / "summary.json"
    prior.save_json(summary_path, summary)
    prior.save_json(output / "receipt.json", {
        "format": "IRIS_V3_ALERT_VETO_FILTER_RECEIPT_V2",
        "summary_sha256": prior.digest(summary_path),
        "predictions_sha256": prior.digest(predictions_path),
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
        "states": {state: result["states"][state]["score_role"] for state in STATES},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
