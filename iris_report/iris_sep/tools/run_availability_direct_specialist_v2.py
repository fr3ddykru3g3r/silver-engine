"""Development-only V2 benchmark for direct remaining-sensor specialists.

Compares the preregistered availability fallback V1 against direct raw-feature
XGBoost specialists on the exact same chronological NEW-crossing cohort.  The
monitor is excluded before modeling and the locked test is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.availability_direct_specialist import (
    DirectSpecialistConfig,
    fit_predict_direct_specialist,
    state_feature_names,
)
from iris_report.iris_sep.tools import run_availability_conditioned_fallback as v1_fallback
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_operator_review_budget_diagnostic as review


PREREG = "config/availability_direct_specialist_v2_preregistration_2026-09-07.json"
STATES = ("NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")
PRIMARY_POLICY = "MAX_TSS"
PRIMARY_REVIEW_FRACTION = 0.05


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


def _keep_gate(v1_metrics: dict, v2_metrics: dict, v1_review: dict, v2_review: dict) -> dict:
    tss_delta = float(v2_metrics["TSS"] - v1_metrics["TSS"])
    brier_delta = float(v2_metrics["BRIER"] - v1_metrics["BRIER"])
    ece_delta = float(v2_metrics["ECE"] - v1_metrics["ECE"])
    v1_enrichment = v1_review["enrichment_vs_random_review"]
    v2_enrichment = v2_review["enrichment_vs_random_review"]
    enrichment_delta = None
    if v1_enrichment is not None and v2_enrichment is not None:
        enrichment_delta = float(v2_enrichment - v1_enrichment)
    checks = {
        "TSS_not_lower_by_more_than_0p02": math.isfinite(tss_delta) and tss_delta >= -0.02,
        "Brier_not_worse_by_more_than_0p005": math.isfinite(brier_delta) and brier_delta <= 0.005,
        "ECE_not_worse_by_more_than_0p01": math.isfinite(ece_delta) and ece_delta <= 0.01,
        "review_enrichment_not_lower_by_more_than_1x": enrichment_delta is not None
        and math.isfinite(enrichment_delta)
        and enrichment_delta >= -1.0,
        "probabilities_finite": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas_v2_minus_v1": {
            "TSS": tss_delta,
            "BRIER": brier_delta,
            "ECE": ece_delta,
            "review_enrichment": enrichment_delta,
        },
    }


def run(features: Path, events: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    root = Path(__file__).resolve().parents[1]
    prereg = root / PREREG
    if not prereg.exists():
        raise ValueError("direct-specialist V2 preregistration missing")

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)

    monitor = roles == "monitor"
    monitor_rows_excluded = int(monitor.sum())
    if monitor_rows_excluded:
        keep = ~monitor
        frame = frame.loc[keep].reset_index(drop=True)
        y = np.asarray(y)[keep]
        event_ids = np.asarray(event_ids)[keep]
        roles = np.asarray(roles)[keep]
        units = np.asarray(units)[keep]
    if np.any(roles == "monitor"):
        raise ValueError("monitor exclusion failed")

    score = roles == "score"
    fit = roles == "fit"
    if not score.any() or int(np.sum(y[score])) == 0:
        raise ValueError("event-bearing score role required")

    # Rebuild V1 on the same monitor-excluded cohort.  This is the frozen scalar
    # expert-fusion comparator, not a new candidate.
    v1_model = v1_fallback._fit_all_states(frame, y, roles, units, base, xrs, proton)
    v1_probability = v1_model["probability"]

    v2_probability: dict[str, np.ndarray] = {"FULL": np.asarray(v1_probability["FULL"], dtype=float)}
    v2_thresholds: dict[str, dict] = {"FULL": v1_model["thresholds"]["FULL"]}
    v2_calibration: dict[str, float] = {"FULL": float(v1_model["calibration_intercept"]["FULL"])}
    v2_receipts: dict[str, dict] = {}

    cfg = DirectSpecialistConfig()
    for state in STATES:
        names = state_feature_names(state, solar=base, xrs=xrs, proton=proton)
        direct = fit_predict_direct_specialist(
            frame=frame,
            labels=y,
            fit_mask=fit,
            feature_names=names,
            config=cfg,
        )
        calibrated, intercept = cf.calibrated_probability(direct["probability"], y, roles)
        v2_probability[state] = np.asarray(calibrated, dtype=float)
        v2_calibration[state] = float(intercept)
        v2_thresholds[state] = cf.thresholds(y, calibrated, roles)
        v2_receipts[state] = {
            "feature_names": list(direct["feature_names"]),
            "feature_schema_sha256": direct["feature_schema_sha256"],
            "seeds": list(direct["seeds"]),
            "fit_rows": int(direct["fit_rows"]),
            "fit_positives": int(direct["fit_positives"]),
            "imputation_used": bool(direct["imputation_used"]),
            "reconstruction_used": bool(direct["reconstruction_used"]),
            "runtime_retraining": bool(direct["runtime_retraining"]),
        }

    prevalence = float(v1_model["fit_prevalence"])
    state_results = {}
    keep_results = {}
    issue_score = frame.loc[score, "window_end"]
    labels_score = np.asarray(y)[score]

    for state in STATES:
        v1_metrics = v1_fallback._whole_score_metrics(
            y,
            v1_probability[state],
            roles,
            v1_model["thresholds"][state][PRIMARY_POLICY],
            prevalence,
        )
        v2_metrics = v1_fallback._whole_score_metrics(
            y,
            v2_probability[state],
            roles,
            v2_thresholds[state][PRIMARY_POLICY],
            prevalence,
        )
        v1_review = review.budget_metrics(
            labels_score,
            np.asarray(v1_probability[state])[score],
            issue_score,
            PRIMARY_REVIEW_FRACTION,
        )
        v2_review = review.budget_metrics(
            labels_score,
            np.asarray(v2_probability[state])[score],
            issue_score,
            PRIMARY_REVIEW_FRACTION,
        )
        gate = _keep_gate(v1_metrics, v2_metrics, v1_review, v2_review)
        keep_results[state] = gate
        state_results[state] = {
            "v1_scalar_fusion": {
                "metrics": v1_metrics,
                "review_5pct": v1_review,
            },
            "v2_direct_specialist": {
                "metrics": v2_metrics,
                "review_5pct": v2_review,
                "calibration_intercept": v2_calibration[state],
                "thresholds": v2_thresholds[state],
                "receipt": v2_receipts[state],
            },
            "keep_gate": gate,
        }

    # The one-feed states are the actual attempt to avoid unnecessary
    # degradation.  Solar-only remains informative but is not allowed to rescue
    # the final candidate decision.
    one_feed_candidate_pass = all(keep_results[state]["passed"] for state in ("NO_XRS", "NO_PROTON"))

    predictions = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
    })
    for state in ("FULL", *STATES):
        predictions[f"p_v1_{state}"] = v1_probability[state]
        predictions[f"p_v2_{state}"] = v2_probability[state]
    predictions_path = output / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.17g")

    summary = {
        "status": "COMPLETED_DIRECT_REMAINING_SENSOR_SPECIALIST_V2_DEVELOPMENT_ONLY",
        "target": "new_sep_10mev_10pfu_within_24h",
        "preregistration": PREREG,
        "preregistration_sha256": digest(prereg),
        "runner_sha256": digest(Path(__file__)),
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "development_score_already_inspected": True,
        "imputation_used": False,
        "reconstruction_used": False,
        "runtime_retraining": False,
        "primary_policy": PRIMARY_POLICY,
        "primary_review_fraction": PRIMARY_REVIEW_FRACTION,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric": dropped,
        "states": state_results,
        "candidate_for_future_untouched_evaluation": bool(one_feed_candidate_pass),
        "candidate_rule": "both NO_XRS and NO_PROTON must pass the preregistered keep gate; development cannot grant NORMAL trust",
        "claim_boundary": "Development-only architecture decision. NORMAL/VALID trust still requires independent locked evidence through availability_trust.py.",
    }
    summary_path = output / "summary.json"
    save_json(summary_path, summary)
    receipt = {
        "status": "DIRECT_SPECIALIST_V2_RECEIPT",
        "summary_sha256": digest(summary_path),
        "predictions_sha256": digest(predictions_path),
        "locked_test_accessed": False,
        "monitor_used": False,
        "normal_status_granted": False,
    }
    save_json(output / "receipt.json", receipt)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_finite(run(args.features, args.events, args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
