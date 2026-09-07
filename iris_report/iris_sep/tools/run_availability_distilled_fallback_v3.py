"""Development-only benchmark for the distilled availability fallback V3.

V3 preserves the V1 family-specialist architecture.  On fit-only cross-fitted
rows, the full three-expert stack acts as a soft teacher for two missing-feed
students.  Students receive only the evidence available at runtime.

The monitor is excluded before modeling and the locked test is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.availability_fallback import (
    evidence_columns,
)
from iris_report.iris_sep.src.iris_sep.modeling.distilled_evidence_stack import (
    DistilledEvidenceStackConfig,
    DistilledPositiveEvidenceStack,
)
from iris_report.iris_sep.src.iris_sep.modeling.positive_evidence_stack import (
    EvidenceStackConfig,
    PositiveEvidenceStack,
)
from iris_report.iris_sep.tools import run_availability_conditioned_fallback as v1_fallback
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_operator_review_budget_diagnostic as review
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as public_benchmark


PREREG = "config/availability_distilled_fallback_v3_preregistration_2026-09-07.json"
PRIMARY_POLICY = "MAX_TSS"
PRIMARY_REVIEW_FRACTION = 0.05
ONE_FEED_STATES = ("NO_XRS", "NO_PROTON")
ALL_STATES = ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")
TEACHER_WEIGHT = 0.35
L2_WEIGHT = 0.03
FULL_REPRO_TOLERANCE = 1e-10


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


def _fit_distilled_states(frame, y, roles, units, base, xrs, proton):
    fit = roles == "fit"
    fit_prevalence = float(np.mean(y[fit]))
    xrs_rel = cf.family_reliability(frame, xrs)
    proton_rel = cf.family_reliability(frame, proton)

    folds = cf.build_inner_folds(frame, y, roles, units)
    oof_rows, oof_evidence, oof_labels, fold_receipts = [], [], [], []
    for fold in folds:
        tr, sc, prev = fold["train_idx"], fold["score_idx"], fold["train_prevalence"]
        ps = cf.specialist_predict(frame, base, y, tr, sc)
        px = cf.specialist_predict(frame, xrs, y, tr, sc)
        pp = cf.specialist_predict(frame, proton, y, tr, sc)
        oof_rows.append(sc)
        oof_labels.append(y[sc])
        oof_evidence.append(np.column_stack([
            cf.centered_evidence(ps, prev),
            cf.centered_evidence(px, prev, xrs_rel[sc]),
            cf.centered_evidence(pp, prev, proton_rel[sc]),
        ]))
        fold_receipts.append({k: val for k, val in fold.items() if k not in ("train_idx", "score_idx")})

    rows = np.concatenate(oof_rows)
    evidence = np.concatenate(oof_evidence, axis=0)
    labels = np.concatenate(oof_labels)
    order = np.argsort(rows)
    rows, evidence, labels = rows[order], evidence[order], labels[order]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("OOF rows overlap")

    teacher = PositiveEvidenceStack(EvidenceStackConfig(expert_count=3)).fit(evidence, labels)
    teacher_probability = public_benchmark.sigmoid(teacher.decision_function(evidence))

    students = {}
    for state in ONE_FEED_STATES:
        cols = evidence_columns(state)
        student = DistilledPositiveEvidenceStack(
            DistilledEvidenceStackConfig(
                expert_count=len(cols),
                teacher_weight=TEACHER_WEIGHT,
                l2_weight=L2_WEIGHT,
            )
        ).fit(evidence[:, cols], labels, teacher_probability)
        students[state] = student

    family_models = {
        "solar": transfer.fit_family_models(frame, base, y, fit),
        "xrs": transfer.fit_family_models(frame, xrs, y, fit),
        "proton": transfer.fit_family_models(frame, proton, y, fit),
    }
    raw = {
        "solar": transfer.predict_family(family_models["solar"], frame, base),
        "xrs": transfer.predict_family(family_models["xrs"], frame, xrs),
        "proton": transfer.predict_family(family_models["proton"], frame, proton),
    }
    full_evidence = np.column_stack([
        cf.centered_evidence(raw["solar"], fit_prevalence),
        cf.centered_evidence(raw["xrs"], fit_prevalence, xrs_rel),
        cf.centered_evidence(raw["proton"], fit_prevalence, proton_rel),
    ])

    raw_state = {
        "FULL": public_benchmark.sigmoid(teacher.decision_function(full_evidence)),
        "NO_XRS": public_benchmark.sigmoid(
            students["NO_XRS"].decision_function(full_evidence[:, evidence_columns("NO_XRS")])
        ),
        "NO_PROTON": public_benchmark.sigmoid(
            students["NO_PROTON"].decision_function(full_evidence[:, evidence_columns("NO_PROTON")])
        ),
        "NO_XRS_OR_PROTON": np.asarray(raw["solar"], dtype=float).copy(),
    }

    probability, calibration, thresholds = {}, {}, {}
    for state in ALL_STATES:
        probability[state], calibration[state] = cf.calibrated_probability(raw_state[state], y, roles)
        thresholds[state] = cf.thresholds(y, probability[state], roles)

    return {
        "fit_prevalence": fit_prevalence,
        "teacher": teacher,
        "students": students,
        "probability": probability,
        "calibration_intercept": calibration,
        "thresholds": thresholds,
        "oof_rows": int(len(rows)),
        "oof_positives": int(labels.sum()),
        "inner_folds": fold_receipts,
        "teacher_probability_min": float(np.min(teacher_probability)),
        "teacher_probability_max": float(np.max(teacher_probability)),
    }


def _keep_gate(v1_metrics: dict, v3_metrics: dict, v1_review: dict, v3_review: dict) -> dict:
    tss_delta = float(v3_metrics["TSS"] - v1_metrics["TSS"])
    brier_delta = float(v3_metrics["BRIER"] - v1_metrics["BRIER"])
    ece_delta = float(v3_metrics["ECE"] - v1_metrics["ECE"])
    v1_enrichment = v1_review["enrichment_vs_random_review"]
    v3_enrichment = v3_review["enrichment_vs_random_review"]
    enrichment_delta = None
    if v1_enrichment is not None and v3_enrichment is not None:
        enrichment_delta = float(v3_enrichment - v1_enrichment)
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
        "deltas_v3_minus_v1": {
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
        raise ValueError("distilled fallback V3 preregistration missing")

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
    if not score.any() or int(np.sum(y[score])) == 0:
        raise ValueError("event-bearing score role required")

    v1_model = v1_fallback._fit_all_states(frame, y, roles, units, base, xrs, proton)
    v3_model = _fit_distilled_states(frame, y, roles, units, base, xrs, proton)
    v1_probability = v1_model["probability"]
    v3_probability = v3_model["probability"]

    # The full-data model must be unchanged.  If this fails, the experiment has
    # changed more than the missing-feed fallback and is invalid.
    full_diff = float(np.max(np.abs(v1_probability["FULL"] - v3_probability["FULL"])))
    if full_diff > FULL_REPRO_TOLERANCE:
        raise ValueError(f"V3 changed the FULL forecast: max abs diff {full_diff}")

    prevalence = float(v1_model["fit_prevalence"])
    issue_score = frame.loc[score, "window_end"]
    labels_score = np.asarray(y)[score]
    state_results = {}
    keep_results = {}

    for state in ONE_FEED_STATES:
        v1_metrics = v1_fallback._whole_score_metrics(
            y,
            v1_probability[state],
            roles,
            v1_model["thresholds"][state][PRIMARY_POLICY],
            prevalence,
        )
        v3_metrics = v1_fallback._whole_score_metrics(
            y,
            v3_probability[state],
            roles,
            v3_model["thresholds"][state][PRIMARY_POLICY],
            prevalence,
        )
        v1_review = review.budget_metrics(
            labels_score,
            np.asarray(v1_probability[state])[score],
            issue_score,
            PRIMARY_REVIEW_FRACTION,
        )
        v3_review = review.budget_metrics(
            labels_score,
            np.asarray(v3_probability[state])[score],
            issue_score,
            PRIMARY_REVIEW_FRACTION,
        )
        gate = _keep_gate(v1_metrics, v3_metrics, v1_review, v3_review)
        keep_results[state] = gate
        state_results[state] = {
            "v1_scalar_fusion": {
                "metrics": v1_metrics,
                "review_5pct": v1_review,
                "stack_diagnostics": v1_model["stacks"][state].diagnostics(),
            },
            "v3_distilled_fusion": {
                "metrics": v3_metrics,
                "review_5pct": v3_review,
                "calibration_intercept": float(v3_model["calibration_intercept"][state]),
                "thresholds": v3_model["thresholds"][state],
                "stack_diagnostics": v3_model["students"][state].diagnostics(),
            },
            "keep_gate": gate,
        }

    candidate = all(keep_results[state]["passed"] for state in ONE_FEED_STATES)

    predictions = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
    })
    for state in ALL_STATES:
        predictions[f"p_v1_{state}"] = v1_probability[state]
        predictions[f"p_v3_{state}"] = v3_probability[state]
    predictions_path = output / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.17g")

    summary = {
        "status": "COMPLETED_DISTILLED_AVAILABILITY_FALLBACK_V3_DEVELOPMENT_ONLY",
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
        "teacher_weight": TEACHER_WEIGHT,
        "hard_label_weight": 1.0 - TEACHER_WEIGHT,
        "l2_weight": L2_WEIGHT,
        "full_state_max_abs_difference_vs_v1": full_diff,
        "full_state_reproduction_tolerance": FULL_REPRO_TOLERANCE,
        "primary_policy": PRIMARY_POLICY,
        "primary_review_fraction": PRIMARY_REVIEW_FRACTION,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric": dropped,
        "oof_rows": v3_model["oof_rows"],
        "oof_positives": v3_model["oof_positives"],
        "teacher_diagnostics": v3_model["teacher"].diagnostics(),
        "teacher_probability_range": [v3_model["teacher_probability_min"], v3_model["teacher_probability_max"]],
        "states": state_results,
        "candidate_for_future_untouched_evaluation": bool(candidate),
        "candidate_rule": "both one-feed states must pass the preregistered V3 keep gate; development cannot grant NORMAL trust",
        "claim_boundary": "Development-only architecture decision. Independent evidence is still required for NORMAL/VALID trust.",
    }
    summary_path = output / "summary.json"
    save_json(summary_path, summary)
    save_json(output / "receipt.json", {
        "status": "DISTILLED_FALLBACK_V3_RECEIPT",
        "summary_sha256": digest(summary_path),
        "predictions_sha256": digest(predictions_path),
        "locked_test_accessed": False,
        "monitor_used": False,
        "normal_status_granted": False,
    })
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
