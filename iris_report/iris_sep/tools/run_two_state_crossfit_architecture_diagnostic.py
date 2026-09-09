"""One-shot two-state specialist architecture diagnostic for IRIS-SEP.

The specialist state is [current 24 h, previous 24 h, delta].  Cross-fitting,
meta-fusion, calibration and threshold semantics are frozen from the promoted
static cross-fitted evidence stack. Score and monitor remain development-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.positive_evidence_stack import PositiveEvidenceStack
from iris_report.iris_sep.src.iris_sep.modeling.two_state_features import build_two_state_features
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as static_cf

TEMPORAL_NAME = "IRIS_TWO_STATE_CROSSFIT_STACK_V1"
STATIC_COMPARATORS = (
    "BASE_SOLAR",
    "LATE_FUSION_SOLAR_XRS_PROTON",
    "IRIS_CROSSFIT_EVIDENCE_STACK_V1",
)
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value):
    if isinstance(value, dict): return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray): return [finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic): return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def specialist_predict(feature_frame: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, predict_idx: np.ndarray) -> np.ndarray:
    predictions = []
    for seed in v1.SEEDS:
        predictions.append(v1.fit_xgb(feature_frame.loc[train_idx], y[train_idx], feature_frame.loc[predict_idx], seed))
    return np.median(np.stack(predictions, axis=0), axis=0)


def full_specialist(feature_frame: pd.DataFrame, y: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    predictions = []
    for seed in v1.SEEDS:
        predictions.append(v1.fit_xgb(feature_frame.loc[fit_mask], y[fit_mask], feature_frame, seed))
    return np.median(np.stack(predictions, axis=0), axis=0)


def calibrated_probability(raw_probability: np.ndarray, y: np.ndarray, roles: np.ndarray):
    cal = roles == "calibration"
    intercept = v1.fit_intercept(raw_probability[cal], y[cal])
    return v1.sigmoid(v1.logit(raw_probability) + intercept), float(intercept)


def thresholds(y, p, roles):
    mask = roles == "threshold"
    pod80 = v1.minimum_far_at_pod(y[mask], p[mask], 0.8)
    if pod80 is None: raise ValueError("POD80 threshold unavailable")
    return {"MAX_TSS": float(v1.select_threshold(y[mask], p[mask])), "POD80_MIN_FAR": float(pod80["threshold"])}


def evaluate(y, p, threshold, roles, role, prevalence):
    mask = roles == role
    return {
        **v1.threshold_metrics(y[mask], p[mask], threshold),
        **v1.probability_metrics(y[mask], p[mask], prevalence),
        "matched_detection": {str(pod): v1.minimum_far_at_pod(y[mask], p[mask], pod) for pod in (0.6, 0.7, 0.8, 0.9)},
        "rows": int(mask.sum()), "positives": int(y[mask].sum()),
    }


def load_static_predictions(path: Path, frame: pd.DataFrame, y: np.ndarray, roles: np.ndarray, units: np.ndarray):
    static = pd.read_csv(path)
    if len(static) != len(frame): raise ValueError("static comparator row count mismatch")
    if not np.array_equal(static["label"].to_numpy(dtype=int), y): raise ValueError("static comparator label mismatch")
    if not np.array_equal(static["role"].astype(str).to_numpy(), roles.astype(str)): raise ValueError("static comparator role mismatch")
    if not np.array_equal(static["unit_id"].fillna("").astype(str).to_numpy(), units.astype(str)): raise ValueError("static comparator unit mismatch")
    left = pd.DatetimeIndex(pd.to_datetime(static["issue_time"], utc=True, errors="raise"))
    right = pd.DatetimeIndex(pd.to_datetime(frame["window_end"], utc=True, errors="raise"))
    if not left.equals(right): raise ValueError("static comparator issue-time mismatch")
    missing = [name for name in STATIC_COMPARATORS if name not in static.columns]
    if missing: raise ValueError(f"static comparator columns missing: {missing}")
    return {name: static[name].to_numpy(dtype=float) for name in STATIC_COMPARATORS}


def run(features: Path, events: Path, static_predictions: Path, output: Path):
    output = Path(output)
    if output.exists(): raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    fit = roles == "fit"; cal = roles == "calibration"
    fit_prevalence = float(np.mean(y[fit]))

    static_probability = load_static_predictions(static_predictions, frame, y, roles, units)

    solar_state, solar_state_receipt = build_two_state_features(frame, base)
    xrs_state, xrs_state_receipt = build_two_state_features(frame, xrs)
    proton_state, proton_state_receipt = build_two_state_features(frame, proton)
    xrs_rel = static_cf.family_reliability(frame, xrs)
    proton_rel = static_cf.family_reliability(frame, proton)

    folds = static_cf.build_inner_folds(frame, y, roles, units)
    oof_rows = []
    oof_evidence = []
    oof_labels = []
    fold_receipts = []
    for fold in folds:
        tr = fold["train_idx"]; sc = fold["score_idx"]; prevalence = fold["train_prevalence"]
        ps = specialist_predict(solar_state, y, tr, sc)
        px = specialist_predict(xrs_state, y, tr, sc)
        pp = specialist_predict(proton_state, y, tr, sc)
        evidence = np.column_stack([
            static_cf.centered_evidence(ps, prevalence),
            static_cf.centered_evidence(px, prevalence, xrs_rel[sc]),
            static_cf.centered_evidence(pp, prevalence, proton_rel[sc]),
        ])
        oof_rows.append(sc); oof_evidence.append(evidence); oof_labels.append(y[sc])
        fold_receipts.append({k: v for k, v in fold.items() if k not in ("train_idx", "score_idx")})

    oof_rows = np.concatenate(oof_rows)
    oof_evidence = np.concatenate(oof_evidence, axis=0)
    oof_labels = np.concatenate(oof_labels)
    order = np.argsort(oof_rows)
    oof_rows = oof_rows[order]; oof_evidence = oof_evidence[order]; oof_labels = oof_labels[order]
    if len(np.unique(oof_rows)) != len(oof_rows): raise ValueError("OOF rows overlap")

    stack = PositiveEvidenceStack().fit(oof_evidence, oof_labels)

    raw_solar = full_specialist(solar_state, y, fit)
    raw_xrs = full_specialist(xrs_state, y, fit)
    raw_proton = full_specialist(proton_state, y, fit)
    full_evidence = np.column_stack([
        static_cf.centered_evidence(raw_solar, fit_prevalence),
        static_cf.centered_evidence(raw_xrs, fit_prevalence, xrs_rel),
        static_cf.centered_evidence(raw_proton, fit_prevalence, proton_rel),
    ])
    raw_stack_probability = v1.sigmoid(stack.decision_function(full_evidence))
    temporal_probability, calibration_intercept = calibrated_probability(raw_stack_probability, y, roles)

    probability = dict(static_probability)
    probability[TEMPORAL_NAME] = temporal_probability
    model_thresholds = {name: thresholds(y, p, roles) for name, p in probability.items()}

    summary = {
        "status": "COMPLETED_TWO_STATE_CROSSFIT_ARCHITECTURE_DEVELOPMENT_DIAGNOSTIC",
        "target": v1.TARGET,
        "locked_test_accessed": False,
        "score_and_monitor_already_inspected": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "static_predictions_sha256": digest(static_predictions),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric_columns": dropped,
        "state_receipts": {"solar": solar_state_receipt, "xrs": xrs_state_receipt, "proton": proton_state_receipt},
        "temporal_model_metadata": {
            **stack.diagnostics(),
            "post_stack_calibration_intercept": calibration_intercept,
            "oof_rows": int(len(oof_rows)),
            "oof_positives": int(oof_labels.sum()),
            "inner_folds": fold_receipts,
        },
        "models": {},
        "paired_comparisons": {},
    }

    for name, p in probability.items():
        summary["models"][name] = {"thresholds": model_thresholds[name], "policies": {}}
        for policy in POLICIES:
            t = model_thresholds[name][policy]
            summary["models"][name]["policies"][policy] = {
                "score": evaluate(y, p, t, roles, "score", fit_prevalence),
                "monitor": evaluate(y, p, t, roles, "monitor", fit_prevalence),
            }

    for policy in POLICIES:
        for role in ("score", "monitor"):
            role_mask = np.where(roles == role, role, "outside")
            for other in STATIC_COMPARATORS:
                key = f"{TEMPORAL_NAME}_minus_{other}_{policy}_{role}"
                summary["paired_comparisons"][key] = cs.bootstrap_difference(
                    y, probability[TEMPORAL_NAME], model_thresholds[TEMPORAL_NAME][policy],
                    probability[other], model_thresholds[other][policy],
                    units, role_mask, role, seed=20260906, replicates=10000,
                )

    rows = pd.DataFrame({"issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "role": roles, "unit_id": units, "label": y})
    for name, p in probability.items(): rows[name] = p
    rows.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": "DEVELOPMENT_ONLY_TWO_STATE_CROSSFIT_RUN",
        "preregistration": "config/two_state_crossfit_architecture_preregistration_2026-09-06.json",
        "locked_test_accessed": False,
        "temporal_lags_used": [24],
        "post_result_hyperparameter_changes": False,
        "static_comparator_predictions_reused_immutably": True,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--static-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.features, args.events, args.static_predictions, args.output)


if __name__ == "__main__": main()
