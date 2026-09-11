"""Preregistered Phase II genuine-new-onset forecaster.

This is a development-only extension of the frozen IRIS-SEP benchmark.  It does
not modify the historical benchmark, inspect the protected post-2025 cohort, or
claim universal state of the art.  Architecture selection happens only inside
the outer fit role using expanding, purged, physical-unit cross-fit predictions.
Calibration, threshold selection, score, and monitor roles remain separated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from iris_report.iris_sep.src.iris_sep.modeling.two_state_features import build_two_state_features
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as static_cf

STUDY_ID = "IRIS_SEP_ONSET_FORECASTER_PHASE2_V1"
PREREGISTRATION = "config/onset_forecaster_phase2_preregistration_2026-09-11.json"
CANDIDATE_ORDER = (
    "BASE_SOLAR_EVENT_WEIGHTED",
    "BASE_PLUS_XRS_EVENT_WEIGHTED",
    "FULL_CONTEXT_EVENT_WEIGHTED",
    "BASE_SOLAR_TWO_STATE_EVENT_WEIGHTED",
    "BASE_PLUS_XRS_TWO_STATE_EVENT_WEIGHTED",
    "FULL_CONTEXT_TWO_STATE_EVENT_WEIGHTED",
)
REGULAR_BASELINES = (
    "BASE_SOLAR_REGULAR",
    "BASE_PLUS_XRS_REGULAR",
    "FULL_CONTEXT_REGULAR",
)
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_REPLICATES = 10000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return [finite(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def event_balanced_weights(y: np.ndarray, units: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Equalize physical units within class, then equalize total class weight.

    Each positive episode receives equal total mass; each quiet block receives
    equal total mass.  Positive and negative classes each receive half of the
    total mass.  Rows inside a unit split its unit mass equally.
    """
    idx = np.asarray(train_idx, dtype=int)
    if idx.ndim != 1 or len(idx) == 0 or len(np.unique(idx)) != len(idx):
        raise ValueError("train_idx must be unique, one-dimensional, and non-empty")
    yy = np.asarray(y, dtype=np.int8)[idx]
    uu = np.asarray(units, dtype=str)[idx]
    if len(np.unique(yy)) != 2:
        raise ValueError("weighted training subset must contain both classes")
    table = pd.DataFrame({"local": np.arange(len(idx)), "unit": uu, "label": yy})
    if (table.groupby("unit")["label"].nunique() > 1).any():
        raise ValueError("physical unit mixes labels")
    unit_label = table.groupby("unit", sort=False)["label"].first()
    class_units = {label: unit_label.index[unit_label == label].tolist() for label in (0, 1)}
    if not class_units[0] or not class_units[1]:
        raise ValueError("both unit classes are required")

    weights = np.zeros(len(idx), dtype=np.float64)
    for label in (0, 1):
        unit_mass = 0.5 / float(len(class_units[label]))
        for unit in class_units[label]:
            rows = table.loc[table["unit"] == unit, "local"].to_numpy(dtype=int)
            weights[rows] = unit_mass / float(len(rows))
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid event-balanced weights")
    weights *= float(len(weights)) / float(weights.sum())
    return weights


def weight_receipt(y: np.ndarray, units: np.ndarray, train_idx: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    idx = np.asarray(train_idx, dtype=int)
    yy = np.asarray(y, dtype=np.int8)[idx]
    uu = np.asarray(units, dtype=str)[idx]
    result: dict[str, object] = {
        "rows": int(len(idx)),
        "mean_weight": float(np.mean(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "class_weight_sums": {},
        "unit_total_mass_range_by_class": {},
    }
    for label in (0, 1):
        mask = yy == label
        result["class_weight_sums"][str(label)] = float(np.sum(weights[mask]))
        totals = []
        for unit in np.unique(uu[mask]):
            totals.append(float(np.sum(weights[(uu == unit) & mask])))
        result["unit_total_mass_range_by_class"][str(label)] = [float(min(totals)), float(max(totals))]
    return result


def _weighted_xgb(train_x, train_y, predict_x, sample_weight, seed: int) -> np.ndarray:
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=1.0,
        n_jobs=2,
        random_state=int(seed),
    )
    model.fit(train_x, train_y, sample_weight=sample_weight, verbose=False)
    return model.predict_proba(predict_x)[:, 1]


def ensemble_predict(
    features: pd.DataFrame,
    y: np.ndarray,
    units: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    *,
    event_weighted: bool,
) -> np.ndarray:
    train_idx = np.asarray(train_idx, dtype=int)
    predict_idx = np.asarray(predict_idx, dtype=int)
    predictions = []
    if event_weighted:
        weights = event_balanced_weights(y, units, train_idx)
        for seed in v1.SEEDS:
            predictions.append(
                _weighted_xgb(
                    features.iloc[train_idx],
                    y[train_idx],
                    features.iloc[predict_idx],
                    weights,
                    int(seed),
                )
            )
    else:
        for seed in v1.SEEDS:
            predictions.append(
                v1.fit_xgb(
                    features.iloc[train_idx],
                    y[train_idx],
                    features.iloc[predict_idx],
                    int(seed),
                )
            )
    return np.median(np.stack(predictions, axis=0), axis=0)


def _thresholds(y: np.ndarray, p: np.ndarray, roles: np.ndarray) -> dict[str, float]:
    mask = roles == "threshold"
    pod80 = v1.minimum_far_at_pod(y[mask], p[mask], 0.8)
    if pod80 is None:
        raise ValueError("POD80 threshold unavailable")
    return {
        "MAX_TSS": float(v1.select_threshold(y[mask], p[mask])),
        "POD80_MIN_FAR": float(pod80["threshold"]),
    }


def _calibrate(raw: np.ndarray, y: np.ndarray, roles: np.ndarray) -> tuple[np.ndarray, float]:
    mask = roles == "calibration"
    intercept = float(v1.fit_intercept(raw[mask], y[mask]))
    probability = v1.sigmoid(v1.logit(raw) + intercept)
    return probability, intercept


def _evaluate(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    roles: np.ndarray,
    role: str,
    prevalence: float,
) -> dict[str, object]:
    mask = roles == role
    if int(mask.sum()) == 0 or len(np.unique(y[mask])) != 2:
        raise ValueError(f"role {role} lacks both classes")
    return {
        **v1.threshold_metrics(y[mask], p[mask], threshold),
        **v1.probability_metrics(y[mask], p[mask], prevalence),
        "matched_detection": {
            str(pod): v1.minimum_far_at_pod(y[mask], p[mask], pod)
            for pod in (0.6, 0.7, 0.8, 0.9)
        },
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
        "units": int(len(np.unique(np.asarray(roles)[mask]))),
    }


def build_feature_frames(frame: pd.DataFrame, base: list[str], xrs: list[str], proton: list[str]):
    base_xrs = list(base) + list(xrs)
    full = list(base) + list(xrs) + list(proton)
    for name in full:
        low = name.lower()
        if low.startswith("future_") or name in {"OSEP_label", "GSEP_label"}:
            raise ValueError(f"forbidden learned feature: {name}")
    current = {
        "BASE_SOLAR": frame.loc[:, base].apply(pd.to_numeric, errors="coerce"),
        "BASE_PLUS_XRS": frame.loc[:, base_xrs].apply(pd.to_numeric, errors="coerce"),
        "FULL_CONTEXT": frame.loc[:, full].apply(pd.to_numeric, errors="coerce"),
    }
    two_state = {}
    receipts = {}
    for key, names in (("BASE_SOLAR", base), ("BASE_PLUS_XRS", base_xrs), ("FULL_CONTEXT", full)):
        two_state[key], receipts[key] = build_two_state_features(frame, names)
    return current, two_state, receipts


def inner_select(
    frame: pd.DataFrame,
    y: np.ndarray,
    units: np.ndarray,
    roles: np.ndarray,
    current: dict[str, pd.DataFrame],
    two_state: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    folds = static_cf.build_inner_folds(frame, y, roles, units)
    specs = {
        "BASE_SOLAR_EVENT_WEIGHTED": current["BASE_SOLAR"],
        "BASE_PLUS_XRS_EVENT_WEIGHTED": current["BASE_PLUS_XRS"],
        "FULL_CONTEXT_EVENT_WEIGHTED": current["FULL_CONTEXT"],
        "BASE_SOLAR_TWO_STATE_EVENT_WEIGHTED": two_state["BASE_SOLAR"],
        "BASE_PLUS_XRS_TWO_STATE_EVENT_WEIGHTED": two_state["BASE_PLUS_XRS"],
        "FULL_CONTEXT_TWO_STATE_EVENT_WEIGHTED": two_state["FULL_CONTEXT"],
    }
    collected_rows: list[np.ndarray] = []
    collected_y: list[np.ndarray] = []
    collected: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATE_ORDER}
    fold_receipts = []
    for fold in folds:
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        score_idx = np.asarray(fold["score_idx"], dtype=int)
        collected_rows.append(score_idx)
        collected_y.append(y[score_idx])
        for name in CANDIDATE_ORDER:
            collected[name].append(
                ensemble_predict(specs[name], y, units, train_idx, score_idx, event_weighted=True)
            )
        fold_receipts.append({
            **{k: v for k, v in fold.items() if k not in ("train_idx", "score_idx")},
            "event_weight_receipt": weight_receipt(
                y, units, train_idx, event_balanced_weights(y, units, train_idx)
            ),
        })

    rows = np.concatenate(collected_rows)
    yy = np.concatenate(collected_y)
    order = np.argsort(rows)
    rows = rows[order]
    yy = yy[order]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("inner cross-fit score rows overlap")

    diagnostics: dict[str, object] = {}
    ranking = []
    prevalence = float(np.mean(y[roles == "fit"]))
    for candidate_index, name in enumerate(CANDIDATE_ORDER):
        probability = np.concatenate(collected[name])[order]
        threshold = float(v1.select_threshold(yy, probability))
        metrics = {
            **v1.threshold_metrics(yy, probability, threshold),
            **v1.probability_metrics(yy, probability, prevalence),
            "threshold": threshold,
            "rows": int(len(yy)),
            "positives": int(yy.sum()),
        }
        diagnostics[name] = metrics
        far = float(metrics["FAR"]) if math.isfinite(float(metrics["FAR"])) else 1.0
        hss = float(metrics["HSS"]) if math.isfinite(float(metrics["HSS"])) else -1.0
        ranking.append((float(metrics["TSS"]), -far, hss, -candidate_index, name))
    selected = max(ranking)[-1]
    return selected, diagnostics, fold_receipts


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    if digest(features) != v1.EXPECTED_FEATURE_SHA256:
        raise ValueError("feature-table hash mismatch")
    if digest(events) != v1.EXPECTED_EVENT_SHA256:
        raise ValueError("event-catalogue hash mismatch")

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    fit_idx = np.flatnonzero(roles == "fit")
    if len(np.unique(y[fit_idx])) != 2:
        raise ValueError("outer fit role lacks both classes")
    fit_prevalence = float(np.mean(y[fit_idx]))

    current, two_state, state_receipts = build_feature_frames(frame, base, xrs, proton)
    selected, inner_diagnostics, inner_folds = inner_select(
        frame, y, units, roles, current, two_state
    )

    selected_specs = {
        "BASE_SOLAR_EVENT_WEIGHTED": current["BASE_SOLAR"],
        "BASE_PLUS_XRS_EVENT_WEIGHTED": current["BASE_PLUS_XRS"],
        "FULL_CONTEXT_EVENT_WEIGHTED": current["FULL_CONTEXT"],
        "BASE_SOLAR_TWO_STATE_EVENT_WEIGHTED": two_state["BASE_SOLAR"],
        "BASE_PLUS_XRS_TWO_STATE_EVENT_WEIGHTED": two_state["BASE_PLUS_XRS"],
        "FULL_CONTEXT_TWO_STATE_EVENT_WEIGHTED": two_state["FULL_CONTEXT"],
    }
    baseline_specs = {
        "BASE_SOLAR_REGULAR": current["BASE_SOLAR"],
        "BASE_PLUS_XRS_REGULAR": current["BASE_PLUS_XRS"],
        "FULL_CONTEXT_REGULAR": current["FULL_CONTEXT"],
    }

    raw: dict[str, np.ndarray] = {}
    all_idx = np.arange(len(frame), dtype=int)
    for name, feature_frame in baseline_specs.items():
        raw[name] = ensemble_predict(
            feature_frame, y, units, fit_idx, all_idx, event_weighted=False
        )
    raw[selected] = ensemble_predict(
        selected_specs[selected], y, units, fit_idx, all_idx, event_weighted=True
    )

    probability: dict[str, np.ndarray] = {}
    calibration: dict[str, float] = {}
    thresholds: dict[str, dict[str, float]] = {}
    model_names = [*REGULAR_BASELINES, selected]
    for name in model_names:
        probability[name], calibration[name] = _calibrate(raw[name], y, roles)
        thresholds[name] = _thresholds(y, probability[name], roles)

    summary: dict[str, object] = {
        "study_id": STUDY_ID,
        "status": "PHASE2_RESULT_PENDING_GATE",
        "scope": "DEVELOPMENT_ONLY_PUBLIC_HISTORICAL_DATA",
        "target": v1.TARGET,
        "preregistration": PREREGISTRATION,
        "locked_test_accessed": False,
        "protected_post_2025_outcomes_accessed": False,
        "score_and_monitor_prior_exposure_disclosed": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric_columns": dropped,
        "feature_family_sizes": {
            "base": int(len(base)),
            "xrs": int(len(xrs)),
            "proton": int(len(proton)),
        },
        "two_state_receipts": state_receipts,
        "outer_fit_weight_receipt": weight_receipt(
            y, units, fit_idx, event_balanced_weights(y, units, fit_idx)
        ),
        "architecture_selection": {
            "selected": selected,
            "candidate_order": list(CANDIDATE_ORDER),
            "selection_used_only_outer_fit_crossfit": True,
            "score_or_monitor_used_for_selection": False,
            "inner_candidate_metrics": inner_diagnostics,
            "inner_folds": inner_folds,
        },
        "models": {},
        "paired_comparisons": {},
    }

    for name in model_names:
        summary["models"][name] = {
            "calibration_intercept": calibration[name],
            "thresholds": thresholds[name],
            "policies": {},
        }
        for policy in POLICIES:
            threshold = thresholds[name][policy]
            summary["models"][name]["policies"][policy] = {
                "score": _evaluate(y, probability[name], threshold, roles, "score", fit_prevalence),
                "monitor": _evaluate(y, probability[name], threshold, roles, "monitor", fit_prevalence),
            }

    for baseline in REGULAR_BASELINES:
        for policy in POLICIES:
            for role in ("score", "monitor"):
                key = f"{selected}_minus_{baseline}_{policy}_{role}"
                summary["paired_comparisons"][key] = cs.bootstrap_difference(
                    y,
                    probability[selected], thresholds[selected][policy],
                    probability[baseline], thresholds[baseline][policy],
                    units, roles, role,
                    seed=BOOTSTRAP_SEED,
                    replicates=BOOTSTRAP_REPLICATES,
                )

    # Conservative primary comparator: whichever regular baseline has the best
    # already-evaluated score TSS.  This choice can only make the candidate's
    # superiority gate harder to pass and never changes the selected candidate.
    best_baseline = max(
        REGULAR_BASELINES,
        key=lambda name: float(summary["models"][name]["policies"]["MAX_TSS"]["score"]["TSS"]),
    )
    candidate_score = summary["models"][selected]["policies"]["MAX_TSS"]["score"]
    baseline_score = summary["models"][best_baseline]["policies"]["MAX_TSS"]["score"]
    comparison_key = f"{selected}_minus_{best_baseline}_MAX_TSS_score"
    comparison = summary["paired_comparisons"][comparison_key]
    point_delta = float(candidate_score["TSS"] - baseline_score["TSS"])
    gate_checks = {
        "TSS_delta_point_gt_zero": bool(point_delta > 0.0),
        "TSS_delta_ci95_lower_gt_zero": bool(float(comparison["ci_lower_95"]) > 0.0),
        "HSS_candidate_gt_baseline": bool(float(candidate_score["HSS"]) > float(baseline_score["HSS"])),
        "FAR_candidate_lt_baseline": bool(float(candidate_score["FAR"]) < float(baseline_score["FAR"])),
    }
    gate_passed = bool(all(gate_checks.values()))
    summary["primary_comparison"] = {
        "candidate": selected,
        "baseline": best_baseline,
        "policy": "MAX_TSS",
        "role": "score",
        "point_TSS_delta": point_delta,
        "bootstrap": comparison,
        "checks": gate_checks,
        "passed": gate_passed,
    }
    summary["status"] = (
        "PHASE2_DEVELOPMENT_SUPERIORITY_GATE_PASSED"
        if gate_passed
        else "NO_DEMONSTRATED_PHASE2_SUPERIORITY"
    )

    predictions = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
    })
    for name in model_names:
        predictions[name] = probability[name]
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "study_id": STUDY_ID,
        "status": summary["status"],
        "preregistration": PREREGISTRATION,
        "selected_candidate": selected,
        "locked_test_accessed": False,
        "protected_post_2025_outcomes_accessed": False,
        "frozen_benchmark_modified": False,
        "score_or_monitor_used_for_architecture_selection": False,
        "post_score_candidate_switching": False,
        "historical_result_is_development_only": True,
        "universal_state_of_the_art_claim_authorized": False,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
