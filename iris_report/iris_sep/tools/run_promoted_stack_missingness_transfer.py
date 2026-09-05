"""Transfer the frozen missingness experiment to the promoted cross-fitted stack.

Development-only. The score block has been inspected previously. The model,
calibration and thresholds are fitted once on clean data and frozen before any
synthetic outage is applied. All three previously tested recovery arms and all
three previously tested outage fractions are retained by preregistration.
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

from iris_report.iris_sep.src.iris_sep.modeling.positive_evidence_stack import PositiveEvidenceStack
from iris_report.iris_sep.src.iris_sep.missingness_experiment import (
    deterministic_transient_random_holdout,
    recover_causal_forward_fill,
    recover_train_median,
)
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1


FRACTIONS = (0.05, 0.20, 0.40)
SEED = 20260906
ARMS = ("MASK_AWARE_NO_FILL", "TRAIN_FIT_MEDIAN", "CAUSAL_FORWARD_FILL")
PREREG = "config/promoted_stack_missingness_transfer_preregistration_2026-09-06.json"


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


def make_xgb(train_y, seed: int) -> XGBClassifier:
    prevalence = float(np.mean(train_y))
    return XGBClassifier(
        n_estimators=500,
        learning_rate=.03,
        max_depth=3,
        min_child_weight=5,
        subsample=.8,
        colsample_bytree=.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=(1 - prevalence) / prevalence,
        n_jobs=2,
        random_state=int(seed),
    )


def fit_family_models(frame, names, y, fit_mask):
    models = []
    train_y = y[fit_mask]
    for seed in v1.SEEDS:
        model = make_xgb(train_y, int(seed))
        model.fit(frame.loc[fit_mask, names], train_y, verbose=False)
        models.append(model)
    return models


def predict_family(models, frame, names):
    return np.median(
        np.stack([m.predict_proba(frame.loc[:, names])[:, 1] for m in models], axis=0),
        axis=0,
    )


def build_clean_model(frame, y, roles, units, base, xrs, proton):
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
    stack = PositiveEvidenceStack().fit(evidence, labels)

    models = {
        "solar": fit_family_models(frame, base, y, fit),
        "xrs": fit_family_models(frame, xrs, y, fit),
        "proton": fit_family_models(frame, proton, y, fit),
    }
    raw = {
        "solar": predict_family(models["solar"], frame, base),
        "xrs": predict_family(models["xrs"], frame, xrs),
        "proton": predict_family(models["proton"], frame, proton),
    }
    full_evidence = np.column_stack([
        cf.centered_evidence(raw["solar"], fit_prevalence),
        cf.centered_evidence(raw["xrs"], fit_prevalence, xrs_rel),
        cf.centered_evidence(raw["proton"], fit_prevalence, proton_rel),
    ])
    raw_stack = v1.sigmoid(stack.decision_function(full_evidence))
    clean_probability, intercept = cf.calibrated_probability(raw_stack, y, roles)
    policy_thresholds = cf.thresholds(y, clean_probability, roles)
    return {
        "stack": stack,
        "models": models,
        "fit_prevalence": fit_prevalence,
        "calibration_intercept": float(intercept),
        "thresholds": policy_thresholds,
        "clean_probability": clean_probability,
        "folds": fold_receipts,
        "oof_rows": int(len(rows)),
        "oof_positives": int(labels.sum()),
    }


def candidate_probability(clean_model, frame, base, xrs, proton):
    raw_solar = predict_family(clean_model["models"]["solar"], frame, base)
    raw_xrs = predict_family(clean_model["models"]["xrs"], frame, xrs)
    raw_proton = predict_family(clean_model["models"]["proton"], frame, proton)
    xrs_rel = cf.family_reliability(frame, xrs)
    proton_rel = cf.family_reliability(frame, proton)
    evidence = np.column_stack([
        cf.centered_evidence(raw_solar, clean_model["fit_prevalence"]),
        cf.centered_evidence(raw_xrs, clean_model["fit_prevalence"], xrs_rel),
        cf.centered_evidence(raw_proton, clean_model["fit_prevalence"], proton_rel),
    ])
    raw_stack = v1.sigmoid(clean_model["stack"].decision_function(evidence))
    return v1.sigmoid(v1.logit(raw_stack) + clean_model["calibration_intercept"])


def score_metrics(y, p, roles, threshold, prevalence):
    score = roles == "score"
    return {
        **v1.threshold_metrics(y[score], p[score], threshold),
        **v1.probability_metrics(y[score], p[score], prevalence),
        "matched_detection": {
            str(pod): v1.minimum_far_at_pod(y[score], p[score], pod)
            for pod in (0.6, 0.7, 0.8, 0.9)
        },
    }


def bootstrap_probability_differences(y, reference, candidate, units, roles, replicates=10000):
    score = roles == "score"
    yy, rr, cc, uu = y[score], reference[score], candidate[score], units[score]
    unique = np.unique(uu)
    row_map = {u: np.flatnonzero(uu == u) for u in unique}
    rng = np.random.default_rng(SEED)
    brier, drift = [], []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([row_map[u] for u in sampled])
        brier.append(float(np.mean((cc[ix] - yy[ix]) ** 2) - np.mean((rr[ix] - yy[ix]) ** 2)))
        drift.append(float(np.mean(np.abs(cc[ix] - rr[ix]))))
    brier = np.asarray(brier); drift = np.asarray(drift)
    return {
        "valid_replicates": int(replicates),
        "brier_delta_median": float(np.median(brier)),
        "brier_delta_ci95": [float(np.quantile(brier, .025)), float(np.quantile(brier, .975))],
        "probability_abs_drift_median": float(np.median(drift)),
        "probability_abs_drift_ci95": [float(np.quantile(drift, .025)), float(np.quantile(drift, .975))],
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists(): raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    score = roles == "score"
    if np.any(roles[score] == "monitor"):
        raise ValueError("monitor leaked into score")

    feature_names = list(base) + list(xrs) + list(proton)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature families overlap")
    raw_values = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    observed = np.isfinite(raw_values)
    structural = ~observed
    fit_rows = roles == "fit"

    clean = build_clean_model(frame, y, roles, units, base, xrs, proton)
    reference = clean["clean_probability"]
    clean_metrics = {
        policy: score_metrics(y, reference, roles, threshold, clean["fit_prevalence"])
        for policy, threshold in clean["thresholds"].items()
    }

    summary = {
        "status": "COMPLETED_PROMOTED_STACK_MISSINGNESS_TRANSFER_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": digest(Path(__file__).resolve().parents[1] / PREREG),
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_block_prior_inspection_disclosed": True,
        "source_hashes": {"features": digest(features), "events": digest(events)},
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "feature_count": int(len(feature_names)),
        "observed_cells": int(observed.sum()),
        "preexisting_unavailable_cells": int(structural.sum()),
        "model": {
            "id": "IRIS_CROSSFIT_EVIDENCE_STACK_V1",
            "stack_diagnostics": clean["stack"].diagnostics(),
            "post_stack_calibration_intercept": clean["calibration_intercept"],
            "thresholds": clean["thresholds"],
            "oof_rows": clean["oof_rows"],
            "oof_positives": clean["oof_positives"],
            "inner_folds": clean["folds"],
            "clean_score_metrics": clean_metrics,
        },
        "fractions": {},
    }

    prediction_columns = {
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
        "reference": reference,
    }

    for fraction in FRACTIONS:
        holdout = deterministic_transient_random_holdout(
            observed, structural, missing_fraction=fraction, seed=SEED, row_eligibility=score
        )
        experimental = observed & ~holdout
        no_fill_values = raw_values.copy()
        no_fill_values[holdout] = np.nan
        median = recover_train_median(
            raw_values, observed, structural, holdout, fit_rows=fit_rows
        )
        forward = recover_causal_forward_fill(raw_values, observed, structural, holdout)
        arm_values = {
            "MASK_AWARE_NO_FILL": no_fill_values,
            "TRAIN_FIT_MEDIAN": median.values,
            "CAUSAL_FORWARD_FILL": forward.values,
        }
        arm_available = {
            "MASK_AWARE_NO_FILL": experimental,
            "TRAIN_FIT_MEDIAN": median.available_mask,
            "CAUSAL_FORWARD_FILL": forward.available_mask,
        }
        fraction_result = {
            "fraction": float(fraction),
            "held_out_cells": int(holdout.sum()),
            "eligible_score_observed_cells": int((observed & score[:, None]).sum()),
            "arms": {},
        }
        tag = str(fraction).replace(".", "_")
        for arm in ARMS:
            candidate_frame = frame.copy()
            values = arm_values[arm].copy()
            # The availability mask is authoritative. Any unresolved cell stays NaN.
            values[~arm_available[arm]] = np.nan
            candidate_frame.loc[:, feature_names] = values
            p = candidate_probability(clean, candidate_frame, base, xrs, proton)
            arm_result = {
                "coverage": 1.0,
                "unresolved_heldout_cells": int((holdout & ~arm_available[arm]).sum()),
                "probability_abs_drift": float(np.mean(np.abs(p[score] - reference[score]))),
                "policies": {},
                "probability_bootstrap": bootstrap_probability_differences(y, reference, p, units, roles),
            }
            for policy, threshold in clean["thresholds"].items():
                candidate_metrics = score_metrics(y, p, roles, threshold, clean["fit_prevalence"])
                reference_metrics = clean_metrics[policy]
                role_mask = np.where(score, "score", "outside")
                tss_boot = cs.bootstrap_difference(
                    y, p, threshold, reference, threshold, units, role_mask, "score",
                    seed=SEED, replicates=10000,
                )
                arm_result["policies"][policy] = {
                    "reference": reference_metrics,
                    "candidate": candidate_metrics,
                    "delta_candidate_minus_reference": {
                        "TSS": float(candidate_metrics["TSS"] - reference_metrics["TSS"]),
                        "BRIER": float(candidate_metrics["BRIER"] - reference_metrics["BRIER"]),
                        "ECE": float(candidate_metrics["ECE"] - reference_metrics["ECE"]),
                    },
                    "paired_TSS_bootstrap": tss_boot,
                }
            fraction_result["arms"][arm] = arm_result
            prediction_columns[f"p_{tag}_{arm}"] = p
        summary["fractions"][str(fraction)] = fraction_result

    predictions = pd.DataFrame(prediction_columns)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": summary["status"],
        "preregistration": PREREG,
        "preregistration_sha256": summary["preregistration_sha256"],
        "predictions_sha256": summary["predictions_sha256"],
        "locked_test_accessed": False,
        "monitor_used": False,
        "all_predeclared_fractions_reported": True,
        "all_predeclared_arms_reported": True,
        "retraining_after_outage": False,
        "recalibration_after_outage": False,
        "rethresholding_after_outage": False,
        "claim_boundary": "Development-only transfer on already-inspected score identities; no final robustness or superiority claim.",
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__": main()
