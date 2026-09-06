"""Cross-fitted IRIS-SEP specialist evidence stack diagnostic.

Development-only: score/monitor blocks were previously inspected. Fusion weights
are learned only from out-of-fold predictions inside the outer fit role. The
calibration role is used only for one final intercept and the threshold role is
used only for decision thresholds. Locked test is never accessed.
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
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs


FOLD_BOUNDARIES = (0.40, 0.55, 0.70, 0.85, 1.00)
EVIDENCE_LIMIT = 6.0
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
MODELS = ("BASE_SOLAR", "LATE_FUSION_SOLAR_XRS_PROTON", "IRIS_CROSSFIT_EVIDENCE_STACK_V1")


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


def family_reliability(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.mean(np.isfinite(values), axis=1).astype(np.float64)


def centered_evidence(probability, prevalence: float, reliability=None) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    climate = math.log(prevalence) - math.log1p(-prevalence)
    evidence = np.clip(v1.logit(p) - climate, -EVIDENCE_LIMIT, EVIDENCE_LIMIT)
    if reliability is not None:
        r = np.asarray(reliability, dtype=np.float64)
        if r.shape != evidence.shape or ((r < 0) | (r > 1) | ~np.isfinite(r)).any():
            raise ValueError("invalid reliability")
        evidence = evidence * r
    return evidence


def specialist_predict(frame, names, y, train_idx, predict_idx):
    preds = []
    train_y = y[train_idx]
    for seed in v1.SEEDS:
        preds.append(v1.fit_xgb(frame.loc[train_idx, names], train_y, frame.loc[predict_idx, names], seed))
    return np.median(np.stack(preds, axis=0), axis=0)


def build_inner_folds(frame: pd.DataFrame, y: np.ndarray, roles: np.ndarray, units: np.ndarray):
    fit_idx = np.flatnonzero(roles == "fit")
    table = pd.DataFrame({
        "row": fit_idx,
        "unit_id": units[fit_idx],
        "time": frame.loc[fit_idx, "window_end"].to_numpy(),
        "label": y[fit_idx],
    })
    summary = (
        table.groupby("unit_id", sort=False)
        .agg(start=("time", "min"), end=("time", "max"), label=("label", "max"))
        .sort_values(["start", "end"])
    )
    if (table.groupby("unit_id")["label"].nunique() > 1).any():
        raise ValueError("inner unit mixes labels")
    n_positive = int(summary["label"].sum())
    if n_positive < 40:
        raise ValueError("too few fit positive units for four expanding folds")
    cuts = [int(math.floor(n_positive * f)) for f in FOLD_BOUNDARIES[:-1]] + [n_positive]
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("inner positive cuts are not strictly increasing")

    before_positive = {}
    seen = 0
    for uid, row in summary.iterrows():
        before_positive[str(uid)] = seen
        seen += int(row.label)

    folds = []
    for fold_id, (low, high) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        train_units = [u for u in summary.index.astype(str) if before_positive[u] < low]
        score_units = [u for u in summary.index.astype(str) if low <= before_positive[u] < high]
        if not train_units or not score_units:
            raise ValueError(f"empty inner fold {fold_id}")
        train_end = summary.loc[train_units, "end"].max()
        score_units = [u for u in score_units if summary.loc[u, "start"] > train_end + pd.Timedelta(hours=24)]
        train_rows = table.loc[table["unit_id"].isin(train_units), "row"].to_numpy(dtype=int)
        score_rows = table.loc[table["unit_id"].isin(score_units), "row"].to_numpy(dtype=int)
        if len(train_rows) == 0 or len(score_rows) == 0:
            raise ValueError(f"fold {fold_id} empty after purge")
        if len(np.unique(y[train_rows])) != 2 or len(np.unique(y[score_rows])) != 2:
            raise ValueError(f"fold {fold_id} lacks both classes")
        folds.append({
            "fold": fold_id,
            "train_idx": train_rows,
            "score_idx": score_rows,
            "train_positive_units_cut": int(low),
            "score_positive_units_end_cut": int(high),
            "train_prevalence": float(np.mean(y[train_rows])),
            "train_rows": int(len(train_rows)),
            "score_rows": int(len(score_rows)),
            "train_positives": int(y[train_rows].sum()),
            "score_positives": int(y[score_rows].sum()),
        })
    return folds


def calibrated_probability(raw_probability: np.ndarray, y: np.ndarray, roles: np.ndarray):
    cal = roles == "calibration"
    intercept = v1.fit_intercept(raw_probability[cal], y[cal])
    p = v1.sigmoid(v1.logit(raw_probability) + intercept)
    return p, float(intercept)


def thresholds(y, p, roles):
    m = roles == "threshold"
    pod80 = v1.minimum_far_at_pod(y[m], p[m], 0.8)
    if pod80 is None: raise ValueError("POD80 threshold unavailable")
    return {"MAX_TSS": float(v1.select_threshold(y[m], p[m])), "POD80_MIN_FAR": float(pod80["threshold"])}


def evaluate(y, p, threshold, roles, role, prevalence):
    m = roles == role
    return {
        **v1.threshold_metrics(y[m], p[m], threshold),
        **v1.probability_metrics(y[m], p[m], prevalence),
        "matched_detection": {str(pod): v1.minimum_far_at_pod(y[m], p[m], pod) for pod in (0.6, 0.7, 0.8, 0.9)},
        "rows": int(m.sum()), "positives": int(y[m].sum()),
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists(): raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    fit = roles == "fit"; cal = roles == "calibration"
    fit_prevalence = float(np.mean(y[fit]))
    xrs_rel = family_reliability(frame, xrs); proton_rel = family_reliability(frame, proton)

    folds = build_inner_folds(frame, y, roles, units)
    oof_rows = []
    oof_evidence = []
    oof_labels = []
    fold_receipts = []
    for fold in folds:
        tr = fold["train_idx"]; sc = fold["score_idx"]; prev = fold["train_prevalence"]
        ps = specialist_predict(frame, base, y, tr, sc)
        px = specialist_predict(frame, xrs, y, tr, sc)
        pp = specialist_predict(frame, proton, y, tr, sc)
        evidence = np.column_stack([
            centered_evidence(ps, prev),
            centered_evidence(px, prev, xrs_rel[sc]),
            centered_evidence(pp, prev, proton_rel[sc]),
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

    raw_solar = cs.fit_xgb_family(frame, base, y, fit)
    raw_xrs = cs.fit_xgb_family(frame, xrs, y, fit)
    raw_proton = cs.fit_xgb_family(frame, proton, y, fit)
    full_evidence = np.column_stack([
        centered_evidence(raw_solar, fit_prevalence),
        centered_evidence(raw_xrs, fit_prevalence, xrs_rel),
        centered_evidence(raw_proton, fit_prevalence, proton_rel),
    ])
    stack_raw = v1.sigmoid(stack.decision_function(full_evidence))
    stack_probability, stack_cal_intercept = calibrated_probability(stack_raw, y, roles)

    base_probability, base_intercept = calibrated_probability(raw_solar, y, roles)
    late_probability, _late_threshold, late_meta = cs.fit_late_stack([raw_solar, raw_xrs, raw_proton], y, roles)

    probability = {
        "BASE_SOLAR": base_probability,
        "LATE_FUSION_SOLAR_XRS_PROTON": late_probability,
        "IRIS_CROSSFIT_EVIDENCE_STACK_V1": stack_probability,
    }
    metadata = {
        "BASE_SOLAR": {"calibration_intercept": base_intercept},
        "LATE_FUSION_SOLAR_XRS_PROTON": late_meta,
        "IRIS_CROSSFIT_EVIDENCE_STACK_V1": {
            **stack.diagnostics(),
            "post_stack_calibration_intercept": stack_cal_intercept,
            "oof_rows": int(len(oof_rows)),
            "oof_positives": int(oof_labels.sum()),
            "inner_folds": fold_receipts,
            "evidence_limit": EVIDENCE_LIMIT,
        },
    }
    model_thresholds = {name: thresholds(y, p, roles) for name, p in probability.items()}

    summary = {
        "status": "COMPLETED_CROSSFIT_EVIDENCE_STACK_DEVELOPMENT_DIAGNOSTIC",
        "target": v1.TARGET,
        "locked_test_accessed": False,
        "score_and_monitor_already_inspected": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric_columns": dropped,
        "models": {},
        "paired_comparisons": {},
    }
    for name in MODELS:
        summary["models"][name] = {"metadata": metadata[name], "thresholds": model_thresholds[name], "policies": {}}
        for policy in POLICIES:
            t = model_thresholds[name][policy]
            summary["models"][name]["policies"][policy] = {
                "score": evaluate(y, probability[name], t, roles, "score", fit_prevalence),
                "monitor": evaluate(y, probability[name], t, roles, "monitor", fit_prevalence),
            }

    for policy in POLICIES:
        for role in ("score", "monitor"):
            role_mask = np.where(roles == role, role, "outside")
            for other in ("BASE_SOLAR", "LATE_FUSION_SOLAR_XRS_PROTON"):
                key = f"IRIS_CROSSFIT_EVIDENCE_STACK_V1_minus_{other}_{policy}_{role}"
                summary["paired_comparisons"][key] = cs.bootstrap_difference(
                    y,
                    probability["IRIS_CROSSFIT_EVIDENCE_STACK_V1"], model_thresholds["IRIS_CROSSFIT_EVIDENCE_STACK_V1"][policy],
                    probability[other], model_thresholds[other][policy],
                    units, role_mask, role, seed=20260906, replicates=10000,
                )

    rows = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles, "unit_id": units, "label": y,
    })
    for name in MODELS: rows[name] = probability[name]
    rows.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": "DEVELOPMENT_ONLY_CROSSFIT_STACK_RUN",
        "preregistration": "config/crossfit_evidence_stack_preregistration_2026-09-06.json",
        "locked_test_accessed": False,
        "calibration_used_for_meta_weights": False,
        "threshold_used_for_meta_weights": False,
        "score_or_monitor_used_for_meta_weights": False,
        "post_result_hyperparameter_changes": False,
    })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); run(a.features, a.events, a.output)


if __name__ == "__main__": main()
