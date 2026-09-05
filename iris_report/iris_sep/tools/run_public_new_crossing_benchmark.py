"""Frozen public-data NEW-crossing benchmark for IRIS-SEP.

This runner uses the public SEP-PRISM 24 h predictor table plus the CLEAR
Benchmark operational-event start/end catalogue.  It constructs the target
independently from event onset times:

    issue is eligible only when no >=10 pfu operational event is active;
    target=1 iff a NEW operational event starts within the next 24 hours.

It is a TRAIN-ONLY DEVELOPMENT benchmark.  The final locked evaluation remains
untouched.  No result from this runner is a final superiority claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


EXPECTED_PRISM_COMMIT = "e138dcd72c1952a00e11e1a0b025337f9e7c93fb"
EXPECTED_FEATURE_SHA256 = "4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb"
EXPECTED_EVENT_SHA256 = "0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0"
TARGET = "new_sep_10mev_10pfu_within_24h"
SEEDS = (7, 13, 26, 42, 73)
PURGE_HOURS = 24
QUIET_BLOCK_DAYS = 7
EVENT_START = ">10.0 MeV 10.0 pfu SEP Start Time"
EVENT_END = ">10.0 MeV 10.0 pfu SEP End Time"
ROLE_ORDER = ("fit", "calibration", "threshold", "score")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def fit_intercept(probabilities, labels):
    z = logit(probabilities)
    y = np.asarray(labels, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both classes")
    b = 0.0
    for _ in range(100):
        p = sigmoid(z + b)
        g = float(np.sum(p - y))
        h = float(np.sum(p * (1 - p)))
        if h <= 1e-12:
            raise ValueError("degenerate intercept calibration")
        step = g / h
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def threshold_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(p) >= threshold
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred)); tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / denom if denom else float("nan")
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "POD": pod, "FPR": fpr, "FAR": far, "TSS": tss, "HSS": hss}


def select_threshold(y, p):
    candidates = np.unique(np.r_[0.0, np.asarray(p, dtype=float), 1.0])
    scores = [(threshold_metrics(y, p, float(t))["TSS"], float(t)) for t in candidates]
    best = max(s for s, _ in scores)
    return min(t for s, t in scores if np.isclose(s, best, rtol=0, atol=1e-12))


def probability_metrics(y, p, reference_probability):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    rb = float(np.mean((reference_probability - y) ** 2))
    edges = np.linspace(0, 1, 11); ece = 0.0
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        if np.any(m):
            ece += float(np.mean(m) * abs(np.mean(p[m]) - np.mean(y[m])))
    return {
        "BRIER": brier,
        "BRIER_SKILL": float(1 - brier / rb) if rb > 0 else None,
        "ECE": ece,
        "AUPRC": float(average_precision_score(y, p)),
        "AUROC": float(roc_auc_score(y, p)),
    }


def minimum_far_at_pod(y, p, target_pod):
    best = None
    for t in np.unique(np.r_[0.0, p, 1.0]):
        m = threshold_metrics(y, p, float(t))
        if m["POD"] >= target_pod and np.isfinite(m["FAR"]):
            row = (m["FAR"], -float(t), m)
            if best is None or row[:2] < best[:2]:
                best = row
    if best is None:
        return None
    return {"target_POD": target_pod, "achieved_POD": best[2]["POD"], "FAR": best[0], "threshold": -best[1]}


def derive_target(frame: pd.DataFrame, event_frame: pd.DataFrame):
    if EVENT_START not in event_frame or EVENT_END not in event_frame:
        raise ValueError("CLEAR operational start/end columns missing")
    starts = pd.to_datetime(event_frame[EVENT_START], format="%Y/%m/%d %H:%M", utc=True, errors="coerce")
    ends = pd.to_datetime(event_frame[EVENT_END], format="%Y/%m/%d %H:%M", utc=True, errors="coerce")
    valid = starts.notna() & ends.notna()
    events = pd.DataFrame({"start": starts[valid], "end": ends[valid]}).sort_values("start").reset_index(drop=True)
    if events.empty or (events["end"] < events["start"]).any():
        raise ValueError("invalid CLEAR operational event catalogue")

    issue = frame["window_end"]
    active = np.zeros(len(frame), dtype=bool)
    target = np.zeros(len(frame), dtype=np.int8)
    event_ids = np.full(len(frame), "", dtype="U64")
    event_start_seconds = events["start"].astype("int64").to_numpy() // 10**9
    event_end_seconds = events["end"].astype("int64").to_numpy() // 10**9
    issue_seconds = issue.astype("int64").to_numpy() // 10**9
    horizon = 24 * 3600
    for i, t in enumerate(issue_seconds):
        is_active = np.any((event_start_seconds <= t) & (event_end_seconds >= t))
        active[i] = is_active
        if is_active:
            continue
        future = np.flatnonzero((event_start_seconds > t) & (event_start_seconds <= t + horizon))
        if len(future):
            j = int(future[0])
            target[i] = 1
            event_ids[i] = hashlib.sha256(str(int(event_start_seconds[j])).encode()).hexdigest()[:24]
    return target, active, event_ids, events


def build_units(issue_times, targets, event_ids):
    issue_times = pd.Series(issue_times).reset_index(drop=True)
    origin = issue_times.min().floor("D")
    units = []
    for i, (t, y, event_id) in enumerate(zip(issue_times, targets, event_ids)):
        if int(y) == 1:
            uid = f"event-{event_id}"
        else:
            block = int((t - origin).total_seconds() // (QUIET_BLOCK_DAYS * 86400))
            uid = f"quiet-{block:05d}"
        units.append(uid)
    table = pd.DataFrame({"row": np.arange(len(units)), "unit_id": units, "time": issue_times, "label": targets})
    summaries = table.groupby("unit_id", sort=False).agg(start=("time", "min"), end=("time", "max"), label=("label", "max"))
    if (table.groupby("unit_id")["label"].nunique() > 1).any():
        raise ValueError("unit mixes positive and negative rows")
    summaries = summaries.sort_values(["start", "end"])
    positive_units = int(summaries["label"].sum())
    if positive_units < 20:
        raise ValueError(f"too few positive event units: {positive_units}")

    cutoffs = (int(positive_units * .70), int(positive_units * .80), int(positive_units * .90))
    mapping = {}; positives_seen = 0
    for uid, row in summaries.iterrows():
        if positives_seen < cutoffs[0]: role = "fit"
        elif positives_seen < cutoffs[1]: role = "calibration"
        elif positives_seen < cutoffs[2]: role = "threshold"
        else: role = "score"
        mapping[uid] = role
        positives_seen += int(row.label)

    # Strict >24 h purge at every role boundary. Remove complete units from the right block.
    purged = set()
    for left, right in zip(ROLE_ORDER, ROLE_ORDER[1:]):
        left_units = [u for u, r in mapping.items() if r == left and u not in purged]
        if not left_units:
            raise ValueError(f"empty role before purge: {left}")
        left_end = summaries.loc[left_units, "end"].max()
        right_units = [u for u, r in mapping.items() if r == right]
        for uid in right_units:
            if summaries.loc[uid, "start"] <= left_end + pd.Timedelta(hours=PURGE_HOURS):
                purged.add(uid)

    roles = np.array([mapping[u] if u not in purged else "purged" for u in units], dtype="U16")
    for role in ROLE_ORDER:
        labels = np.asarray(targets)[roles == role]
        if len(labels) == 0 or len(np.unique(labels)) != 2:
            raise ValueError(f"role {role} lacks both classes after purge")
    # Verify strict row-time purges.
    for left, right in zip(ROLE_ORDER, ROLE_ORDER[1:]):
        gap = issue_times[roles == right].min() - issue_times[roles == left].max()
        if not gap > pd.Timedelta(hours=PURGE_HOURS):
            raise ValueError(f"strict purge failed: {left}->{right}: {gap}")
    return np.asarray(units, dtype="U64"), roles, sorted(purged), positive_units


def feature_sets(frame):
    excluded = {"window_begin", "window_end"}
    feature_names = []
    dropped_non_numeric = []
    for c in frame.columns:
        low = c.lower()
        if c in excluded or low.startswith("future_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            feature_names.append(c)
        else:
            dropped_non_numeric.append(c)
    if not feature_names:
        raise ValueError("no numeric causal predictors")
    proton = [c for c in feature_names if "protonflux" in c.lower()]
    xrs = [c for c in feature_names if "xrs" in c.lower()]
    base = [c for c in feature_names if c not in set(proton + xrs)]
    if not proton or not xrs or not base:
        raise ValueError("expected base, proton and XRS feature families")
    return {
        "BASE_SOLAR": base,
        "BASE_PLUS_PROTON": base + proton,
        "BASE_PLUS_XRS": base + xrs,
        "FULL_CONTEXT": base + proton + xrs,
    }, dropped_non_numeric


def fit_elastic(train_x, train_y, all_x):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=10000,
            random_state=20260905,
        ),
    )
    model.fit(train_x, train_y)
    if int(np.max(model[-1].n_iter_)) >= 10000:
        raise RuntimeError("elastic net did not converge")
    return model.predict_proba(all_x)[:, 1]


def fit_xgb(train_x, train_y, all_x, seed):
    prevalence = float(np.mean(train_y))
    model = XGBClassifier(
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
    model.fit(train_x, train_y, verbose=False)
    return model.predict_proba(all_x)[:, 1]


def calibrate_and_score(raw_probability, y, roles, prevalence):
    cal = roles == "calibration"; threshold_rows = roles == "threshold"; score = roles == "score"
    intercept = fit_intercept(raw_probability[cal], y[cal])
    probability = sigmoid(logit(raw_probability) + intercept)
    threshold = select_threshold(y[threshold_rows], probability[threshold_rows])
    out = {
        "calibration_intercept": intercept,
        "threshold": float(threshold),
        **threshold_metrics(y[score], probability[score], threshold),
        **probability_metrics(y[score], probability[score], prevalence),
        "matched_detection": {str(pod): minimum_far_at_pod(y[score], probability[score], pod) for pod in (.6, .7, .8, .9)},
        "score_rows": int(np.sum(score)),
        "score_positives": int(np.sum(y[score])),
    }
    return probability, out


def paired_bootstrap(y, p_a, t_a, p_b, t_b, unit_ids, roles, replicates=10000, seed=20260905):
    mask = roles == "score"
    y = y[mask]; a = p_a[mask]; b = p_b[mask]; units = unit_ids[mask]
    unique = np.unique(units); rows = {u: np.flatnonzero(units == u) for u in unique}
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([rows[u] for u in sampled])
        if len(np.unique(y[ix])) != 2:
            continue
        diffs.append(threshold_metrics(y[ix], a[ix], t_a)["TSS"] - threshold_metrics(y[ix], b[ix], t_b)["TSS"])
    if len(diffs) < int(.95 * replicates):
        raise ValueError("fewer than 95% valid bootstrap replicates")
    d = np.asarray(diffs)
    return {
        "valid_replicates": int(len(d)),
        "median_TSS_difference": float(np.median(d)),
        "ci_lower_95": float(np.quantile(d, .025)),
        "ci_upper_95": float(np.quantile(d, .975)),
        "probability_difference_positive": float(np.mean(d > 0)),
    }


def run(features_csv: Path, events_csv: Path, output: Path):
    if output.exists():
        raise ValueError("immutable output directory already exists")
    if digest(features_csv) != EXPECTED_FEATURE_SHA256:
        raise ValueError("SEP-PRISM feature table hash mismatch")
    if digest(events_csv) != EXPECTED_EVENT_SHA256:
        raise ValueError("CLEAR event catalogue hash mismatch")
    output.mkdir(parents=True)

    frame = pd.read_csv(features_csv, low_memory=False)
    events = pd.read_csv(events_csv, low_memory=False)
    required = {"window_begin", "window_end"}
    if not required.issubset(frame.columns):
        raise ValueError("SEP-PRISM table lacks window timestamps")
    frame["window_begin"] = pd.to_datetime(frame["window_begin"], utc=True, errors="raise")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="raise")
    frame = frame.sort_values(["window_end", "window_begin"]).drop_duplicates(["window_begin", "window_end"]).reset_index(drop=True)
    if not ((frame["window_end"] - frame["window_begin"]) == pd.Timedelta(hours=24)).all():
        raise ValueError("predictor windows must be exactly 24 h")
    if np.any(frame["window_end"].diff().dropna() <= pd.Timedelta(0)):
        raise ValueError("issue times must be strictly increasing")

    y_all, active, event_ids_all, parsed_events = derive_target(frame, events)
    eligible = ~active
    frame = frame.loc[eligible].reset_index(drop=True)
    y = y_all[eligible]
    event_ids = event_ids_all[eligible]
    unit_ids, roles, purged_units, positive_units = build_units(frame["window_end"], y, event_ids)
    keep = roles != "purged"
    frame = frame.loc[keep].reset_index(drop=True); y = y[keep]; unit_ids = unit_ids[keep]; roles = roles[keep]

    sets, dropped_non_numeric = feature_sets(frame)
    fit = roles == "fit"
    prevalence = float(np.mean(y[fit]))
    predictions = {}; results = {}

    # Climatology.
    raw = np.full(len(y), prevalence, dtype=float)
    predictions["CLIMATOLOGY"], results["CLIMATOLOGY"] = calibrate_and_score(raw, y, roles, prevalence)

    # Eligible causal persistence baseline from current-window OSEP history, fitted only on fit rows.
    if "OSEP_label" in frame.columns:
        current = pd.to_numeric(frame["OSEP_label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        rates = {}
        for state in (0, 1):
            m = fit & (current == state)
            rates[state] = float((np.sum(y[m]) + 1) / (np.sum(m) + 2)) if np.any(m) else prevalence
        raw = np.array([rates.get(int(v), prevalence) for v in current])
        predictions["CAUSAL_PERSISTENCE"], results["CAUSAL_PERSISTENCE"] = calibrate_and_score(raw, y, roles, prevalence)
        results["CAUSAL_PERSISTENCE"]["fit_conditional_rates"] = {str(k): v for k, v in rates.items()}

    # Elastic net on the fixed full-context schema.
    full_features = sets["FULL_CONTEXT"]
    raw = fit_elastic(frame.loc[fit, full_features], y[fit], frame[full_features])
    predictions["ELASTIC_NET_FULL"], results["ELASTIC_NET_FULL"] = calibrate_and_score(raw, y, roles, prevalence)

    # Four predeclared XGBoost context arms, five fixed seeds, median probability aggregation.
    for arm, names in sets.items():
        seed_predictions = []
        for seed in SEEDS:
            seed_predictions.append(fit_xgb(frame.loc[fit, names], y[fit], frame[names], seed))
        raw = np.median(np.stack(seed_predictions), axis=0)
        name = f"XGBOOST_{arm}"
        predictions[name], results[name] = calibrate_and_score(raw, y, roles, prevalence)
        results[name]["seeds"] = list(SEEDS)
        results[name]["feature_count"] = len(names)

    # Independent target-semantic audit against publisher overlap label.
    target_audit = {}
    if "Future_OSEP_label" in frame.columns:
        publisher = pd.to_numeric(frame["Future_OSEP_label"], errors="raise").to_numpy(dtype=int)
        target_audit = {
            "rows": int(len(y)),
            "new_crossing_positives": int(np.sum(y)),
            "publisher_future_overlap_positives": int(np.sum(publisher)),
            "publisher_positive_new_crossing_negative": int(np.sum((publisher == 1) & (y == 0))),
            "publisher_negative_new_crossing_positive": int(np.sum((publisher == 0) & (y == 1))),
            "agreement_fraction": float(np.mean(publisher == y)),
        }

    full = "XGBOOST_FULL_CONTEXT"; base = "XGBOOST_BASE_SOLAR"; elastic = "ELASTIC_NET_FULL"
    comparisons = {
        "FULL_CONTEXT_minus_BASE_SOLAR": paired_bootstrap(
            y, predictions[full], results[full]["threshold"], predictions[base], results[base]["threshold"], unit_ids, roles
        ),
        "FULL_CONTEXT_minus_ELASTIC_NET": paired_bootstrap(
            y, predictions[full], results[full]["threshold"], predictions[elastic], results[elastic]["threshold"], unit_ids, roles
        ),
    }

    # Save predictions with no future target/source columns.
    pred = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": unit_ids,
        "label": y,
    })
    for name, values in predictions.items():
        pred[name] = values
    pred_path = output / "predictions.csv"
    pred.to_csv(pred_path, index=False, float_format="%.17g")

    # Source/role manifest.  This is the trust anchor for the missingness package.
    role_support = {}
    for role in ROLE_ORDER:
        m = roles == role
        role_support[role] = {
            "rows": int(np.sum(m)), "positives": int(np.sum(y[m])), "units": int(len(np.unique(unit_ids[m]))),
            "from": frame.loc[m, "window_end"].min().isoformat(), "to": frame.loc[m, "window_end"].max().isoformat(),
        }
    source_manifest = {
        "status": "TRAIN_ONLY_PUBLIC_NEW_CROSSING_DEVELOPMENT_NOT_FINAL_LOCKED_EVALUATION",
        "sep_prism_repo_commit": EXPECTED_PRISM_COMMIT,
        "feature_table_sha256": EXPECTED_FEATURE_SHA256,
        "event_catalogue_sha256": EXPECTED_EVENT_SHA256,
        "target": TARGET,
        "target_rule": "eligible iff no operational event active at issue; positive iff operational event start is >issue and <=issue+24h",
        "parsed_operational_events": int(len(parsed_events)),
        "rows_before_active_exclusion": int(len(y_all)),
        "already_active_rows_excluded": int(np.sum(active)),
        "positive_event_units_before_purge": int(positive_units),
        "purged_units": purged_units,
        "purge_hours": PURGE_HOURS,
        "quiet_block_days": QUIET_BLOCK_DAYS,
        "role_support": role_support,
        "feature_families": {k: len(v) for k, v in sets.items()},
        "dropped_non_numeric_columns": dropped_non_numeric,
        "future_columns_excluded_from_predictors": sorted([c for c in frame.columns if c.lower().startswith("future_")]),
        "locked_test_accessed": False,
        "score_role_is_train_only_diagnostic": True,
    }
    manifest_path = output / "source_manifest.json"
    save_json(manifest_path, source_manifest)
    manifest_sha = digest(manifest_path)

    # Safe missingness package: only columns with complete finite values on all retained rows.
    complete = []
    for c in full_features:
        arr = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(arr).all():
            complete.append(c)
    if len(complete) < 2:
        raise ValueError("fewer than two fully observed causal features for safe missingness package")
    values = frame[complete].to_numpy(dtype=np.float64)
    observed = np.ones_like(values, dtype=bool)
    structural = np.zeros_like(values, dtype=bool)
    role_map = {"fit": "fit", "calibration": "calibration", "threshold": "threshold", "score": "score"}
    package_roles = np.array([role_map[r] for r in roles], dtype="U16")
    issue_ids = np.array([hashlib.sha256(t.isoformat().encode()).hexdigest() for t in frame["window_end"]], dtype="U64")
    seconds = frame["window_end"].astype("int64").to_numpy(dtype=np.int64) / 1e9
    package_path = output / "missingness_package.npz"
    np.savez(
        package_path,
        values=values,
        observed_mask=observed,
        structural_unavailable_mask=structural,
        labels=y.astype(np.int8),
        roles=package_roles,
        issue_ids=issue_ids,
        unit_ids=unit_ids,
        issue_time_unix_seconds=seconds,
    )
    package_meta = {
        "format": "IRIS_SEP_TRAIN_ONLY_MISSINGNESS_PACKAGE_V1",
        "target": TARGET,
        "scope": "TRAIN_ONLY_NEW_CROSSING_MISSINGNESS",
        "locked_test_included": False,
        "chronological_roles_verified": True,
        "episode_disjoint_roles_verified": True,
        "purge_hours": PURGE_HOURS,
        "source_manifest_sha256": manifest_sha,
        "feature_names": complete,
        "feature_selection": "ONLY_CAUSAL_NUMERIC_COLUMNS_FINITE_ON_EVERY_RETAINED_ROW",
        "structural_unavailable_mask_rule": "all false because selected columns have zero missing cells in retained cohort",
    }
    package_meta_path = output / "missingness_metadata.json"
    save_json(package_meta_path, package_meta)

    summary = {
        "status": "COMPLETED_TRAIN_ONLY_PUBLIC_NEW_CROSSING_DIAGNOSTIC",
        "target": TARGET,
        "results": results,
        "paired_comparisons": comparisons,
        "target_semantic_audit": target_audit,
        "role_support": role_support,
        "fit_prevalence": prevalence,
        "missingness_package_feature_count": len(complete),
        "missingness_package_features": complete,
        "source_manifest_sha256": manifest_sha,
        "prediction_sha256": digest(pred_path),
        "package_sha256": digest(package_path),
        "metadata_sha256": digest(package_meta_path),
        "locked_test_accessed": False,
        "superiority_established": False,
        "final_claim_allowed": False,
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
