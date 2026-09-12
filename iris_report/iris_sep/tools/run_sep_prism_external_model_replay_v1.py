from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROLLING_REL = Path("Data/rolling_combinded_seq_24hours.csv")
EVENT_REL = Path("GOES_integral_PRIMARY.1986-02-03.2025-09-10_sep_events.csv")
ROLLING_SHA = "4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb"
EVENT_SHA = "0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0"
START_COL = ">10.0 MeV 10.0 pfu SEP Start Time"
END_COL = ">10.0 MeV 10.0 pfu SEP End Time"
SEEDS = [7, 13, 26, 42, 73]
BOOT_REPS = 10_000
BOOT_SEED = 20260910
FOLDS = [
    ("OOF_2005_2010", "1986-02-04", "2000-01-01", "2000-01-01", "2005-01-01", "2005-01-01", "2011-01-01"),
    ("OOF_2011_2017", "1986-02-04", "2005-01-01", "2005-01-01", "2011-01-01", "2011-01-01", "2018-01-01"),
    ("OOF_2018_2025", "1986-02-04", "2011-01-01", "2011-01-01", "2018-01-01", "2018-01-01", "2025-09-11"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n", encoding="utf-8")


def utc(values: Any) -> Any:
    return pd.to_datetime(values, utc=True, errors="coerce")


def build_episodes(events: pd.DataFrame) -> pd.DataFrame:
    starts = utc(events[START_COL])
    ends = utc(events[END_COL])
    ok = starts.notna() & ends.notna() & (ends > starts)
    rows = []
    for ordinal, idx in enumerate(events.index[ok], start=1):
        start = pd.Timestamp(starts.loc[idx])
        rows.append({
            "episode_id": f"PRISM_OP_{ordinal:05d}_{start.strftime('%Y%m%dT%H%M%SZ')}",
            "start": start,
            "end": pd.Timestamp(ends.loc[idx]),
        })
    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)


def classify(issue: pd.Timestamp, episodes: pd.DataFrame) -> dict[str, Any]:
    horizon_end = issue + pd.Timedelta(hours=24)
    overlap = episodes[(episodes["start"] <= horizon_end) & (episodes["end"] >= issue)]
    active = episodes[(episodes["start"] <= issue) & (episodes["end"] > issue)]
    future = episodes[(episodes["start"] > issue) & (episodes["start"] <= horizon_end)]
    ids = overlap["episode_id"].astype(str).tolist()
    if len(active) > 1 or len(future) > 1:
        code, onset_id = "DUPLICATE_OR_AMBIGUOUS", None
    elif len(active) == 1:
        code, onset_id = "ALREADY_ACTIVE_PERSISTENCE", None
    elif len(future) == 1:
        code, onset_id = "ELIGIBLE_ONSET_POSITIVE", str(future.iloc[0]["episode_id"])
    else:
        code, onset_id = "ELIGIBLE_ONSET_NEGATIVE", None
    return {
        "reconstructed_label": int(bool(ids)),
        "eligibility": code,
        "onset_episode_id": onset_id,
        "episode_id": ids[0] if len(ids) == 1 else None,
        "match_count": int(len(ids)),
    }


def quiet_block(issue: pd.Timestamp) -> str:
    day = issue.normalize()
    monday = day - pd.Timedelta(days=int(day.weekday()))
    return f"QUIET::{monday.strftime('%Y-%m-%d')}"


def fit_xgb_ensemble(X: pd.DataFrame, y: np.ndarray) -> list[XGBClassifier]:
    models = []
    for seed in SEEDS:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        )
        model.fit(X, y)
        models.append(model)
    return models


def predict_xgb(models: list[XGBClassifier], X: pd.DataFrame) -> np.ndarray:
    return np.median(np.vstack([m.predict_proba(X)[:, 1] for m in models]), axis=0)


def fit_elastic(X: pd.DataFrame, y: np.ndarray):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            penalty="elasticnet",
            C=1.0,
            l1_ratio=0.5,
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )
    model.fit(X, y)
    return model


def best_tss_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    best_score = -np.inf
    best_threshold: float | None = None
    pos = y == 1
    neg = y == 0
    for threshold in np.unique(np.asarray(probability, dtype=float)):
        alert = probability >= threshold
        pod = float(np.mean(alert[pos]))
        fpr = float(np.mean(alert[neg]))
        score = pod - fpr
        if score > best_score + 1e-15 or (
            abs(score - best_score) <= 1e-15 and (best_threshold is None or threshold > best_threshold)
        ):
            best_score = score
            best_threshold = float(threshold)
    if best_threshold is None:
        raise RuntimeError("threshold selection failed")
    return best_threshold


def metric(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y"].to_numpy(int)
    alert = frame["alert"].to_numpy(int).astype(bool)
    probability = frame["probability"].to_numpy(float)
    weight = frame["weight"].to_numpy(float)
    tp = float(weight[(y == 1) & alert].sum())
    fn = float(weight[(y == 1) & ~alert].sum())
    fp = float(weight[(y == 0) & alert].sum())
    tn = float(weight[(y == 0) & ~alert].sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "pod": float(pod), "fpr": float(fpr), "far": float(far),
        "tss": float(pod - fpr),
        "hss": float(2 * (tp * tn - fn * fp) / denominator) if denominator else np.nan,
        "brier": float(np.sum(weight * (probability - y) ** 2) / weight.sum()),
    }


def views(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    standard = frame.copy()
    standard["y"] = standard["standard_label"].astype(int)
    standard["weight"] = 1.0

    mapped = frame[(frame["standard_label"] == 0) | ((frame["standard_label"] == 1) & (frame["match_count"] == 1))].copy()
    mapped["y"] = mapped["standard_label"].astype(int)
    mapped["weight"] = 1.0

    normalized = mapped.copy()
    counts = Counter(normalized.loc[normalized["y"] == 1, "episode_id"].dropna().astype(str))
    normalized.loc[normalized["y"] == 1, "weight"] = [
        1.0 / counts[str(x)] for x in normalized.loc[normalized["y"] == 1, "episode_id"]
    ]

    onset = frame[frame["eligibility"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
    onset["y"] = onset["eligibility"].eq("ELIGIBLE_ONSET_POSITIVE").astype(int)
    onset["weight"] = 1.0
    return {
        "STANDARD_FULL": standard,
        "MAPPED_STANDARD": mapped,
        "EPISODE_NORMALIZED_OCCURRENCE": normalized,
        "NEW_ONSET": onset,
    }


def unit_contributions(frame: pd.DataFrame, positive_units: list[str], negative_units: list[str]) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((len(positive_units), 4), dtype=float)
    negative = np.zeros((len(negative_units), 4), dtype=float)
    for i, unit in enumerate(positive_units):
        group = frame[(frame["y"] == 1) & frame["episode_id"].eq(unit)]
        w = group["weight"].to_numpy(float)
        a = group["alert"].to_numpy(int).astype(bool)
        positive[i, 0] = w[a].sum()
        positive[i, 1] = w[~a].sum()
    for i, unit in enumerate(negative_units):
        group = frame[(frame["y"] == 0) & frame["quiet_block"].eq(unit)]
        w = group["weight"].to_numpy(float)
        a = group["alert"].to_numpy(int).astype(bool)
        negative[i, 2] = w[a].sum()
        negative[i, 3] = w[~a].sum()
    return positive, negative


def tss_from_confusion(values: np.ndarray) -> np.ndarray:
    tp, fn, fp, tn = [values[..., i] for i in range(4)]
    return tp / (tp + fn) - fp / (fp + tn)


def paired_bootstrap(predictions: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict[str, Any], str, int, int]:
    model_names = sorted(predictions["model"].unique())
    base = predictions[predictions["model"].eq(model_names[0])].copy()
    onset_events = sorted(set(base.loc[
        base["eligibility"].eq("ELIGIBLE_ONSET_POSITIVE") & base["match_count"].eq(1), "episode_id"
    ].dropna().astype(str)))
    negative_units = sorted(set(base.loc[base["standard_label"].eq(0), "quiet_block"].astype(str)))
    if not onset_events or not negative_units:
        raise RuntimeError("paired physical-unit bootstrap population is empty")

    sensitivity = predictions[
        predictions["standard_label"].eq(0)
        | (predictions["match_count"].eq(1) & predictions["episode_id"].astype(str).isin(onset_events))
    ].copy()
    sensitivity.to_csv(out / "matched_episode_sensitivity_predictions.csv", index=False)

    rng = np.random.default_rng(BOOT_SEED)
    positive_indices = rng.integers(0, len(onset_events), size=(BOOT_REPS, len(onset_events)))
    negative_indices = rng.integers(0, len(negative_units), size=(BOOT_REPS, len(negative_units)))
    np.savez_compressed(
        out / "shared_bootstrap_draws.npz",
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        positive_units=np.asarray(onset_events),
        negative_units=np.asarray(negative_units),
    )

    distributions: dict[tuple[str, str], np.ndarray] = {}
    point_rows = []
    for model_name in model_names:
        model_frame = sensitivity[sensitivity["model"].eq(model_name)].copy()
        model_views = views(model_frame)
        for view_name in ["MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"]:
            view = model_views[view_name]
            point_rows.append({"model": model_name, "view": view_name, **metric(view)})
            positive, negative = unit_contributions(view, onset_events, negative_units)
            sampled = positive[positive_indices].sum(axis=1) + negative[negative_indices].sum(axis=1)
            distributions[(model_name, view_name)] = tss_from_confusion(sampled)
    point = pd.DataFrame(point_rows)
    point.to_csv(out / "matched_episode_point_results.csv", index=False)

    def contrast(model: str, earlier: str, later: str) -> dict[str, float]:
        delta = distributions[(model, later)] - distributions[(model, earlier)]
        return {
            "median": float(np.nanmedian(delta)),
            "lo": float(np.nanquantile(delta, 0.025)),
            "hi": float(np.nanquantile(delta, 0.975)),
            "pr_negative": float(np.nanmean(delta < 0)),
            "pr_positive": float(np.nanmean(delta > 0)),
        }

    contrasts: dict[str, Any] = {
        "XGB_JOINT_MULTIPLICITY_EFFECT": contrast("xgb_joint", "MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE"),
        "XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT": contrast("xgb_joint", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"),
        "PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT": contrast("past_proton_ge10_proxy", "MAPPED_STANDARD", "NEW_ONSET"),
    }
    for family, joint, no_proton in [
        ("XGB", "xgb_joint", "xgb_no_proton"),
        ("ELASTIC", "elastic_net_joint", "elastic_net_no_proton"),
    ]:
        onset_delta = distributions[(joint, "NEW_ONSET")] - distributions[(no_proton, "NEW_ONSET")]
        standard_delta = distributions[(joint, "MAPPED_STANDARD")] - distributions[(no_proton, "MAPPED_STANDARD")]
        contrasts[f"{family}_JOINT_MINUS_NO_PROTON_ONSET"] = {
            "median": float(np.nanmedian(onset_delta)),
            "lo": float(np.nanquantile(onset_delta, 0.025)),
            "hi": float(np.nanquantile(onset_delta, 0.975)),
            "pr_positive": float(np.nanmean(onset_delta > 0)),
            "ordering_reversal_fraction": float(np.nanmean(np.sign(standard_delta) != np.sign(onset_delta))),
        }

    directional = [
        contrasts["XGB_JOINT_MULTIPLICITY_EFFECT"],
        contrasts["XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT"],
        contrasts["PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT"],
    ]
    supported = sum(item["hi"] < 0 for item in directional)
    required = (
        contrasts["XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT"]["hi"] < 0
        or contrasts["PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT"]["hi"] < 0
    )
    if supported >= 2 and required:
        gate = "STRONG_MODEL_CONFIRMATION"
    elif supported >= 1:
        gate = "PARTIAL_MODEL_CONFIRMATION"
    else:
        gate = "NULL_MODEL_CONFIRMATION"
    dump_json(out / "bootstrap_contrasts.json", contrasts)
    return point, contrasts, gate, len(onset_events), len(negative_units)


def run(upstream: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise RuntimeError(f"output exists: {out}")
    out.mkdir(parents=True)
    rolling_path = upstream / ROLLING_REL
    event_path = upstream / EVENT_REL
    if sha256_file(rolling_path) != ROLLING_SHA or sha256_file(event_path) != EVENT_SHA:
        raise RuntimeError("pinned SEP-PRISM source hash mismatch")

    data = pd.read_csv(rolling_path, low_memory=False)
    events = pd.read_csv(event_path, low_memory=False)
    data["window_begin"] = utc(data["window_begin"])
    data["window_end"] = utc(data["window_end"])
    if len(data) != 14464 or int(pd.to_numeric(data["Future_OSEP_label"], errors="raise").sum()) != 650:
        raise RuntimeError("pinned public row/positive count changed")
    if data["window_end"].duplicated().any():
        raise RuntimeError("duplicate issue timestamps")

    excluded_exact = {"window_begin", "window_end", "OSEP_label", "GSEP_label"}
    joint_columns = [c for c in data.columns if c not in excluded_exact and not c.startswith("Future_")]
    for column in joint_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    no_proton_columns = [
        c for c in joint_columns if c != "ProtonFlux_label" and not c.startswith("ProtonFlux_")
    ]
    if not joint_columns or not no_proton_columns or "ProtonFlux_max" not in joint_columns:
        raise RuntimeError("frozen feature rule produced invalid feature sets")

    physical = build_episodes(events)
    classifications = [classify(pd.Timestamp(issue), physical) for issue in data["window_end"]]
    labels = pd.DataFrame(classifications)
    data = pd.concat([data.reset_index(drop=True), labels], axis=1)
    data["quiet_block"] = [quiet_block(pd.Timestamp(x)) for x in data["window_end"]]
    if not np.array_equal(data["reconstructed_label"].to_numpy(int), data["Future_OSEP_label"].to_numpy(int)):
        raise RuntimeError("public target no longer matches independent physical reconstruction")

    prediction_rows = []
    fold_receipts = []
    for name, fit_start, fit_end, threshold_start, threshold_end, score_start, score_end in FOLDS:
        bounds = [pd.Timestamp(x, tz="UTC") for x in [fit_start, fit_end, threshold_start, threshold_end, score_start, score_end]]
        fs, fe, ts, te, ss, se = bounds
        fit = data[(data["window_end"] >= fs) & (data["window_end"] < fe)].copy()
        threshold = data[(data["window_end"] >= ts) & (data["window_end"] < te)].copy()
        score = data[(data["window_end"] >= ss) & (data["window_end"] < se)].copy()
        for role, frame in [("fit", fit), ("threshold", threshold), ("score", score)]:
            if frame.empty or frame["Future_OSEP_label"].nunique() != 2:
                raise RuntimeError(f"{name} {role} lacks binary support")

        y_fit = fit["Future_OSEP_label"].to_numpy(int)
        y_threshold = threshold["Future_OSEP_label"].to_numpy(int)
        xgb_joint = fit_xgb_ensemble(fit[joint_columns], y_fit)
        xgb_no_proton = fit_xgb_ensemble(fit[no_proton_columns], y_fit)
        elastic_joint = fit_elastic(fit[joint_columns], y_fit)
        elastic_no_proton = fit_elastic(fit[no_proton_columns], y_fit)

        threshold_probabilities = {
            "xgb_joint": predict_xgb(xgb_joint, threshold[joint_columns]),
            "xgb_no_proton": predict_xgb(xgb_no_proton, threshold[no_proton_columns]),
            "elastic_net_joint": elastic_joint.predict_proba(threshold[joint_columns])[:, 1],
            "elastic_net_no_proton": elastic_no_proton.predict_proba(threshold[no_proton_columns])[:, 1],
        }
        thresholds = {key: best_tss_threshold(y_threshold, value) for key, value in threshold_probabilities.items()}
        fit_prevalence = float(y_fit.mean())
        thresholds["fit_prevalence_climatology"] = fit_prevalence
        thresholds["past_proton_ge10_proxy"] = 0.5

        score_probabilities = {
            "xgb_joint": predict_xgb(xgb_joint, score[joint_columns]),
            "xgb_no_proton": predict_xgb(xgb_no_proton, score[no_proton_columns]),
            "elastic_net_joint": elastic_joint.predict_proba(score[joint_columns])[:, 1],
            "elastic_net_no_proton": elastic_no_proton.predict_proba(score[no_proton_columns])[:, 1],
            "fit_prevalence_climatology": np.full(len(score), fit_prevalence),
            "past_proton_ge10_proxy": (score["ProtonFlux_max"].fillna(-np.inf).to_numpy(float) >= 10.0).astype(float),
        }
        for model_name, probabilities in score_probabilities.items():
            threshold_value = float(thresholds[model_name])
            for offset, (_, row) in enumerate(score.iterrows()):
                probability = float(probabilities[offset])
                prediction_rows.append({
                    "fold": name,
                    "issue_timestamp": row["window_end"],
                    "model": model_name,
                    "probability": probability,
                    "threshold": threshold_value,
                    "alert": int(probability >= threshold_value),
                    "standard_label": int(row["Future_OSEP_label"]),
                    "eligibility": row["eligibility"],
                    "episode_id": row["episode_id"],
                    "onset_episode_id": row["onset_episode_id"],
                    "match_count": int(row["match_count"]),
                    "quiet_block": row["quiet_block"],
                })
        fold_receipts.append({
            "fold": name,
            "fit_rows": int(len(fit)),
            "threshold_rows": int(len(threshold)),
            "score_rows": int(len(score)),
            "fit_positive_windows": int(y_fit.sum()),
            "threshold_positive_windows": int(y_threshold.sum()),
            "score_positive_windows": int(score["Future_OSEP_label"].sum()),
            "fit_prevalence": fit_prevalence,
            "thresholds": thresholds,
        })

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out / "predictions.csv", index=False)
    dump_json(out / "fold_receipts.json", fold_receipts)
    dump_json(out / "feature_schema.json", {
        "joint": joint_columns,
        "no_proton": no_proton_columns,
        "excluded_exact": sorted(excluded_exact),
        "excluded_prefixes": ["Future_"],
    })

    descriptive_rows = []
    for model_name in sorted(predictions["model"].unique()):
        model_views = views(predictions[predictions["model"].eq(model_name)])
        for view_name, frame in model_views.items():
            descriptive_rows.append({
                "model": model_name, "view": view_name,
                "rows": int(len(frame)), "positive_rows": int(frame["y"].sum()),
                **metric(frame),
            })
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(out / "descriptive_results.csv", index=False)

    temporal_rows = []
    for fold_name in [x[0] for x in FOLDS]:
        for model_name in sorted(predictions["model"].unique()):
            subset = predictions[predictions["fold"].eq(fold_name) & predictions["model"].eq(model_name)]
            model_views = views(subset)
            for view_name in ["STANDARD_FULL", "NEW_ONSET"]:
                frame = model_views[view_name]
                temporal_rows.append({
                    "fold": fold_name, "model": model_name, "view": view_name,
                    "rows": int(len(frame)), "positive_rows": int(frame["y"].sum()),
                    **metric(frame),
                })
    pd.DataFrame(temporal_rows).to_csv(out / "per_fold_results.csv", index=False)

    matched_points, contrasts, gate, matched_events, quiet_blocks = paired_bootstrap(predictions, out)
    summary = {
        "format": "IRIS_SEP_PRISM_EXTERNAL_FIXED_MODEL_REPLAY_V1_RESULT",
        "status": gate,
        "score_rows_unique": int(predictions["issue_timestamp"].nunique()),
        "models": sorted(predictions["model"].unique()),
        "joint_feature_count": len(joint_columns),
        "no_proton_feature_count": len(no_proton_columns),
        "matched_onset_episode_units": matched_events,
        "quiet_block_units": quiet_blocks,
        "folds": fold_receipts,
        "primary_contrasts": contrasts,
        "claim_boundary": {
            "external_public_methodological_evidence": True,
            "operational_equivalence_of_retrospectively_fused_features": False,
            "state_of_the_art_model_claim": False,
            "paper_model_reproduction": False,
            "universal_literature_bias_claim": False,
            "protected_local_post_2025_outcomes_accessed": False,
        },
    }
    dump_json(out / "summary.json", summary)

    shutil.copy2(rolling_path, out / "upstream_rolling_combinded_seq_24hours.csv")
    shutil.copy2(event_path, out / "upstream_GOES_operational_event_catalog.csv")
    hash_names = [
        "predictions.csv", "fold_receipts.json", "feature_schema.json", "descriptive_results.csv",
        "per_fold_results.csv", "matched_episode_sensitivity_predictions.csv", "matched_episode_point_results.csv",
        "shared_bootstrap_draws.npz", "bootstrap_contrasts.json", "summary.json",
        "upstream_rolling_combinded_seq_24hours.csv", "upstream_GOES_operational_event_catalog.csv",
    ]
    dump_json(out / "evidence_hashes.json", {name: sha256_file(out / name) for name in hash_names})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.upstream_root, args.output)
    print(json.dumps({
        "status": summary["status"],
        "score_rows_unique": summary["score_rows_unique"],
        "matched_onset_episode_units": summary["matched_onset_episode_units"],
        "primary_contrasts": summary["primary_contrasts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
