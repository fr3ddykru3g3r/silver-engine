from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tools.run_freshness_crossover_study_v1 as fresh
from tools.episode_benchmark_statistics import (
    SharedBootstrapDraws,
    bootstrap_metric_arrays,
    build_shared_stratified_bootstrap_draws,
    percentile_interval,
    unit_confusion_contributions,
    weighted_binary_metrics,
)
from tools.episode_normalized_benchmark import rank_stability
from tools.episode_semantics import (
    OperationalEpisode,
    attrition_counts,
    build_threshold_episodes,
    classify_window_from_episode_intervals,
    episode_normalized_occurrence_weights,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"
PRE = CFG / "episode_normalized_causal_benchmark_v1_preregistration_2026-09-09.json"
MECH = CFG / "episode_normalized_causal_benchmark_v1_mechanism_decomposition_2026-09-09.json"
OOF = CFG / "episode_normalized_causal_benchmark_v1_expanding_oof_2026-09-09.json"
SEED = 20260909
BOOT_REPS = 10_000
UA = "IRIS-SEP-episode-normalized-development-benchmark/1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n", encoding="utf-8")


def retry_fetch(session: requests.Session, url: str, max_bytes: int = 60_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=90, headers={"User-Agent": UA})
            response.raise_for_status()
            body = response.content
            if not body or len(body) > max_bytes:
                raise RuntimeError(f"bad bounded download {url}: {len(body)} bytes")
            return body
        except Exception as exc:
            last = exc
            if attempt == 4:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after bounded retries: {url}") from last


def load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    mech = json.loads(MECH.read_text(encoding="utf-8"))
    oof = json.loads(OOF.read_text(encoding="utf-8"))
    if pre["status"] != "PREREGISTERED_DESIGN_ONLY_PROTECTED_OUTCOMES_UNTOUCHED":
        raise RuntimeError("unexpected preregistration state")
    if mech["status"] != "FROZEN_BEFORE_NEW_REAL_DATA_SCORING":
        raise RuntimeError("mechanism decomposition was not frozen")
    if oof["status"] != "FROZEN_BEFORE_MODEL_SCORE_INSPECTION":
        raise RuntimeError("OOF contract was not frozen")
    return pre, mech, oof


def acquire(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Replace only the transport helper with bounded retry semantics. URLs,
    # file parsing, channel definitions and feature construction remain the
    # already-frozen freshness-study implementation.
    fresh.fetch = retry_fetch
    proton, xrs = fresh.acquire(out)
    return proton, xrs


def year_role(issue: pd.Timestamp) -> str:
    y = issue.year
    if 2011 <= y <= 2014:
        return "fit"
    if y == 2015:
        return "calibration"
    if y == 2016:
        return "threshold"
    if y == 2017:
        return "score"
    return "outside"


def quiet_block_id(issue: pd.Timestamp) -> str:
    day = issue.normalize()
    monday = day - pd.Timedelta(days=int(day.weekday()))
    return f"QUIET::{monday.strftime('%Y-%m-%d')}"


def _valid_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def outcome_support(proton: pd.DataFrame, issue: pd.Timestamp) -> tuple[bool, str, float | None]:
    times = pd.DatetimeIndex(proton["time"])
    ns = times.asi8
    values = proton["proton"].to_numpy(float)
    ins = issue.value
    end = (issue + pd.Timedelta(hours=24)).value

    j = int(np.searchsorted(ns, ins, side="right") - 1)
    if j < 0 or not np.isfinite(values[j]) or values[j] < 0:
        return False, "issue_support_invalid", None
    if ins - ns[j] > int(5 * 60 * 1e9):
        return False, "issue_support_stale", None

    first = int(np.searchsorted(ns, ins, side="left"))
    last_excl = int(np.searchsorted(ns, end, side="left"))
    if first >= last_excl:
        return False, "empty_outcome", float(values[j])
    future_ns = ns[first:last_excl]
    future_v = values[first:last_excl]
    if end - future_ns[-1] > int(5 * 60 * 1e9):
        return False, "outcome_end_gap", float(values[j])
    chain_ns = np.concatenate(([ns[j]], future_ns))
    chain_v = np.concatenate(([values[j]], future_v))
    if np.any(~np.isfinite(chain_v)) or np.any(chain_v < 0):
        return False, "outcome_nonfinite", float(values[j])
    if len(chain_ns) > 1 and np.any(np.diff(chain_ns) > int(5 * 60 * 1e9)):
        return False, "outcome_gap", float(values[j])
    return True, "ok", float(values[j])


def build_candidate_table(proton: pd.DataFrame, xrs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = build_threshold_episodes(
        proton["time"], proton["proton"], threshold=10.0, maximum_gap_minutes=5.0, id_prefix="IRIS_DEV_EP_"
    )
    episode_table = pd.DataFrame(
        [
            {
                "episode_id": ep.episode_id,
                "start": ep.start,
                "end": ep.end,
                "duration_hours": (ep.end - ep.start).total_seconds() / 3600.0,
            }
            for ep in episodes
        ]
    )

    rows: list[dict[str, Any]] = []
    for issue in pd.date_range("2011-01-02", "2017-11-29", freq="D", tz="UTC"):
        role = year_role(issue)
        base: dict[str, Any] = {
            "issue_timestamp": issue,
            "role": role,
            "year": int(issue.year),
            "quiet_block_id": quiet_block_id(issue),
            "source_proton_ok": False,
            "source_xrs_ok": False,
            "outcome_resolved": False,
            "outcome_reason": None,
            "issue_proton_pfu": np.nan,
            "standard_occurrence_label": np.nan,
            "eligibility_code": "EXCLUDED_BY_FROZEN_BOUNDARY" if role == "outside" else None,
            "onset_episode_id": None,
            "occurrence_episode_id_unique": None,
            "occurrence_episode_match_count": 0,
            "standard_interval_consistent": None,
        }
        if role == "outside":
            rows.append(base)
            continue

        resolved, outcome_reason, issue_p = outcome_support(proton, issue)
        base["outcome_resolved"] = resolved
        base["outcome_reason"] = outcome_reason
        base["issue_proton_pfu"] = issue_p if issue_p is not None else np.nan
        if not resolved:
            base["eligibility_code"] = "UNRESOLVED_GAP"
            rows.append(base)
            continue

        try:
            pf = fresh.family_features(proton, issue, 0, "proton")
            xf = fresh.family_features(xrs, issue, 0, "xrs")
        except Exception:
            base["eligibility_code"] = "SOURCE_UNAVAILABLE"
            rows.append(base)
            continue
        proton_ok = (
            _valid_number(pf.get("p_coverage"))
            and float(pf["p_coverage"]) >= 0.95
            and _valid_number(pf.get("p_age"))
            and float(pf["p_age"]) <= 10
        )
        xrs_ok = (
            _valid_number(xf.get("a_coverage"))
            and _valid_number(xf.get("b_coverage"))
            and float(xf["a_coverage"]) >= 0.95
            and float(xf["b_coverage"]) >= 0.95
            and _valid_number(xf.get("a_age"))
            and _valid_number(xf.get("b_age"))
            and float(xf["a_age"]) <= 2
            and float(xf["b_age"]) <= 2
        )
        base["source_proton_ok"] = bool(proton_ok)
        base["source_xrs_ok"] = bool(xrs_ok)
        if not (proton_ok and xrs_ok):
            base["eligibility_code"] = "SOURCE_UNAVAILABLE"
            rows.append({**base, **pf, **xf})
            continue

        horizon_end = issue + pd.Timedelta(hours=24)
        cls = classify_window_from_episode_intervals(issue, horizon_end, episodes)
        base["eligibility_code"] = cls.eligibility_code
        base["onset_episode_id"] = cls.onset_episode_id
        base["occurrence_episode_match_count"] = len(cls.occurrence_episode_ids)
        if len(cls.occurrence_episode_ids) == 1:
            base["occurrence_episode_id_unique"] = cls.occurrence_episode_ids[0]
        base["standard_occurrence_label"] = int(cls.interval_occurrence_label)

        # Independent flux-window check for the standard target.
        times = pd.DatetimeIndex(proton["time"])
        ns = times.asi8
        v = proton["proton"].to_numpy(float)
        a = int(np.searchsorted(ns, issue.value, side="left"))
        b = int(np.searchsorted(ns, horizon_end.value, side="left"))
        flux_label = int(np.any(v[a:b] >= 10.0))
        base["standard_interval_consistent"] = bool(flux_label == cls.interval_occurrence_label)
        if not base["standard_interval_consistent"]:
            raise RuntimeError(f"episode/flux occurrence disagreement at {issue}")
        rows.append({**base, **pf, **xf})

    frame = pd.DataFrame(rows)
    if frame["eligibility_code"].isna().any():
        raise RuntimeError("attrition ledger contains unset terminal codes")
    if len(frame) != len(pd.date_range("2011-01-02", "2017-11-29", freq="D", tz="UTC")):
        raise RuntimeError("candidate issue count changed unexpectedly")
    return frame, episode_table


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    p = sorted([c for c in frame.columns if c.startswith("p_")])
    x = sorted([c for c in frame.columns if c.startswith("a_") or c.startswith("b_")])
    if not p or not x:
        raise RuntimeError("expected proton and XRS feature families")
    return p, x, p + x


def modelable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[
        frame["source_proton_ok"].eq(True)
        & frame["source_xrs_ok"].eq(True)
        & frame["outcome_resolved"].eq(True)
        & frame["standard_occurrence_label"].isin([0, 1])
    ].copy()
    if out.empty:
        raise RuntimeError("no modelable rows")
    return out


def fit_elastic_net(X: pd.DataFrame, y: np.ndarray):
    pipe = make_pipeline(
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
    pipe.fit(X, y)
    return pipe


def predict_elastic(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def fit_xgb_ensemble(X: pd.DataFrame, y: np.ndarray):
    return fresh.fit_ensemble(X, y)


def predict_xgb(models: Sequence[Any], X: pd.DataFrame) -> np.ndarray:
    return np.asarray(fresh.pred_ens(models, X), dtype=float)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    _, threshold, _ = fresh.best_tss(y, p)
    if threshold is None:
        raise RuntimeError("threshold selection failed")
    return float(threshold)


def fit_intercept_calibration(p: np.ndarray, y: np.ndarray) -> float:
    return float(fresh.fit_intercept(p, y))


def apply_intercept_calibration(p: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(fresh.calibrate(p, intercept), dtype=float)


def save_strict_models(out: Path, models: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_dir = out / "strict_models"
    model_dir.mkdir(exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for name, model in models.items():
        if name.startswith("xgb_"):
            for idx, member in enumerate(model):
                path = model_dir / f"{name}_seed_member_{idx}.json"
                member.save_model(path)
                manifest.append({"model": name, "member": idx, "path": str(path.relative_to(out)), "sha256": sha256_bytes(path.read_bytes())})
        elif name == "elastic_net_joint":
            path = model_dir / f"{name}.joblib"
            joblib.dump(model, path)
            manifest.append({"model": name, "path": str(path.relative_to(out)), "sha256": sha256_bytes(path.read_bytes())})
    return manifest


def make_prediction_frame(
    score: pd.DataFrame,
    probs: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    fold_name: str,
) -> pd.DataFrame:
    occ_ids = score["occurrence_episode_id_unique"].where(score["occurrence_episode_id_unique"].notna(), None).tolist()
    labels = score["standard_occurrence_label"].astype(int).tolist()
    occ_weights = episode_normalized_occurrence_weights(occ_ids, labels)

    onset_mask = score["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])
    onset_positive = score["eligibility_code"].eq("ELIGIBLE_ONSET_POSITIVE")
    onset_counts = Counter(score.loc[onset_positive, "onset_episode_id"].dropna().astype(str))
    onset_weights = np.zeros(len(score), dtype=float)
    onset_episode_weights = np.zeros(len(score), dtype=float)
    for i, (_, row) in enumerate(score.iterrows()):
        if row["eligibility_code"] == "ELIGIBLE_ONSET_NEGATIVE":
            onset_weights[i] = 1.0
            onset_episode_weights[i] = 1.0
        elif row["eligibility_code"] == "ELIGIBLE_ONSET_POSITIVE":
            onset_weights[i] = 1.0
            onset_episode_weights[i] = 1.0 / onset_counts[str(row["onset_episode_id"])]

    rows: list[dict[str, Any]] = []
    for model_name, probability in probs.items():
        threshold = float(thresholds[model_name])
        for i, (_, row) in enumerate(score.iterrows()):
            p = float(probability[i])
            rows.append(
                {
                    "fold": fold_name,
                    "issue_timestamp": row["issue_timestamp"],
                    "year": int(row["year"]),
                    "model": model_name,
                    "probability": p,
                    "threshold": threshold,
                    "binary_alert": int(p >= threshold),
                    "standard_occurrence_label": int(row["standard_occurrence_label"]),
                    "eligibility_code": row["eligibility_code"],
                    "onset_label": 1 if row["eligibility_code"] == "ELIGIBLE_ONSET_POSITIVE" else (0 if row["eligibility_code"] == "ELIGIBLE_ONSET_NEGATIVE" else np.nan),
                    "occurrence_episode_id": row["occurrence_episode_id_unique"],
                    "onset_episode_id": row["onset_episode_id"],
                    "occurrence_episode_match_count": int(row["occurrence_episode_match_count"]),
                    "quiet_block_id": row["quiet_block_id"],
                    "standard_window_weight": 1.0,
                    "episode_normalized_occurrence_weight": float(occ_weights[i]),
                    "new_onset_weight": float(onset_weights[i]),
                    "episode_normalized_onset_weight": float(onset_episode_weights[i]),
                    "issue_proton_pfu": float(row["issue_proton_pfu"]),
                    "source_proton_ok": bool(row["source_proton_ok"]),
                    "source_xrs_ok": bool(row["source_xrs_ok"]),
                    "outcome_resolved": bool(row["outcome_resolved"]),
                }
            )
    return pd.DataFrame(rows)


def fit_strict(frame: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    pcols, xcols, joint = feature_columns(frame)
    data = modelable(frame)
    fit = data[data["role"] == "fit"].copy()
    cal = data[data["role"] == "calibration"].copy()
    thr = data[data["role"] == "threshold"].copy()
    score = data[data["role"] == "score"].copy()
    for name, df in (("fit", fit), ("calibration", cal), ("threshold", thr), ("score", score)):
        if df.empty or df["standard_occurrence_label"].nunique() < 2:
            raise RuntimeError(f"strict role {name} lacks binary support")

    yfit = fit["standard_occurrence_label"].astype(int).to_numpy()
    ycal = cal["standard_occurrence_label"].astype(int).to_numpy()
    ythr = thr["standard_occurrence_label"].astype(int).to_numpy()

    xgb_joint = fit_xgb_ensemble(fit[joint], yfit)
    xgb_xrs = fit_xgb_ensemble(fit[xcols], yfit)
    elastic = fit_elastic_net(fit[joint], yfit)
    models = {"xgb_joint": xgb_joint, "xgb_xrs_only": xgb_xrs, "elastic_net_joint": elastic}

    raw_cal = {
        "xgb_joint": predict_xgb(xgb_joint, cal[joint]),
        "xgb_xrs_only": predict_xgb(xgb_xrs, cal[xcols]),
        "elastic_net_joint": predict_elastic(elastic, cal[joint]),
    }
    intercepts = {name: fit_intercept_calibration(p, ycal) for name, p in raw_cal.items()}

    def calibrated(name: str, p: np.ndarray) -> np.ndarray:
        return apply_intercept_calibration(p, intercepts[name])

    thr_probs = {
        "xgb_joint": calibrated("xgb_joint", predict_xgb(xgb_joint, thr[joint])),
        "xgb_xrs_only": calibrated("xgb_xrs_only", predict_xgb(xgb_xrs, thr[xcols])),
        "elastic_net_joint": calibrated("elastic_net_joint", predict_elastic(elastic, thr[joint])),
    }
    thresholds = {name: best_threshold(ythr, p) for name, p in thr_probs.items()}
    fit_prev = float(yfit.mean())
    thresholds["fit_prevalence_climatology"] = fit_prev
    thresholds["current_proton_active_diagnostic"] = 0.5

    score_probs = {
        "xgb_joint": calibrated("xgb_joint", predict_xgb(xgb_joint, score[joint])),
        "xgb_xrs_only": calibrated("xgb_xrs_only", predict_xgb(xgb_xrs, score[xcols])),
        "elastic_net_joint": calibrated("elastic_net_joint", predict_elastic(elastic, score[joint])),
        "fit_prevalence_climatology": np.full(len(score), fit_prev, dtype=float),
        "current_proton_active_diagnostic": (score["issue_proton_pfu"].to_numpy(float) >= 10.0).astype(float),
    }
    pred = make_prediction_frame(score, score_probs, thresholds, fold_name="STRICT_2017")
    manifest = save_strict_models(out, models)
    info = {
        "role_counts": {r: {"rows": int(len(df)), "positive_windows": int(df["standard_occurrence_label"].sum())} for r, df in (("fit", fit), ("calibration", cal), ("threshold", thr), ("score", score))},
        "fit_prevalence": fit_prev,
        "calibration_intercepts": intercepts,
        "thresholds": thresholds,
        "model_manifest": manifest,
        "feature_columns": {"proton": pcols, "xrs": xcols, "joint": joint},
    }
    return pred, info


def fit_oof(frame: pd.DataFrame, contract: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pcols, xcols, joint = feature_columns(frame)
    data = modelable(frame)
    all_pred: list[pd.DataFrame] = []
    fold_receipts: list[dict[str, Any]] = []
    for fold in contract["folds"]:
        fit_start = pd.Timestamp(fold["fit_start"], tz="UTC")
        fit_end = pd.Timestamp(fold["fit_end_exclusive"], tz="UTC")
        th_start = pd.Timestamp(fold["threshold_start"], tz="UTC")
        th_end = pd.Timestamp(fold["threshold_end_exclusive"], tz="UTC")
        sc_start = pd.Timestamp(fold["score_start"], tz="UTC")
        sc_end = pd.Timestamp(fold["score_end_exclusive"], tz="UTC")
        fit = data[(data["issue_timestamp"] >= fit_start) & (data["issue_timestamp"] < fit_end)].copy()
        thr = data[(data["issue_timestamp"] >= th_start) & (data["issue_timestamp"] < th_end)].copy()
        score = data[(data["issue_timestamp"] >= sc_start) & (data["issue_timestamp"] < sc_end)].copy()
        for name, df in (("fit", fit), ("threshold", thr), ("score", score)):
            if df.empty or df["standard_occurrence_label"].nunique() < 2:
                raise RuntimeError(f"{fold['name']} {name} lacks binary support")
        yfit = fit["standard_occurrence_label"].astype(int).to_numpy()
        ythr = thr["standard_occurrence_label"].astype(int).to_numpy()
        xgb_joint = fit_xgb_ensemble(fit[joint], yfit)
        xgb_xrs = fit_xgb_ensemble(fit[xcols], yfit)
        elastic = fit_elastic_net(fit[joint], yfit)
        thr_probs = {
            "xgb_joint": predict_xgb(xgb_joint, thr[joint]),
            "xgb_xrs_only": predict_xgb(xgb_xrs, thr[xcols]),
            "elastic_net_joint": predict_elastic(elastic, thr[joint]),
        }
        thresholds = {name: best_threshold(ythr, p) for name, p in thr_probs.items()}
        fit_prev = float(yfit.mean())
        thresholds["fit_prevalence_climatology"] = fit_prev
        thresholds["current_proton_active_diagnostic"] = 0.5
        score_probs = {
            "xgb_joint": predict_xgb(xgb_joint, score[joint]),
            "xgb_xrs_only": predict_xgb(xgb_xrs, score[xcols]),
            "elastic_net_joint": predict_elastic(elastic, score[joint]),
            "fit_prevalence_climatology": np.full(len(score), fit_prev, dtype=float),
            "current_proton_active_diagnostic": (score["issue_proton_pfu"].to_numpy(float) >= 10.0).astype(float),
        }
        all_pred.append(make_prediction_frame(score, score_probs, thresholds, fold_name=fold["name"]))
        fold_receipts.append(
            {
                "fold": fold["name"],
                "fit_rows": int(len(fit)),
                "fit_positive_windows": int(yfit.sum()),
                "threshold_rows": int(len(thr)),
                "threshold_positive_windows": int(ythr.sum()),
                "score_rows": int(len(score)),
                "score_positive_windows": int(score["standard_occurrence_label"].sum()),
                "fit_prevalence": fit_prev,
                "thresholds": thresholds,
            }
        )
    result = pd.concat(all_pred, ignore_index=True)
    if result.duplicated(["fold", "issue_timestamp", "model"]).any():
        raise RuntimeError("duplicate OOF prediction row")
    return result, fold_receipts


def view_frame(pred: pd.DataFrame, model: str, view: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pred[pred["model"] == model].copy()
    if view == "WINDOW_OCCURRENCE_STANDARD":
        use = df.copy()
        y = use["standard_occurrence_label"].astype(int).to_numpy()
        w = np.ones(len(use), dtype=float)
    elif view == "EPISODE_NORMALIZED_OCCURRENCE":
        # Ambiguous/unmapped positives have zero normalized weight and are not
        # treated as independent physical episodes.
        use = df.copy()
        y = use["standard_occurrence_label"].astype(int).to_numpy()
        w = use["episode_normalized_occurrence_weight"].to_numpy(float)
    elif view == "NEW_ONSET_CAUSAL":
        use = df[df["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
        y = use["onset_label"].astype(int).to_numpy()
        w = use["new_onset_weight"].to_numpy(float)
    elif view == "EPISODE_NORMALIZED_ONSET":
        use = df[df["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
        y = use["onset_label"].astype(int).to_numpy()
        w = use["episode_normalized_onset_weight"].to_numpy(float)
    else:
        raise ValueError(view)
    p = use["probability"].to_numpy(float)
    return use, y, p, w


def extended_metrics(use: pd.DataFrame, y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    threshold = float(use["threshold"].iloc[0])
    base = dict(weighted_binary_metrics(y, p, threshold, w))
    if len(np.unique(y[w > 0])) == 2 and np.sum(w) > 0:
        try:
            base["auroc"] = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            base["auroc"] = float("nan")
        try:
            base["auprc"] = float(average_precision_score(y, p, sample_weight=w))
        except Exception:
            base["auprc"] = float("nan")
    else:
        base["auroc"] = float("nan")
        base["auprc"] = float("nan")
    base["rows"] = int(len(use))
    base["positive_rows"] = int(np.sum(y == 1))
    base["threshold"] = threshold
    return base


def persistence_diagnostic(pred: pd.DataFrame) -> list[dict[str, Any]]:
    active = pred[pred["eligibility_code"] == "ALREADY_ACTIVE_PERSISTENCE"].copy()
    out: list[dict[str, Any]] = []
    if active.empty:
        return out
    for model, group in active.groupby("model"):
        out.append(
            {
                "model": model,
                "persistence_rows": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "median_probability": float(group["probability"].median()),
                "alert_rate": float(group["binary_alert"].mean()),
            }
        )
    return sorted(out, key=lambda r: r["model"])


def bootstrap_for_predictions(pred: pd.DataFrame, *, prefix: str, out: Path) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    base = pred[pred["model"] == sorted(pred["model"].unique())[0]].copy()
    pos = base[(base["standard_occurrence_label"] == 1) & base["occurrence_episode_id"].notna()]
    neg = base[base["standard_occurrence_label"] == 0]
    pos_units = [f"EPISODE::{x}" for x in pos["occurrence_episode_id"].astype(str)]
    neg_units = neg["quiet_block_id"].astype(str).tolist()
    draws = build_shared_stratified_bootstrap_draws(pos_units, neg_units, replicates=BOOT_REPS, seed=SEED)
    np.savez_compressed(
        out / f"{prefix}_shared_bootstrap_draws.npz",
        positive_indices=draws.positive_indices,
        negative_indices=draws.negative_indices,
        positive_units=np.asarray(draws.positive_units, dtype=str),
        negative_units=np.asarray(draws.negative_units, dtype=str),
    )

    arrays: dict[tuple[str, str], np.ndarray] = {}
    result: dict[str, Any] = {
        "replicates": BOOT_REPS,
        "positive_episode_units": len(draws.positive_units),
        "negative_quiet_block_units": len(draws.negative_units),
        "models": {},
    }
    views = ["WINDOW_OCCURRENCE_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET_CAUSAL", "EPISODE_NORMALIZED_ONSET"]
    for model in sorted(pred["model"].unique()):
        result["models"][model] = {}
        for view in views:
            use, y, p, w = view_frame(pred, model, view)
            # Primary paired-bootstrap cohort requires physical episode identity
            # for positives. Standard point metrics remain available separately.
            positive_ids = []
            negative_ids = []
            pos_rows = []
            neg_rows = []
            for i, (_, row) in enumerate(use.iterrows()):
                if y[i] == 1:
                    ep_id = row["onset_episode_id"] if view in {"NEW_ONSET_CAUSAL", "EPISODE_NORMALIZED_ONSET"} else row["occurrence_episode_id"]
                    if pd.isna(ep_id) or ep_id is None:
                        continue
                    positive_ids.append(f"EPISODE::{ep_id}")
                    pos_rows.append(i)
                else:
                    negative_ids.append(str(row["quiet_block_id"]))
                    neg_rows.append(i)
            threshold = float(use["threshold"].iloc[0])
            alerts = (p >= threshold).astype(int)
            pos_contrib = unit_confusion_contributions(
                [positive_ids[j] for j in range(len(pos_rows))],
                y[pos_rows], alerts[pos_rows], w[pos_rows], draws.positive_units,
            )
            neg_contrib = unit_confusion_contributions(
                [negative_ids[j] for j in range(len(neg_rows))],
                y[neg_rows], alerts[neg_rows], w[neg_rows], draws.negative_units,
            )
            metrics = bootstrap_metric_arrays(draws, pos_contrib, neg_contrib)
            arrays[(model, view)] = metrics["tss"]
            lo, hi = percentile_interval(metrics["tss"])
            flo, fhi = percentile_interval(metrics["far"])
            result["models"][model][view] = {
                "tss_interval_95": [lo, hi],
                "far_interval_95": [flo, fhi],
                "finite_tss_replicates": int(np.isfinite(metrics["tss"]).sum()),
            }
    return result, arrays


def point_results(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = sorted(pred["model"].unique())
    views = ["WINDOW_OCCURRENCE_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET_CAUSAL", "EPISODE_NORMALIZED_ONSET"]
    for model in models:
        for view in views:
            use, y, p, w = view_frame(pred, model, view)
            met = extended_metrics(use, y, p, w)
            rows.append({"model": model, "view": view, **met})
    table = pd.DataFrame(rows)
    nontrivial = [m for m in ["xgb_joint", "elastic_net_joint", "xgb_xrs_only"] if m in models]
    standard = {m: float(table[(table.model == m) & (table.view == "WINDOW_OCCURRENCE_STANDARD")].iloc[0].tss) for m in nontrivial}
    normalized = {m: float(table[(table.model == m) & (table.view == "EPISODE_NORMALIZED_OCCURRENCE")].iloc[0].tss) for m in nontrivial}
    onset = {m: float(table[(table.model == m) & (table.view == "NEW_ONSET_CAUSAL")].iloc[0].tss) for m in nontrivial}
    return table, {
        "standard_to_normalized_occurrence_rank": rank_stability(standard, normalized),
        "standard_to_new_onset_rank": rank_stability(standard, onset),
    }


def bootstrap_contrasts(arrays: Mapping[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    models = sorted({k[0] for k in arrays})
    for model in models:
        std = arrays[(model, "WINDOW_OCCURRENCE_STANDARD")]
        norm = arrays[(model, "EPISODE_NORMALIZED_OCCURRENCE")]
        onset = arrays[(model, "NEW_ONSET_CAUSAL")]
        onset_norm = arrays[(model, "EPISODE_NORMALIZED_ONSET")]
        for name, delta in {
            "multiplicity_only_tss_shift": norm - std,
            "persistence_exclusion_tss_shift": onset - norm,
            "onset_episode_weighting_tss_shift": onset_norm - onset,
        }.items():
            lo, hi = percentile_interval(delta)
            result[f"{model}::{name}"] = {
                "interval_95": [lo, hi],
                "median": float(np.nanmedian(delta)),
                "probability_positive": float(np.nanmean(delta > 0)),
                "probability_negative": float(np.nanmean(delta < 0)),
            }
    if ("xgb_joint", "NEW_ONSET_CAUSAL") in arrays and ("xgb_xrs_only", "NEW_ONSET_CAUSAL") in arrays:
        delta = arrays[("xgb_joint", "NEW_ONSET_CAUSAL")] - arrays[("xgb_xrs_only", "NEW_ONSET_CAUSAL")]
        lo, hi = percentile_interval(delta)
        result["proton_aware_minus_blind_onset_tss"] = {"interval_95": [lo, hi], "median": float(np.nanmedian(delta))}
    return result


def rank_reversal_bootstrap(arrays: Mapping[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    models = [m for m in ["xgb_joint", "elastic_net_joint", "xgb_xrs_only"] if (m, "WINDOW_OCCURRENCE_STANDARD") in arrays]
    out: dict[str, Any] = {}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            std_delta = arrays[(a, "WINDOW_OCCURRENCE_STANDARD")] - arrays[(b, "WINDOW_OCCURRENCE_STANDARD")]
            norm_delta = arrays[(a, "EPISODE_NORMALIZED_OCCURRENCE")] - arrays[(b, "EPISODE_NORMALIZED_OCCURRENCE")]
            onset_delta = arrays[(a, "NEW_ONSET_CAUSAL")] - arrays[(b, "NEW_ONSET_CAUSAL")]
            valid = np.isfinite(std_delta) & np.isfinite(norm_delta)
            valid2 = np.isfinite(std_delta) & np.isfinite(onset_delta)
            out[f"{a}__vs__{b}"] = {
                "standard_vs_normalized_occurrence_reversal_fraction": float(np.mean((std_delta[valid] * norm_delta[valid]) < 0)) if np.any(valid) else float("nan"),
                "standard_vs_onset_reversal_fraction": float(np.mean((std_delta[valid2] * onset_delta[valid2]) < 0)) if np.any(valid2) else float("nan"),
            }
    return out


def episode_counts(pred: pd.DataFrame) -> dict[str, Any]:
    base = pred[pred["model"] == sorted(pred["model"].unique())[0]].copy()
    occurrence_positive = base[base["standard_occurrence_label"] == 1]
    onset_positive = base[base["eligibility_code"] == "ELIGIBLE_ONSET_POSITIVE"]
    persistence = base[base["eligibility_code"] == "ALREADY_ACTIVE_PERSISTENCE"]
    mapped = occurrence_positive[occurrence_positive["occurrence_episode_id"].notna()]
    distinct = int(mapped["occurrence_episode_id"].nunique())
    return {
        "score_issue_rows": int(len(base)),
        "standard_positive_windows": int(len(occurrence_positive)),
        "distinct_mapped_positive_episodes": distinct,
        "episode_multiplicity_factor": float(len(mapped) / distinct) if distinct else float("nan"),
        "new_onset_positive_windows": int(len(onset_positive)),
        "distinct_new_onset_episodes": int(onset_positive["onset_episode_id"].nunique()),
        "persistence_windows": int(len(persistence)),
        "ambiguous_positive_windows": int(((base["standard_occurrence_label"] == 1) & base["occurrence_episode_id"].isna()).sum()),
    }


def make_figures(out: Path, point_oof: pd.DataFrame, point_strict: pd.DataFrame, counts_oof: Mapping[str, Any], persistence_oof: Sequence[Mapping[str, Any]]) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    order = ["WINDOW_OCCURRENCE_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET_CAUSAL", "EPISODE_NORMALIZED_ONSET"]
    show_models = [m for m in ["xgb_joint", "elastic_net_joint", "xgb_xrs_only", "current_proton_active_diagnostic"] if m in set(point_oof.model)]
    x = np.arange(len(order), dtype=float)
    width = 0.8 / max(1, len(show_models))
    plt.figure(figsize=(10, 6))
    for j, model in enumerate(show_models):
        vals = [float(point_oof[(point_oof.model == model) & (point_oof.view == v)].iloc[0].tss) for v in order]
        plt.bar(x - 0.4 + width / 2 + j * width, vals, width=width, label=model)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, ["Standard", "Episode-normalized\noccurrence", "New-onset", "Episode-normalized\nonset"])
    plt.ylabel("TSS")
    plt.title("How evaluation definition changes measured skill (development OOF)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "01_tss_by_evaluation_view.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Positive windows", "Physical episodes", "New-onset windows", "Persistence windows"]
    vals = [counts_oof["standard_positive_windows"], counts_oof["distinct_mapped_positive_episodes"], counts_oof["new_onset_positive_windows"], counts_oof["persistence_windows"]]
    plt.bar(labels, vals)
    plt.ylabel("Count")
    plt.title("Windows are not physical SEP episodes")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "02_windows_vs_physical_episodes.png", dpi=180)
    plt.close()

    if persistence_oof:
        p = pd.DataFrame(persistence_oof).sort_values("model")
        plt.figure(figsize=(8, 5))
        plt.bar(p["model"], p["alert_rate"])
        plt.ylim(0, 1)
        plt.ylabel("Alert rate on already-active persistence cases")
        plt.title("Persistence recognition by model")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(figdir / "03_persistence_alert_rate.png", dpi=180)
        plt.close()

    # Strict-vs-OOF sensitivity for the main joint model.
    vals = []
    labs = []
    for tag, table in (("OOF 2014-17", point_oof), ("Strict 2017", point_strict)):
        row = table[(table.model == "xgb_joint") & (table.view == "NEW_ONSET_CAUSAL")]
        if len(row):
            labs.append(tag); vals.append(float(row.iloc[0].tss))
    plt.figure(figsize=(6, 4))
    plt.bar(labs, vals)
    plt.axhline(0, linewidth=1)
    plt.ylabel("New-onset TSS")
    plt.title("Sensitivity to evaluation period")
    plt.tight_layout()
    plt.savefig(figdir / "04_strict_vs_expanding_oof.png", dpi=180)
    plt.close()


def hash_outputs(out: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.relative_to(out)): sha256_bytes(path.read_bytes()) for path in paths if path.is_file()}


def run(out: Path) -> dict[str, Any]:
    pre, mech, oof_contract = load_contracts()
    out.mkdir(parents=True, exist_ok=False)
    proton, xrs = acquire(out)
    candidates, episodes = build_candidate_table(proton, xrs)
    candidates.to_csv(out / "candidate_attrition_ledger.csv", index=False)
    episodes.to_csv(out / "derived_threshold_episodes.csv", index=False)

    strict_pred, strict_info = fit_strict(candidates, out)
    oof_pred, fold_receipts = fit_oof(candidates, oof_contract)
    strict_pred.to_csv(out / "predictions_strict_2017.csv", index=False)
    oof_pred.to_csv(out / "predictions_expanding_oof_2014_2017.csv", index=False)

    strict_point, strict_rank = point_results(strict_pred)
    oof_point, oof_rank = point_results(oof_pred)
    strict_point.to_csv(out / "results_strict_2017.csv", index=False)
    oof_point.to_csv(out / "results_expanding_oof_2014_2017.csv", index=False)

    strict_boot, strict_arrays = bootstrap_for_predictions(strict_pred, prefix="strict_2017", out=out)
    oof_boot, oof_arrays = bootstrap_for_predictions(oof_pred, prefix="expanding_oof", out=out)
    strict_contrasts = bootstrap_contrasts(strict_arrays)
    oof_contrasts = bootstrap_contrasts(oof_arrays)
    strict_reversals = rank_reversal_bootstrap(strict_arrays)
    oof_reversals = rank_reversal_bootstrap(oof_arrays)

    strict_persistence = persistence_diagnostic(strict_pred)
    oof_persistence = persistence_diagnostic(oof_pred)
    strict_counts = episode_counts(strict_pred)
    oof_counts = episode_counts(oof_pred)

    attrition = {
        "total_candidate_issues": int(len(candidates)),
        "terminal_codes": attrition_counts(candidates["eligibility_code"].tolist()),
        "by_year": {
            str(year): attrition_counts(group["eligibility_code"].tolist())
            for year, group in candidates.groupby("year")
        },
    }
    dump_json(out / "attrition_summary.json", attrition)
    dump_json(out / "strict_training_receipt.json", strict_info)
    dump_json(out / "oof_fold_receipts.json", fold_receipts)
    dump_json(out / "strict_bootstrap_intervals.json", strict_boot)
    dump_json(out / "oof_bootstrap_intervals.json", oof_boot)
    dump_json(out / "strict_contrasts.json", strict_contrasts)
    dump_json(out / "oof_contrasts.json", oof_contrasts)
    dump_json(out / "strict_rank_reversal_bootstrap.json", strict_reversals)
    dump_json(out / "oof_rank_reversal_bootstrap.json", oof_reversals)
    dump_json(out / "strict_persistence_diagnostic.json", strict_persistence)
    dump_json(out / "oof_persistence_diagnostic.json", oof_persistence)

    make_figures(out, oof_point, strict_point, oof_counts, oof_persistence)

    underpowered = oof_counts["distinct_mapped_positive_episodes"] < int(oof_contract["underpowered_reporting"]["minimum_distinct_positive_episodes_for_non_underpowered_development_mechanism_claim"])
    summary = {
        "format": "IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_DEVELOPMENT_RESULT_V1",
        "claim_boundary": {
            "development_only": True,
            "independent_final_evidence": False,
            "protected_post_2025_outcomes_accessed": False,
            "post_result_tuning_performed": False,
        },
        "contracts": {
            "preregistration_sha256": sha256_bytes(PRE.read_bytes()),
            "mechanism_decomposition_sha256": sha256_bytes(MECH.read_bytes()),
            "expanding_oof_sha256": sha256_bytes(OOF.read_bytes()),
        },
        "attrition": attrition,
        "strict_2017": {
            "episode_counts": strict_counts,
            "rank_stability": strict_rank,
            "persistence_diagnostic": strict_persistence,
            "bootstrap_contrasts": strict_contrasts,
            "rank_reversal_bootstrap": strict_reversals,
        },
        "expanding_oof_2014_2017": {
            "episode_counts": oof_counts,
            "rank_stability": oof_rank,
            "persistence_diagnostic": oof_persistence,
            "bootstrap_contrasts": oof_contrasts,
            "rank_reversal_bootstrap": oof_reversals,
            "underpowered": bool(underpowered),
            "underpowered_status": "UNDERPOWERED_DEVELOPMENT_MECHANISM_RESULT" if underpowered else "DEVELOPMENT_EPISODE_SUPPORT_GATE_PASSED",
        },
        "model_score_files": {
            "strict": "results_strict_2017.csv",
            "oof": "results_expanding_oof_2014_2017.csv",
        },
        "prediction_files": {
            "strict": "predictions_strict_2017.csv",
            "oof": "predictions_expanding_oof_2014_2017.csv",
        },
    }
    dump_json(out / "summary.json", summary)

    evidence_paths = [
        out / "candidate_attrition_ledger.csv",
        out / "derived_threshold_episodes.csv",
        out / "predictions_strict_2017.csv",
        out / "predictions_expanding_oof_2014_2017.csv",
        out / "results_strict_2017.csv",
        out / "results_expanding_oof_2014_2017.csv",
        out / "strict_2017_shared_bootstrap_draws.npz",
        out / "expanding_oof_shared_bootstrap_draws.npz",
        out / "summary.json",
        out / "source_manifest.json",
    ]
    hashes = hash_outputs(out, evidence_paths)
    dump_json(out / "evidence_hashes.json", hashes)
    summary["evidence_hashes_sha256"] = sha256_bytes((out / "evidence_hashes.json").read_bytes())
    dump_json(out / "summary.json", summary)

    print(json.dumps({
        "strict_2017": strict_counts,
        "expanding_oof": oof_counts,
        "underpowered": underpowered,
        "oof_rank_standard_to_onset": oof_rank["standard_to_new_onset_rank"],
    }, indent=2, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
