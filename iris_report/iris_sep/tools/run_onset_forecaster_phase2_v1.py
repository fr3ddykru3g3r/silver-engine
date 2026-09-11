"""Development-only Phase II forecaster for genuine 24 h SEP onset.

This module is deliberately isolated from the frozen episode-benchmark result.
It reuses the frozen public-source adapter and episode semantics, but trains new
models only on historical development data. Protected post-2025 outcomes are
not queried here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import tools.run_episode_normalized_development_benchmark_v1 as bench
import tools.run_freshness_crossover_study_v1 as fresh

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "onset_forecaster_phase2_v1_preregistration_2026-09-11.json"
SEEDS = (7, 13, 26, 42, 73)
ELIGIBLE = ("ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE")


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_contract() -> dict[str, Any]:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    if cfg["study_id"] != "IRIS_SEP_ONSET_FORECASTER_PHASE2_V1":
        raise RuntimeError("unexpected Phase II study id")
    if cfg["status"] != "FROZEN_BEFORE_PHASE2_SCORE_INSPECTION":
        raise RuntimeError("Phase II design is not frozen")
    if cfg["separation"]["protected_post_2025_outcomes_may_be_accessed"] is not False:
        raise RuntimeError("protected outcome boundary changed")
    return cfg


def onset_rows(frame: pd.DataFrame, role: str | None = None) -> pd.DataFrame:
    out = frame[frame["eligibility_code"].isin(ELIGIBLE)].copy()
    if role is not None:
        out = out[out["role"].eq(role)].copy()
    out["y_onset"] = out["eligibility_code"].eq("ELIGIBLE_ONSET_POSITIVE").astype(int)
    return out


def safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    av = pd.to_numeric(a, errors="coerce").astype(float)
    bv = pd.to_numeric(b, errors="coerce").astype(float)
    return av / np.where(np.abs(bv) > eps, bv, np.nan)


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    created: list[str] = []

    def add(name: str, values: Iterable[float] | pd.Series) -> None:
        out[name] = np.asarray(values, dtype=float)
        created.append(name)

    add("eng_log1p_p_last", np.log1p(np.maximum(pd.to_numeric(out["p_last"], errors="coerce"), 0)))
    add("eng_log1p_p_max", np.log1p(np.maximum(pd.to_numeric(out["p_max"], errors="coerce"), 0)))
    add("eng_p_last_over_mean", safe_ratio(out["p_last"], out["p_mean"]))
    add("eng_p_max_over_mean", safe_ratio(out["p_max"], out["p_mean"]))
    add("eng_p_last_minus_median", pd.to_numeric(out["p_last"], errors="coerce") - pd.to_numeric(out["p_median"], errors="coerce"))
    add("eng_p_ge5_fraction", safe_ratio(out["p_count_ge_5"], out["p_n"]))
    add("eng_p_ge8_fraction", safe_ratio(out["p_count_ge_8"], out["p_n"]))
    add("eng_b_last_over_mean", safe_ratio(out["b_last"], out["b_mean"]))
    add("eng_b_max_over_mean", safe_ratio(out["b_max"], out["b_mean"]))
    add("eng_a_last_over_mean", safe_ratio(out["a_last"], out["a_mean"]))
    add("eng_b_over_a_last", safe_ratio(out["b_last"], out["a_last"]))
    add("eng_b_over_a_max", safe_ratio(out["b_max"], out["a_max"]))
    add("eng_b_ge1e5_fraction", safe_ratio(out["b_count_ge_1e-05"], out["b_n"]))
    add("eng_b_ge1e4_fraction", safe_ratio(out["b_count_ge_0.0001"], out["b_n"]))
    add("eng_joint_impulsiveness", pd.to_numeric(out["b_max_pos_log_change"], errors="coerce") * (1.0 + pd.to_numeric(out["p_max_pos_log_change"], errors="coerce")))
    return out, created


def xgb_params(seed: int, scale_pos_weight: float, cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["training"]["xgboost"]
    return {
        "n_estimators": int(p["n_estimators"]),
        "learning_rate": float(p["learning_rate"]),
        "max_depth": int(p["max_depth"]),
        "min_child_weight": float(p["min_child_weight"]),
        "subsample": float(p["subsample"]),
        "colsample_bytree": float(p["colsample_bytree"]),
        "reg_lambda": float(p["reg_lambda"]),
        "reg_alpha": float(p["reg_alpha"]),
        "tree_method": str(p["tree_method"]),
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": 2,
        "scale_pos_weight": float(scale_pos_weight),
    }


def class_ratio(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        raise RuntimeError("training split lacks binary onset support")
    return float(neg / pos)


def fit_ensemble(X: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any]) -> list[XGBClassifier]:
    ratio = class_ratio(y)
    models: list[XGBClassifier] = []
    for seed in SEEDS:
        model = XGBClassifier(**xgb_params(seed, ratio, cfg))
        model.fit(X, y)
        models.append(model)
    return models


def predict(models: list[XGBClassifier], X: pd.DataFrame) -> np.ndarray:
    return np.median(np.vstack([m.predict_proba(X)[:, 1] for m in models]), axis=0)


def calibrate_intercept(p_cal: np.ndarray, y_cal: np.ndarray) -> tuple[float, np.ndarray]:
    if len(np.unique(y_cal)) != 2:
        raise RuntimeError("calibration split lacks binary onset support")
    intercept = float(fresh.fit_intercept(np.asarray(p_cal, float), np.asarray(y_cal, int)))
    return intercept, np.asarray(fresh.calibrate(np.asarray(p_cal, float), intercept), float)


def apply_calibration(p: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(fresh.calibrate(np.asarray(p, float), float(intercept)), float)


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    y = np.asarray(y, int)
    alert = np.asarray(p, float) >= float(threshold)
    return {
        "tp": int(np.sum((y == 1) & alert)),
        "fn": int(np.sum((y == 1) & ~alert)),
        "fp": int(np.sum((y == 0) & alert)),
        "tn": int(np.sum((y == 0) & ~alert)),
    }


def derived_metrics(c: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr if np.isfinite(pod) and np.isfinite(fpr) else float("nan")
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fn * fp) / denom if denom else float("nan")
    return {"POD": float(pod), "FPR": float(fpr), "FAR": float(far), "TSS": float(tss), "HSS": float(hss)}


def full_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    c = confusion(y, p, threshold)
    m: dict[str, Any] = {**c, **derived_metrics(c), "threshold": float(threshold)}
    m["Brier"] = float(brier_score_loss(y, p))
    m["AUPRC"] = float(average_precision_score(y, p))
    m["AUROC"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return m


def select_threshold(y: np.ndarray, p: np.ndarray, tolerance: float = 0.02) -> tuple[float, dict[str, Any]]:
    values = np.unique(np.clip(np.asarray(p, float), 0.0, 1.0))
    if len(values) == 0:
        raise RuntimeError("no threshold candidates")
    rows: list[tuple[float, float, float, dict[str, Any]]] = []
    for th in values:
        met = full_metrics(y, p, float(th))
        rows.append((float(met["TSS"]), float(met["FAR"]), float(th), met))
    best_tss = max(row[0] for row in rows)
    near = [row for row in rows if row[0] >= best_tss - tolerance - 1e-12]
    near.sort(key=lambda row: (row[1] if np.isfinite(row[1]) else 1.0, -row[2]))
    chosen = near[0]
    return chosen[2], {"max_tss": best_tss, "chosen": chosen[3], "near_optimal_count": len(near)}


def unit_bootstrap_delta(
    score: pd.DataFrame,
    reference_p: np.ndarray,
    candidate_p: np.ndarray,
    reference_threshold: float,
    candidate_threshold: float,
    *,
    reps: int = 5000,
    seed: int = 20260911,
) -> dict[str, Any]:
    use = score.copy().reset_index(drop=True)
    y = use["y_onset"].to_numpy(int)
    positive_units = sorted(use.loc[y == 1, "onset_episode_id"].dropna().astype(str).unique().tolist())
    negative_units = sorted(use.loc[y == 0, "quiet_block_id"].dropna().astype(str).unique().tolist())
    if not positive_units or not negative_units:
        return {"status": "INSUFFICIENT_UNITS"}
    pos_map = {u: np.where((y == 1) & use["onset_episode_id"].astype(str).eq(u).to_numpy())[0] for u in positive_units}
    neg_map = {u: np.where((y == 0) & use["quiet_block_id"].astype(str).eq(u).to_numpy())[0] for u in negative_units}
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=float)
    for r in range(reps):
        pu = rng.choice(positive_units, size=len(positive_units), replace=True)
        nu = rng.choice(negative_units, size=len(negative_units), replace=True)
        idx = np.concatenate([*(pos_map[str(u)] for u in pu), *(neg_map[str(u)] for u in nu)])
        ref = derived_metrics(confusion(y[idx], reference_p[idx], reference_threshold))["TSS"]
        cand = derived_metrics(confusion(y[idx], candidate_p[idx], candidate_threshold))["TSS"]
        deltas[r] = cand - ref
    finite = deltas[np.isfinite(deltas)]
    return {
        "status": "OK",
        "reps": int(reps),
        "positive_episode_units": len(positive_units),
        "negative_quiet_block_units": len(negative_units),
        "median_delta_tss": float(np.median(finite)),
        "interval_95_delta_tss": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))],
        "fraction_positive": float(np.mean(finite > 0)),
        "note": "development-only physical-unit bootstrap; not independent confirmation"
    }


def run(out: Path) -> dict[str, Any]:
    cfg = load_contract()
    out.mkdir(parents=True, exist_ok=True)
    proton, xrs = bench.acquire(out)
    candidate, episodes = bench.build_candidate_table(proton, xrs)
    frame = bench.modelable(candidate)
    frame, engineered = add_engineered_features(frame)
    pcols, xcols, joint = bench.feature_columns(frame)
    eng_cols = joint + engineered

    onset = onset_rows(frame)
    roles = {role: onset_rows(frame, role) for role in ("fit", "calibration", "threshold", "score")}
    for role, part in roles.items():
        if part.empty or part["y_onset"].nunique() != 2:
            raise RuntimeError(f"{role} onset split lacks binary support")

    # Conventional occurrence-trained XRS baseline, then onset calibration and thresholding.
    occurrence_fit = frame[frame["role"].eq("fit")].copy()
    occurrence_y = occurrence_fit["standard_occurrence_label"].astype(int).to_numpy()
    baseline_models = fit_ensemble(occurrence_fit[xcols], occurrence_y, cfg)

    model_specs = {
        "occurrence_trained_xrs_baseline": (baseline_models, xcols),
        "direct_onset_xrs": (fit_ensemble(roles["fit"][xcols], roles["fit"]["y_onset"].to_numpy(int), cfg), xcols),
        "direct_onset_joint": (fit_ensemble(roles["fit"][joint], roles["fit"]["y_onset"].to_numpy(int), cfg), joint),
        "direct_onset_engineered": (fit_ensemble(roles["fit"][eng_cols], roles["fit"]["y_onset"].to_numpy(int), cfg), eng_cols),
    }

    probabilities: dict[str, dict[str, np.ndarray]] = {}
    thresholds: dict[str, float] = {}
    selection: dict[str, Any] = {}
    intercepts: dict[str, float] = {}

    for name, (models, cols) in model_specs.items():
        raw_cal = predict(models, roles["calibration"][cols])
        intercept, _ = calibrate_intercept(raw_cal, roles["calibration"]["y_onset"].to_numpy(int))
        intercepts[name] = intercept
        probabilities[name] = {}
        for role in roles:
            probabilities[name][role] = apply_calibration(predict(models, roles[role][cols]), intercept)
        th, diag = select_threshold(roles["threshold"]["y_onset"].to_numpy(int), probabilities[name]["threshold"])
        thresholds[name] = th
        selection[name] = diag

    # Threshold-only blend search. Score rows are never used for alpha or threshold selection.
    blend_candidates: list[dict[str, Any]] = []
    ythr = roles["threshold"]["y_onset"].to_numpy(int)
    for alpha in cfg["blend_policy"]["candidate_alphas"]:
        a = float(alpha)
        pthr = a * probabilities["direct_onset_xrs"]["threshold"] + (1.0 - a) * probabilities["direct_onset_engineered"]["threshold"]
        th, diag = select_threshold(ythr, pthr)
        blend_candidates.append({"alpha": a, "threshold": th, "diag": diag})
    blend_candidates.sort(key=lambda row: (-float(row["diag"]["chosen"]["TSS"]), float(row["diag"]["chosen"]["FAR"]), -float(row["threshold"])))
    best_blend = blend_candidates[0]
    alpha = float(best_blend["alpha"])
    probabilities["threshold_selected_blend"] = {
        role: alpha * probabilities["direct_onset_xrs"][role] + (1.0 - alpha) * probabilities["direct_onset_engineered"][role]
        for role in roles
    }
    thresholds["threshold_selected_blend"] = float(best_blend["threshold"])
    selection["threshold_selected_blend"] = {"alpha": alpha, **best_blend["diag"]}

    rows: list[dict[str, Any]] = []
    yscore = roles["score"]["y_onset"].to_numpy(int)
    for name in probabilities:
        met = full_metrics(yscore, probabilities[name]["score"], thresholds[name])
        rows.append({"model": name, "role": "score", **met})
    result_table = pd.DataFrame(rows).sort_values(["TSS", "HSS"], ascending=False)
    result_table.to_csv(out / "phase2_score_metrics.csv", index=False)

    reference = result_table[result_table["model"].eq("occurrence_trained_xrs_baseline")].iloc[0]
    candidates = result_table[~result_table["model"].eq("occurrence_trained_xrs_baseline")].copy()
    candidates["passes_development_gate"] = (
        (candidates["TSS"] > float(reference["TSS"]))
        & (candidates["HSS"] >= float(reference["HSS"]))
        & (candidates["FAR"] < float(reference["FAR"]))
    )
    candidates = candidates.sort_values(["passes_development_gate", "TSS", "HSS", "FAR"], ascending=[False, False, False, True])
    champion_name = str(candidates.iloc[0]["model"])
    champion = result_table[result_table["model"].eq(champion_name)].iloc[0]

    bootstrap = unit_bootstrap_delta(
        roles["score"],
        probabilities["occurrence_trained_xrs_baseline"]["score"],
        probabilities[champion_name]["score"],
        thresholds["occurrence_trained_xrs_baseline"],
        thresholds[champion_name],
    )

    prediction_out = roles["score"][["issue_timestamp", "y_onset", "eligibility_code", "onset_episode_id", "quiet_block_id"]].copy()
    for name in probabilities:
        prediction_out[f"p__{name}"] = probabilities[name]["score"]
        prediction_out[f"alert__{name}"] = (probabilities[name]["score"] >= thresholds[name]).astype(int)
    prediction_out.to_csv(out / "phase2_score_predictions.csv", index=False)

    summary = {
        "study_id": cfg["study_id"],
        "status": "DEVELOPMENT_COMPLETE",
        "protected_post_2025_outcomes_accessed": False,
        "score_rows": int(len(roles["score"])),
        "score_positive_onsets": int(roles["score"]["y_onset"].sum()),
        "engineered_feature_count": len(engineered),
        "threshold_selection": selection,
        "calibration_intercepts": intercepts,
        "reference_model": "occurrence_trained_xrs_baseline",
        "champion_model": champion_name,
        "champion_passes_development_gate": bool(candidates.iloc[0]["passes_development_gate"]),
        "reference_metrics": {k: float(reference[k]) for k in ("TSS", "HSS", "FAR", "FPR", "POD", "Brier", "AUPRC", "AUROC")},
        "champion_metrics": {k: float(champion[k]) for k in ("TSS", "HSS", "FAR", "FPR", "POD", "Brier", "AUPRC", "AUROC")},
        "champion_minus_reference_tss": float(champion["TSS"] - reference["TSS"]),
        "bootstrap_champion_minus_reference": bootstrap,
        "claim_boundary": cfg["claim_boundary"],
        "note": "This run is historical development evidence. It cannot establish published-model or prospective superiority."
    }
    dump_json(out / "phase2_summary.json", summary)
    episodes.to_csv(out / "episode_table.csv", index=False)
    dump_json(out / "contract_snapshot.json", cfg)
    dump_json(out / "evidence_hashes.json", {
        p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "evidence_hashes.json"
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
