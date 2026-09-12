"""Independent V2 verifier for the audit-corrected episode benchmark artifact.

This verifier deliberately does not import either benchmark runner. It uses only
persisted CSV/JSON/NPZ evidence and independently recomputes:
- row alert consistency with row-specific frozen thresholds;
- attrition reconciliation;
- episode-normalized weight identities;
- full point confusion/probability metrics;
- mapped-episode point metrics;
- shared physical-unit bootstrap TSS/FAR intervals;
- mapped cross-view TSS contrasts and rank-reversal fractions;
- all listed evidence hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FULL_DEFS = {
    "WINDOW_OCCURRENCE_STANDARD": ("standard_occurrence_label", "standard_window_weight", False, False),
    "EPISODE_NORMALIZED_OCCURRENCE": ("standard_occurrence_label", "episode_normalized_occurrence_weight", False, False),
    "NEW_ONSET_CAUSAL": ("onset_label", "new_onset_weight", True, False),
    "EPISODE_NORMALIZED_ONSET": ("onset_label", "episode_normalized_onset_weight", True, False),
}
MAPPED_DEFS = {
    "MAPPED_STANDARD_OCCURRENCE": ("standard_occurrence_label", "__ones__", False, True),
    "MAPPED_EPISODE_NORMALIZED_OCCURRENCE": ("standard_occurrence_label", "episode_normalized_occurrence_weight", False, True),
    "MAPPED_NEW_ONSET_CAUSAL": ("onset_label", "new_onset_weight", True, True),
    "MAPPED_EPISODE_NORMALIZED_ONSET": ("onset_label", "episode_normalized_onset_weight", True, True),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _subset(df: pd.DataFrame, onset: bool, mapped: bool) -> pd.DataFrame:
    use = df.copy()
    if onset:
        use = use[use["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
    elif mapped:
        use = use[(use["standard_occurrence_label"] == 0) | use["occurrence_episode_id"].notna()].copy()
    return use


def _weights(use: pd.DataFrame, name: str) -> np.ndarray:
    return np.ones(len(use), dtype=float) if name == "__ones__" else use[name].astype(float).to_numpy()


def metric(use: pd.DataFrame, label_col: str, weight_col: str) -> dict[str, float]:
    y = use[label_col].astype(int).to_numpy()
    p = use["probability"].astype(float).to_numpy()
    a = use["binary_alert"].astype(int).to_numpy()
    w = _weights(use, weight_col)
    tp = float(w[(y == 1) & (a == 1)].sum())
    fn = float(w[(y == 1) & (a == 0)].sum())
    fp = float(w[(y == 0) & (a == 1)].sum())
    tn = float(w[(y == 0) & (a == 0)].sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr if np.isfinite(pod) and np.isfinite(fpr) else float("nan")
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fn * fp) / denom if denom else float("nan")
    brier = float(np.sum(w * (p - y) ** 2) / w.sum()) if w.sum() else float("nan")
    auroc = float("nan")
    auprc = float("nan")
    mask = w > 0
    if np.any(mask) and len(np.unique(y[mask])) == 2:
        try:
            auroc = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            pass
        try:
            auprc = float(average_precision_score(y, p, sample_weight=w))
        except Exception:
            pass
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "pod": pod, "fpr": fpr, "far": far, "tss": tss, "hss": hss, "brier": brier, "auroc": auroc, "auprc": auprc, "weight_sum": float(w.sum())}


def _close(a: Any, b: Any, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol, equal_nan=True))
    except Exception:
        return False


def verify_prediction_rows(pred: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if pred.duplicated(["fold", "issue_timestamp", "model"]).any():
        problems.append("duplicate prediction row")
    p = pred["probability"].astype(float).to_numpy()
    t = pred["threshold"].astype(float).to_numpy()
    a = pred["binary_alert"].astype(int).to_numpy()
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        problems.append("invalid probability")
    if np.any(~np.isfinite(t)):
        problems.append("invalid threshold")
    if not np.array_equal((p >= t).astype(int), a):
        problems.append("binary alert != probability >= row threshold")
    signatures = []
    for _, group in pred.groupby("model"):
        signatures.append(tuple(zip(group["fold"].astype(str), group["issue_timestamp"].astype(str))))
    if len(set(signatures)) != 1:
        problems.append("model cohorts differ")

    base = pred[pred.model == sorted(pred.model.unique())[0]].copy()
    occurrence = base[(base.standard_occurrence_label == 1) & base.occurrence_episode_id.notna()].groupby("occurrence_episode_id")["episode_normalized_occurrence_weight"].sum()
    if len(occurrence) and not np.allclose(occurrence, 1.0, atol=1e-12, rtol=0):
        problems.append("occurrence episode weights do not sum to one")
    onset = base[base.eligibility_code == "ELIGIBLE_ONSET_POSITIVE"].groupby("onset_episode_id")["episode_normalized_onset_weight"].sum()
    if len(onset) and not np.allclose(onset, 1.0, atol=1e-12, rtol=0):
        problems.append("onset episode weights do not sum to one")
    if np.any(base.loc[base.eligibility_code == "ALREADY_ACTIVE_PERSISTENCE", "new_onset_weight"].astype(float) != 0):
        problems.append("persistence carries onset weight")
    return problems


def compare_result_table(pred: pd.DataFrame, reported: pd.DataFrame, definitions: dict[str, tuple[str, str, bool, bool]]) -> list[str]:
    problems: list[str] = []
    numeric = ("tp", "fn", "fp", "tn", "pod", "fpr", "far", "tss", "hss", "brier", "auroc", "auprc", "weight_sum")
    expected_rows = len(pred.model.unique()) * len(definitions)
    if len(reported) != expected_rows:
        problems.append(f"reported result row count {len(reported)} != {expected_rows}")
    for _, row in reported.iterrows():
        model = str(row["model"])
        view = str(row["view"])
        if view not in definitions:
            problems.append(f"unknown view {view}")
            continue
        label_col, weight_col, onset, mapped = definitions[view]
        use = _subset(pred[pred.model == model].copy(), onset, mapped)
        calc = metric(use, label_col, weight_col)
        for key in numeric:
            if key in row.index and not _close(row[key], calc[key]):
                problems.append(f"{model}/{view}/{key}: {row[key]} != {calc[key]}")
        if int(row["rows"]) != len(use):
            problems.append(f"{model}/{view}/rows mismatch")
        if int(row["positive_rows"]) != int((use[label_col].astype(int) == 1).sum()):
            problems.append(f"{model}/{view}/positive_rows mismatch")
        thresholds = use.threshold.astype(float).to_numpy()
        if int(row["threshold_count"]) != len(np.unique(thresholds)):
            problems.append(f"{model}/{view}/threshold_count mismatch")
        if not _close(row["threshold_min"], thresholds.min()) or not _close(row["threshold_max"], thresholds.max()):
            problems.append(f"{model}/{view}/threshold range mismatch")
    return problems


def load_draws(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "positive_indices": np.asarray(data["positive_indices"], dtype=int),
        "negative_indices": np.asarray(data["negative_indices"], dtype=int),
        "positive_units": tuple(map(str, data["positive_units"].tolist())),
        "negative_units": tuple(map(str, data["negative_units"].tolist())),
    }


def draw_multiplicity(indices: np.ndarray, unit_count: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], unit_count), dtype=float)
    for idx in range(indices.shape[0]):
        out[idx] = np.bincount(indices[idx], minlength=unit_count)
    return out


def unit_contributions(units: list[str], y: np.ndarray, alerts: np.ndarray, weights: np.ndarray, order: tuple[str, ...]) -> np.ndarray:
    pos = {unit: idx for idx, unit in enumerate(order)}
    out = np.zeros((len(order), 4), dtype=float)  # tp, fn, fp, tn
    for unit, label, alert, weight in zip(units, y, alerts, weights, strict=True):
        if unit not in pos:
            continue
        idx = pos[unit]
        if label == 1 and alert == 1:
            out[idx, 0] += weight
        elif label == 1:
            out[idx, 1] += weight
        elif alert == 1:
            out[idx, 2] += weight
        else:
            out[idx, 3] += weight
    return out


def bootstrap_arrays(pred: pd.DataFrame, draws: dict[str, Any], model: str, view: str) -> dict[str, np.ndarray]:
    base = pred[pred.model == model].copy()
    if view in {"MAPPED_STANDARD_OCCURRENCE", "MAPPED_EPISODE_NORMALIZED_OCCURRENCE"}:
        use = _subset(base, False, True)
        y = use.standard_occurrence_label.astype(int).to_numpy()
        weights = np.ones(len(use), dtype=float) if view == "MAPPED_STANDARD_OCCURRENCE" else use.episode_normalized_occurrence_weight.astype(float).to_numpy()
        units = [f"EPISODE::{ep}" if int(label) == 1 else str(q) for ep, q, label in zip(use.occurrence_episode_id, use.quiet_block_id, y, strict=True)]
    else:
        use = _subset(base, True, True)
        y = use.onset_label.astype(int).to_numpy()
        weights = use.new_onset_weight.astype(float).to_numpy() if view == "MAPPED_NEW_ONSET_CAUSAL" else use.episode_normalized_onset_weight.astype(float).to_numpy()
        units = [f"EPISODE::{ep}" if int(label) == 1 else str(q) for ep, q, label in zip(use.onset_episode_id, use.quiet_block_id, y, strict=True)]
    alerts = use.binary_alert.astype(int).to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_contrib = unit_contributions([units[i] for i in pos_idx], y[pos_idx], alerts[pos_idx], weights[pos_idx], draws["positive_units"])
    neg_contrib = unit_contributions([units[i] for i in neg_idx], y[neg_idx], alerts[neg_idx], weights[neg_idx], draws["negative_units"])
    pm = draw_multiplicity(draws["positive_indices"], len(draws["positive_units"]))
    nm = draw_multiplicity(draws["negative_indices"], len(draws["negative_units"]))
    counts = pm @ pos_contrib + nm @ neg_contrib
    tp, fn, fp, tn = counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = tp / (tp + fn)
        fpr = fp / (fp + tn)
        far = fp / (tp + fp)
        tss = pod - fpr
    return {"tss": tss, "far": far}


def interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def verify_bootstrap(pred: pd.DataFrame, draw_path: Path, interval_path: Path, contrast_path: Path, rank_path: Path) -> list[str]:
    problems: list[str] = []
    draws = load_draws(draw_path)
    pi, ni = draws["positive_indices"], draws["negative_indices"]
    if pi.ndim != 2 or ni.ndim != 2 or pi.shape[0] != ni.shape[0]:
        problems.append("bootstrap dimensions invalid")
        return problems
    if pi.shape[1] != len(draws["positive_units"]) or ni.shape[1] != len(draws["negative_units"]):
        problems.append("bootstrap width mismatch")
    if len(draws["positive_units"]) and (pi.min() < 0 or pi.max() >= len(draws["positive_units"])):
        problems.append("positive draw index out of range")
    if len(draws["negative_units"]) and (ni.min() < 0 or ni.max() >= len(draws["negative_units"])):
        problems.append("negative draw index out of range")

    reported_intervals = json.loads(interval_path.read_text())
    reported_contrasts = json.loads(contrast_path.read_text())
    reported_rank = json.loads(rank_path.read_text())
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for model in sorted(pred.model.unique()):
        for view in MAPPED_DEFS:
            arr = bootstrap_arrays(pred, draws, model, view)
            arrays[(model, view)] = arr["tss"]
            rep = reported_intervals["models"][model][view]
            if not np.allclose(rep["tss_interval_95"], interval(arr["tss"]), atol=1e-10, rtol=1e-9, equal_nan=True):
                problems.append(f"{model}/{view} TSS interval mismatch")
            if not np.allclose(rep["far_interval_95"], interval(arr["far"]), atol=1e-10, rtol=1e-9, equal_nan=True):
                problems.append(f"{model}/{view} FAR interval mismatch")

    for model in sorted(pred.model.unique()):
        standard = arrays[(model, "MAPPED_STANDARD_OCCURRENCE")]
        normalized = arrays[(model, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")]
        onset = arrays[(model, "MAPPED_NEW_ONSET_CAUSAL")]
        onset_norm = arrays[(model, "MAPPED_EPISODE_NORMALIZED_ONSET")]
        for name, delta in {
            "multiplicity_only_tss_shift": normalized - standard,
            "persistence_exclusion_tss_shift": onset - normalized,
            "onset_episode_weighting_tss_shift": onset_norm - onset,
        }.items():
            rep = reported_contrasts[f"{model}::{name}"]
            if not np.allclose(rep["interval_95"], interval(delta), atol=1e-10, rtol=1e-9, equal_nan=True):
                problems.append(f"{model}/{name} interval mismatch")
            if not _close(rep["median"], np.nanmedian(delta)):
                problems.append(f"{model}/{name} median mismatch")

    models = [m for m in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only") if (m, "MAPPED_STANDARD_OCCURRENCE") in arrays]
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            standard_delta = arrays[(a, "MAPPED_STANDARD_OCCURRENCE")] - arrays[(b, "MAPPED_STANDARD_OCCURRENCE")]
            normalized_delta = arrays[(a, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")] - arrays[(b, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")]
            onset_delta = arrays[(a, "MAPPED_NEW_ONSET_CAUSAL")] - arrays[(b, "MAPPED_NEW_ONSET_CAUSAL")]
            valid1 = np.isfinite(standard_delta) & np.isfinite(normalized_delta)
            valid2 = np.isfinite(standard_delta) & np.isfinite(onset_delta)
            rep = reported_rank[f"{a}__vs__{b}"]
            r1 = float(np.mean((standard_delta[valid1] * normalized_delta[valid1]) < 0)) if np.any(valid1) else float("nan")
            r2 = float(np.mean((standard_delta[valid2] * onset_delta[valid2]) < 0)) if np.any(valid2) else float("nan")
            if not _close(rep["standard_vs_normalized_occurrence_reversal_fraction"], r1):
                problems.append(f"{a}/{b} normalized reversal mismatch")
            if not _close(rep["standard_vs_onset_reversal_fraction"], r2):
                problems.append(f"{a}/{b} onset reversal mismatch")
            if not np.allclose(rep["onset_tss_difference_interval_95"], interval(onset_delta), atol=1e-10, rtol=1e-9, equal_nan=True):
                problems.append(f"{a}/{b} onset difference interval mismatch")
    return problems


def verify(root: Path) -> dict[str, Any]:
    root = Path(root)
    checks: dict[str, Any] = {}

    manifest = json.loads((root / "evidence_hashes.json").read_text())
    bad_hashes = []
    for rel, expected in manifest.items():
        path = root / rel
        if not path.exists() or sha256_file(path) != expected:
            bad_hashes.append(rel)
    checks["hashes"] = {"passed": not bad_hashes, "listed_files": len(manifest), "mismatches": bad_hashes}

    ledger = pd.read_csv(root / "candidate_attrition_ledger.csv")
    attrition = json.loads((root / "attrition_summary.json").read_text())
    actual_codes = dict(sorted(Counter(ledger.eligibility_code.astype(str)).items()))
    attrition_problems = []
    if len(ledger) != int(attrition["total_candidate_issues"]):
        attrition_problems.append("candidate total mismatch")
    if actual_codes != {str(k): int(v) for k, v in attrition["terminal_codes"].items()}:
        attrition_problems.append("terminal-code mismatch")
    checks["attrition"] = {"passed": not attrition_problems, "rows": len(ledger), "terminal_codes": actual_codes, "problems": attrition_problems}

    for tag in ("strict_2017", "expanding_oof_2014_2017"):
        pred = pd.read_csv(root / f"predictions_{tag}.csv")
        full = pd.read_csv(root / f"results_{tag}.csv")
        mapped = pd.read_csv(root / f"results_{tag}_mapped_episode_sensitivity.csv")
        prediction_problems = verify_prediction_rows(pred)
        full_problems = compare_result_table(pred, full, FULL_DEFS)
        mapped_problems = compare_result_table(pred, mapped, MAPPED_DEFS)
        prefix = "strict" if tag == "strict_2017" else "oof"
        draw_file = root / ("strict_2017_shared_bootstrap_draws.npz" if tag == "strict_2017" else "expanding_oof_shared_bootstrap_draws.npz")
        bootstrap_problems = verify_bootstrap(
            pred,
            draw_file,
            root / f"{prefix}_bootstrap_intervals.json",
            root / f"{prefix}_contrasts.json",
            root / f"{prefix}_rank_reversal_bootstrap.json",
        )
        checks[f"{tag}_predictions"] = {"passed": not prediction_problems, "rows": len(pred), "problems": prediction_problems}
        checks[f"{tag}_full_points"] = {"passed": not full_problems, "reported_rows": len(full), "problems": full_problems}
        checks[f"{tag}_mapped_points"] = {"passed": not mapped_problems, "reported_rows": len(mapped), "problems": mapped_problems}
        checks[f"{tag}_bootstrap"] = {"passed": not bootstrap_problems, "problems": bootstrap_problems}

    passed = all(item["passed"] for item in checks.values())
    return {"format": "IRIS_EPISODE_DEVELOPMENT_ARTIFACT_INDEPENDENT_VERIFICATION_V2", "passed": passed, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.artifact_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
