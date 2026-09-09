"""Post-result audit-corrected execution wrapper for episode benchmark V1.

This wrapper DOES NOT change sources, features, fitted models, fold definitions,
threshold selection, probabilities, or row-level decisions. It first executes
the frozen benchmark through the runtime compatibility wrapper, then corrects
only evaluation/evidence-accounting defects documented in
EPISODE_BENCHMARK_POST_RESULT_AUDIT_CORRECTION_2026-09-09.md:

1. OOF confusion metrics use each persisted row's frozen binary alert instead
   of applying one fold's threshold globally.
2. Physical-unit bootstrap inference is reported on an explicit uniquely
   mapped-episode sensitivity cohort; full-window point estimates remain
   separate and visible.
3. Evidence hashes are generated only after final result/summary files exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import tools.run_episode_normalized_development_benchmark_v1 as benchmark
import tools.run_episode_normalized_development_benchmark_v1_compat as compat
from tools.episode_benchmark_statistics import (
    SharedBootstrapDraws,
    bootstrap_metric_arrays,
    percentile_interval,
    unit_confusion_contributions,
)
from tools.episode_normalized_benchmark import rank_stability

CORRECTION_DOC = "architecture/EPISODE_BENCHMARK_POST_RESULT_AUDIT_CORRECTION_2026-09-09.md"
MAPPED_VIEWS = (
    "MAPPED_STANDARD_OCCURRENCE",
    "MAPPED_EPISODE_NORMALIZED_OCCURRENCE",
    "MAPPED_NEW_ONSET_CAUSAL",
    "MAPPED_EPISODE_NORMALIZED_ONSET",
)
FULL_VIEWS = (
    "WINDOW_OCCURRENCE_STANDARD",
    "EPISODE_NORMALIZED_OCCURRENCE",
    "NEW_ONSET_CAUSAL",
    "EPISODE_NORMALIZED_ONSET",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n", encoding="utf-8")


def _metric_row(use: pd.DataFrame, label_col: str, weight_col: str) -> dict[str, Any]:
    if use.empty:
        raise RuntimeError("cannot score empty evaluation view")
    y = use[label_col].astype(int).to_numpy()
    p = use["probability"].astype(float).to_numpy()
    alerts = use["binary_alert"].astype(int).to_numpy()
    w = use[weight_col].astype(float).to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isin(alerts, [0, 1]).all():
        raise RuntimeError("non-binary label or alert")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise RuntimeError("invalid persisted probability")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise RuntimeError("invalid evaluation weight")
    expected_alert = (p >= use["threshold"].astype(float).to_numpy()).astype(int)
    if not np.array_equal(expected_alert, alerts):
        raise RuntimeError("persisted binary alert disagrees with row-specific frozen threshold")

    tp = float(w[(y == 1) & (alerts == 1)].sum())
    fn = float(w[(y == 1) & (alerts == 0)].sum())
    fp = float(w[(y == 0) & (alerts == 1)].sum())
    tn = float(w[(y == 0) & (alerts == 0)].sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr if np.isfinite(pod) and np.isfinite(fpr) else float("nan")
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fn * fp) / denom if denom else float("nan")
    brier = float(np.sum(w * (p - y) ** 2) / w.sum()) if w.sum() else float("nan")

    mask = w > 0
    auroc = float("nan")
    auprc = float("nan")
    if np.any(mask) and len(np.unique(y[mask])) == 2:
        try:
            auroc = float(roc_auc_score(y, p, sample_weight=w))
        except Exception:
            pass
        try:
            auprc = float(average_precision_score(y, p, sample_weight=w))
        except Exception:
            pass

    threshold_values = use["threshold"].astype(float).to_numpy()
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "pod": pod,
        "fpr": fpr,
        "far": far,
        "tss": tss,
        "hss": hss,
        "brier": brier,
        "weight_sum": float(w.sum()),
        "auroc": auroc,
        "auprc": auprc,
        "rows": int(len(use)),
        "positive_rows": int(np.sum(y == 1)),
        "threshold_policy": "ROW_SPECIFIC_FROZEN",
        "threshold_count": int(len(np.unique(threshold_values))),
        "threshold_min": float(np.min(threshold_values)),
        "threshold_max": float(np.max(threshold_values)),
    }


def _onset_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()


def corrected_full_point_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(pred["model"].unique()):
        base = pred[pred["model"] == model].copy()
        onset = _onset_rows(base)
        definitions = (
            ("WINDOW_OCCURRENCE_STANDARD", base, "standard_occurrence_label", "standard_window_weight"),
            ("EPISODE_NORMALIZED_OCCURRENCE", base, "standard_occurrence_label", "episode_normalized_occurrence_weight"),
            ("NEW_ONSET_CAUSAL", onset, "onset_label", "new_onset_weight"),
            ("EPISODE_NORMALIZED_ONSET", onset, "onset_label", "episode_normalized_onset_weight"),
        )
        for view, use, label_col, weight_col in definitions:
            rows.append({"model": model, "view": view, **_metric_row(use, label_col, weight_col)})
    return pd.DataFrame(rows)


def mapped_point_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(pred["model"].unique()):
        base = pred[pred["model"] == model].copy()
        mapped_occ = base[(base["standard_occurrence_label"] == 0) | base["occurrence_episode_id"].notna()].copy()
        mapped_occ["mapped_standard_weight"] = 1.0
        onset = _onset_rows(base)
        definitions = (
            ("MAPPED_STANDARD_OCCURRENCE", mapped_occ, "standard_occurrence_label", "mapped_standard_weight"),
            ("MAPPED_EPISODE_NORMALIZED_OCCURRENCE", mapped_occ, "standard_occurrence_label", "episode_normalized_occurrence_weight"),
            ("MAPPED_NEW_ONSET_CAUSAL", onset, "onset_label", "new_onset_weight"),
            ("MAPPED_EPISODE_NORMALIZED_ONSET", onset, "onset_label", "episode_normalized_onset_weight"),
        )
        for view, use, label_col, weight_col in definitions:
            rows.append({"model": model, "view": view, **_metric_row(use, label_col, weight_col)})
    return pd.DataFrame(rows)


def _rank_report(point: pd.DataFrame, standard_view: str, normalized_view: str, onset_view: str) -> dict[str, Any]:
    models = [m for m in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only") if m in set(point["model"])]
    def values(view: str) -> dict[str, float]:
        return {m: float(point[(point.model == m) & (point.view == view)].iloc[0].tss) for m in models}
    return {
        "standard_to_normalized_occurrence_rank": rank_stability(values(standard_view), values(normalized_view)),
        "standard_to_new_onset_rank": rank_stability(values(standard_view), values(onset_view)),
    }


def _load_draws(path: Path) -> SharedBootstrapDraws:
    data = np.load(path, allow_pickle=False)
    draws = SharedBootstrapDraws(
        tuple(map(str, data["positive_units"].tolist())),
        tuple(map(str, data["negative_units"].tolist())),
        np.asarray(data["positive_indices"], dtype=np.int32),
        np.asarray(data["negative_indices"], dtype=np.int32),
    )
    draws.validate()
    return draws


def _mapped_view_rows(base: pd.DataFrame, view: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if view in {"MAPPED_STANDARD_OCCURRENCE", "MAPPED_EPISODE_NORMALIZED_OCCURRENCE"}:
        use = base[(base["standard_occurrence_label"] == 0) | base["occurrence_episode_id"].notna()].copy()
        y = use["standard_occurrence_label"].astype(int).to_numpy()
        weights = np.ones(len(use), dtype=float) if view == "MAPPED_STANDARD_OCCURRENCE" else use["episode_normalized_occurrence_weight"].astype(float).to_numpy()
        units = [f"EPISODE::{ep}" if int(label) == 1 else str(q) for ep, q, label in zip(use["occurrence_episode_id"], use["quiet_block_id"], y, strict=True)]
    else:
        use = _onset_rows(base)
        y = use["onset_label"].astype(int).to_numpy()
        weights = use["new_onset_weight"].astype(float).to_numpy() if view == "MAPPED_NEW_ONSET_CAUSAL" else use["episode_normalized_onset_weight"].astype(float).to_numpy()
        units = [f"EPISODE::{ep}" if int(label) == 1 else str(q) for ep, q, label in zip(use["onset_episode_id"], use["quiet_block_id"], y, strict=True)]
    alerts = use["binary_alert"].astype(int).to_numpy()
    probabilities = use["probability"].astype(float).to_numpy()
    if not np.array_equal(alerts, (probabilities >= use["threshold"].astype(float).to_numpy()).astype(int)):
        raise RuntimeError("row-specific alert mismatch during bootstrap correction")
    return use, y, alerts, weights, units


def corrected_bootstrap(pred: pd.DataFrame, draw_path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    draws = _load_draws(draw_path)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    report: dict[str, Any] = {
        "estimand": "UNIQUELY_MAPPED_PHYSICAL_EPISODE_SENSITIVITY",
        "replicates": int(draws.positive_indices.shape[0]),
        "positive_episode_units": len(draws.positive_units),
        "negative_quiet_block_units": len(draws.negative_units),
        "models": {},
    }
    for model in sorted(pred["model"].unique()):
        base = pred[pred["model"] == model].copy()
        report["models"][model] = {}
        for view in MAPPED_VIEWS:
            _, y, alerts, weights, units = _mapped_view_rows(base, view)
            pos_rows = np.where(y == 1)[0]
            neg_rows = np.where(y == 0)[0]
            pos_contrib = unit_confusion_contributions(
                [units[i] for i in pos_rows], y[pos_rows], alerts[pos_rows], weights[pos_rows], draws.positive_units
            )
            neg_contrib = unit_confusion_contributions(
                [units[i] for i in neg_rows], y[neg_rows], alerts[neg_rows], weights[neg_rows], draws.negative_units
            )
            metrics = bootstrap_metric_arrays(draws, pos_contrib, neg_contrib)
            arrays[(model, view)] = metrics["tss"]
            tlo, thi = percentile_interval(metrics["tss"])
            flo, fhi = percentile_interval(metrics["far"])
            report["models"][model][view] = {
                "tss_interval_95": [tlo, thi],
                "far_interval_95": [flo, fhi],
                "finite_tss_replicates": int(np.isfinite(metrics["tss"]).sum()),
                "finite_far_replicates": int(np.isfinite(metrics["far"]).sum()),
            }
    return report, arrays


def corrected_contrasts(arrays: Mapping[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"estimand": "UNIQUELY_MAPPED_PHYSICAL_EPISODE_SENSITIVITY"}
    models = sorted({model for model, _ in arrays})
    for model in models:
        standard = arrays[(model, "MAPPED_STANDARD_OCCURRENCE")]
        normalized = arrays[(model, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")]
        onset = arrays[(model, "MAPPED_NEW_ONSET_CAUSAL")]
        onset_norm = arrays[(model, "MAPPED_EPISODE_NORMALIZED_ONSET")]
        for name, delta in {
            "multiplicity_only_tss_shift": normalized - standard,
            "persistence_exclusion_tss_shift": onset - normalized,
            "onset_episode_weighting_tss_shift": onset_norm - onset,
        }.items():
            lo, hi = percentile_interval(delta)
            finite = np.isfinite(delta)
            result[f"{model}::{name}"] = {
                "interval_95": [lo, hi],
                "median": float(np.nanmedian(delta)),
                "probability_positive": float(np.mean(delta[finite] > 0)) if np.any(finite) else float("nan"),
                "probability_negative": float(np.mean(delta[finite] < 0)) if np.any(finite) else float("nan"),
            }
    if ("xgb_joint", "MAPPED_NEW_ONSET_CAUSAL") in arrays and ("xgb_xrs_only", "MAPPED_NEW_ONSET_CAUSAL") in arrays:
        delta = arrays[("xgb_joint", "MAPPED_NEW_ONSET_CAUSAL")] - arrays[("xgb_xrs_only", "MAPPED_NEW_ONSET_CAUSAL")]
        lo, hi = percentile_interval(delta)
        result["proton_aware_minus_blind_onset_tss"] = {"interval_95": [lo, hi], "median": float(np.nanmedian(delta))}
    return result


def corrected_rank_reversal(arrays: Mapping[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    models = [m for m in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only") if (m, "MAPPED_STANDARD_OCCURRENCE") in arrays]
    out: dict[str, Any] = {"estimand": "UNIQUELY_MAPPED_PHYSICAL_EPISODE_SENSITIVITY"}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            standard_delta = arrays[(a, "MAPPED_STANDARD_OCCURRENCE")] - arrays[(b, "MAPPED_STANDARD_OCCURRENCE")]
            normalized_delta = arrays[(a, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")] - arrays[(b, "MAPPED_EPISODE_NORMALIZED_OCCURRENCE")]
            onset_delta = arrays[(a, "MAPPED_NEW_ONSET_CAUSAL")] - arrays[(b, "MAPPED_NEW_ONSET_CAUSAL")]
            valid_norm = np.isfinite(standard_delta) & np.isfinite(normalized_delta)
            valid_onset = np.isfinite(standard_delta) & np.isfinite(onset_delta)
            onset_lo, onset_hi = percentile_interval(onset_delta)
            out[f"{a}__vs__{b}"] = {
                "standard_vs_normalized_occurrence_reversal_fraction": float(np.mean((standard_delta[valid_norm] * normalized_delta[valid_norm]) < 0)) if np.any(valid_norm) else float("nan"),
                "standard_vs_onset_reversal_fraction": float(np.mean((standard_delta[valid_onset] * onset_delta[valid_onset]) < 0)) if np.any(valid_onset) else float("nan"),
                "onset_tss_difference_interval_95": [onset_lo, onset_hi],
                "onset_tss_difference_median": float(np.nanmedian(onset_delta)),
            }
    return out


def _preserve(path: Path) -> None:
    if path.exists():
        target = path.with_name("legacy_uncorrected_" + path.name)
        if not target.exists():
            shutil.copy2(path, target)


def _rewrite_figures(out: Path, oof_full: pd.DataFrame, oof_mapped: pd.DataFrame) -> None:
    # Import lazily so this post-processing remains testable without a display.
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    models = [m for m in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only", "current_proton_active_diagnostic") if m in set(oof_full.model)]
    views = list(FULL_VIEWS)
    labels = ["Standard", "Episode-normalized\noccurrence", "New-onset", "Episode-normalized\nonset"]
    x = np.arange(len(views), dtype=float)
    width = 0.8 / max(1, len(models))
    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        vals = [float(oof_full[(oof_full.model == model) & (oof_full.view == view)].iloc[0].tss) for view in views]
        plt.bar(x - 0.4 + width / 2 + idx * width, vals, width=width, label=model)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("TSS using fold-specific frozen alerts")
    plt.title("Evaluation definition changes measured skill (corrected development OOF)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "01_tss_by_evaluation_view_CORRECTED.png", dpi=180)
    plt.close()

    subset = oof_mapped[oof_mapped.model.isin(["xgb_joint", "xgb_xrs_only", "elastic_net_joint"])].copy()
    pivot = subset.pivot(index="model", columns="view", values="tss")
    plt.figure(figsize=(8, 5))
    for model in pivot.index:
        plt.plot([0, 1], [pivot.loc[model, "MAPPED_STANDARD_OCCURRENCE"], pivot.loc[model, "MAPPED_NEW_ONSET_CAUSAL"]], marker="o", label=model)
    plt.xticks([0, 1], ["Mapped standard occurrence", "Mapped new-onset causal"])
    plt.ylabel("TSS")
    plt.title("Model ranking sensitivity to the forecasting question")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "05_mapped_standard_to_onset_rank.png", dpi=180)
    plt.close()


def postprocess(out: Path) -> dict[str, Any]:
    strict_pred = pd.read_csv(out / "predictions_strict_2017.csv")
    oof_pred = pd.read_csv(out / "predictions_expanding_oof_2014_2017.csv")

    for name in (
        "results_strict_2017.csv",
        "results_expanding_oof_2014_2017.csv",
        "strict_bootstrap_intervals.json",
        "oof_bootstrap_intervals.json",
        "strict_contrasts.json",
        "oof_contrasts.json",
        "strict_rank_reversal_bootstrap.json",
        "oof_rank_reversal_bootstrap.json",
        "summary.json",
        "evidence_hashes.json",
    ):
        _preserve(out / name)

    strict_full = corrected_full_point_table(strict_pred)
    oof_full = corrected_full_point_table(oof_pred)
    strict_mapped = mapped_point_table(strict_pred)
    oof_mapped = mapped_point_table(oof_pred)
    strict_full.to_csv(out / "results_strict_2017.csv", index=False)
    oof_full.to_csv(out / "results_expanding_oof_2014_2017.csv", index=False)
    strict_mapped.to_csv(out / "results_strict_2017_mapped_episode_sensitivity.csv", index=False)
    oof_mapped.to_csv(out / "results_expanding_oof_2014_2017_mapped_episode_sensitivity.csv", index=False)

    strict_boot, strict_arrays = corrected_bootstrap(strict_pred, out / "strict_2017_shared_bootstrap_draws.npz")
    oof_boot, oof_arrays = corrected_bootstrap(oof_pred, out / "expanding_oof_shared_bootstrap_draws.npz")
    strict_contrasts = corrected_contrasts(strict_arrays)
    oof_contrasts = corrected_contrasts(oof_arrays)
    strict_reversals = corrected_rank_reversal(strict_arrays)
    oof_reversals = corrected_rank_reversal(oof_arrays)
    dump_json(out / "strict_bootstrap_intervals.json", strict_boot)
    dump_json(out / "oof_bootstrap_intervals.json", oof_boot)
    dump_json(out / "strict_contrasts.json", strict_contrasts)
    dump_json(out / "oof_contrasts.json", oof_contrasts)
    dump_json(out / "strict_rank_reversal_bootstrap.json", strict_reversals)
    dump_json(out / "oof_rank_reversal_bootstrap.json", oof_reversals)

    old_summary = json.loads((out / "legacy_uncorrected_summary.json").read_text(encoding="utf-8"))
    summary = {
        "format": "IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_DEVELOPMENT_RESULT_V1_AUDIT_CORRECTED",
        "claim_boundary": old_summary.get("claim_boundary", {}),
        "contracts": old_summary.get("contracts", {}),
        "audit_correction": {
            "status": "POST_RESULT_ESTIMATOR_AND_EVIDENCE_CORRECTION_ONLY",
            "model_refit_due_to_audit": False,
            "threshold_reselection_due_to_audit": False,
            "cohort_or_feature_change_due_to_audit": False,
            "full_point_confusion_rule": "persisted row-specific binary_alert",
            "bootstrap_estimand": "uniquely mapped physical-episode sensitivity",
            "ambiguous_positive_windows_in_full_point_only": True,
            "correction_document": CORRECTION_DOC,
            "supersedes_old_oof_summary_for_inference": True,
        },
        "attrition": old_summary.get("attrition", {}),
        "strict_2017": {
            **old_summary.get("strict_2017", {}),
            "full_point_rank_stability": _rank_report(strict_full, "WINDOW_OCCURRENCE_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET_CAUSAL"),
            "mapped_point_rank_stability": _rank_report(strict_mapped, "MAPPED_STANDARD_OCCURRENCE", "MAPPED_EPISODE_NORMALIZED_OCCURRENCE", "MAPPED_NEW_ONSET_CAUSAL"),
            "mapped_bootstrap_contrasts": strict_contrasts,
            "mapped_rank_reversal_bootstrap": strict_reversals,
        },
        "expanding_oof_2014_2017": {
            **old_summary.get("expanding_oof_2014_2017", {}),
            "full_point_rank_stability": _rank_report(oof_full, "WINDOW_OCCURRENCE_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET_CAUSAL"),
            "mapped_point_rank_stability": _rank_report(oof_mapped, "MAPPED_STANDARD_OCCURRENCE", "MAPPED_EPISODE_NORMALIZED_OCCURRENCE", "MAPPED_NEW_ONSET_CAUSAL"),
            "mapped_bootstrap_contrasts": oof_contrasts,
            "mapped_rank_reversal_bootstrap": oof_reversals,
        },
        "authoritative_files": {
            "strict_full_point": "results_strict_2017.csv",
            "oof_full_point": "results_expanding_oof_2014_2017.csv",
            "strict_mapped_sensitivity": "results_strict_2017_mapped_episode_sensitivity.csv",
            "oof_mapped_sensitivity": "results_expanding_oof_2014_2017_mapped_episode_sensitivity.csv",
            "strict_bootstrap": "strict_bootstrap_intervals.json",
            "oof_bootstrap": "oof_bootstrap_intervals.json",
        },
        "legacy_checkpoint_files_preserved": True,
    }
    dump_json(out / "summary.json", summary)
    _rewrite_figures(out, oof_full, oof_mapped)

    # Finalize hashes LAST. Do not rewrite any hashed file afterward.
    hash_manifest: dict[str, str] = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "evidence_hashes.json":
            continue
        hash_manifest[str(path.relative_to(out))] = sha256_file(path)
    dump_json(out / "evidence_hashes.json", hash_manifest)

    print(json.dumps({
        "audit_corrected": True,
        "oof_full_standard_tss": {
            model: float(oof_full[(oof_full.model == model) & (oof_full.view == "WINDOW_OCCURRENCE_STANDARD")].iloc[0].tss)
            for model in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only", "current_proton_active_diagnostic")
        },
        "oof_full_onset_tss": {
            model: float(oof_full[(oof_full.model == model) & (oof_full.view == "NEW_ONSET_CAUSAL")].iloc[0].tss)
            for model in ("xgb_joint", "elastic_net_joint", "xgb_xrs_only", "current_proton_active_diagnostic")
        },
        "oof_mapped_rank_reversal": oof_reversals,
    }, indent=2, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compat.install_compatibility_patch()
    benchmark.run(args.output)
    postprocess(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
