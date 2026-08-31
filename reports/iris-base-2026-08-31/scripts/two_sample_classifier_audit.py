#!/usr/bin/env python3
"""Group-held-out two-sample classifier audit for real versus BASE fields.

This is an independent descriptive diagnostic.  It does not replace the
predeclared generic/physics gates and it never uses validation/test flare
outcomes.  The grouping unit is the connected physical-region identifier; the
same identifier is shared by the real and generated views when available so a
classifier cannot win by memorising a region that appears in only one split.
"""

from __future__ import annotations

import json
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
REPORT_DIR = Path(os.environ.get("IRIS_REPORT_DIR", Path(__file__).resolve().parents[1]))
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = REPORT_DIR / "artifacts"


PROXY_COLUMNS = [
    "unsigned_flux_proxy",
    "signed_flux_imbalance",
    "rms_field",
    "field_energy_proxy",
    "gradient_rms",
    "high_gradient_fraction",
    "positive_components",
    "negative_components",
    "strong_component_count",
    "strong_centroid_separation",
    "spectral_centroid",
    "spectral_high_fraction",
    "spectral_potential_proxy",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    print("loading two-sample inputs", flush=True)
    real_df = pd.read_csv(TABLE_DIR / "physics_proxy_real.csv")
    synthetic_df = pd.read_csv(TABLE_DIR / "physics_proxy_synthetic.csv")
    real_cache = RUN_ROOT / "work" / "prepared" / "positive_train_pergroup4" / "raw.npy"
    real_meta = RUN_ROOT / "work" / "prepared" / "positive_train_pergroup4" / "metadata.json"
    real_subset = RUN_ROOT / "work" / "runs" / "physics_screening_2026" / "hj" / "training_subset.csv.gz"
    if not real_cache.is_file() or not real_meta.is_file() or not real_subset.is_file():
        raise FileNotFoundError("prepared positive cache and training subset are required")
    raw = np.asarray(np.load(real_cache, mmap_mode="r"), dtype=np.float32)
    metadata = json.loads(real_meta.read_text())
    subset = pd.read_csv(real_subset)
    if list(metadata["sample_ids"]) != subset["sample_id"].astype(str).tolist():
        raise RuntimeError("prepared cache ordering does not match the saved training subset")
    if len(raw) != len(real_df) or len(raw) != len(subset):
        raise RuntimeError((raw.shape, len(real_df), len(subset)))
    synthetic = np.stack([np.load(Path(path), allow_pickle=False) for path in synthetic_df["array_path"]])
    if synthetic.ndim == 3:
        synthetic = synthetic[:, None]
    if raw.ndim == 3:
        raw = raw[:, None]
    if raw.shape[1:] != synthetic.shape[1:]:
        raise RuntimeError((raw.shape, synthetic.shape))
    real_df = real_df.copy()
    real_df["region_group_id"] = subset["region_group_id"].astype(str).values
    synthetic_df = synthetic_df.copy()
    synthetic_df["region_group_id"] = synthetic_df["source_region_group_id"].astype(str)
    print(f"loaded real={len(real_df)} synthetic={len(synthetic_df)}", flush=True)
    return real_df, synthetic_df, raw, synthetic


def block_mean(fields: np.ndarray, block: int = 8) -> np.ndarray:
    n, c, h, w = fields.shape
    if c != 1 or h % block or w % block:
        raise ValueError(fields.shape)
    return fields.reshape(n, c, h // block, block, w // block, block).mean(axis=(3, 5)).reshape(n, -1)


def run_classifier(
    name: str,
    features: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict:
    if features.shape[0] != labels.shape[0] or len(groups) != len(labels):
        raise ValueError((features.shape, labels.shape, groups.shape))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_rows = []
    coefficient_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels, groups), 1):
        if len(np.unique(labels[test_idx])) < 2:
            raise RuntimeError(f"fold {fold} has only one class")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
            ]
        )
        model.fit(features[train_idx], labels[train_idx])
        probability = model.predict_proba(features[test_idx])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        fold_rows.append(
            {
                "model": name,
                "fold": fold,
                "test_rows": int(len(test_idx)),
                "test_groups": int(len(np.unique(groups[test_idx]))),
                "auc": float(roc_auc_score(labels[test_idx], probability)),
                "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], prediction)),
            }
        )
        coefficient_rows.append(np.abs(model.named_steps["logistic"].coef_[0]))
    folds = pd.DataFrame(fold_rows)
    coefficients = np.mean(np.vstack(coefficient_rows), axis=0)
    ranked = sorted(
        (
            {"model": name, "feature": feature_names[i], "mean_abs_standardized_coefficient": float(coefficients[i])}
            for i in range(len(feature_names))
        ),
        key=lambda row: row["mean_abs_standardized_coefficient"],
        reverse=True,
    )
    return {
        "name": name,
        "feature_count": int(features.shape[1]),
        "folds": fold_rows,
        "auc_mean": float(folds["auc"].mean()),
        "auc_std": float(folds["auc"].std(ddof=1)),
        "auc_min": float(folds["auc"].min()),
        "auc_max": float(folds["auc"].max()),
        "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
        "top_standardized_coefficients": ranked[:10],
        "all_standardized_coefficients": ranked,
    }


def write_classifier_svg(models: list[dict], path: Path) -> None:
    """Write a dependency-free audit plot.

    The local machine may already be running a protected scientific Python
    process, so this diagnostic deliberately avoids importing Matplotlib.
    SVG keeps the figure vector-based and inspectable while leaving the
    numerical tables and JSON as the authoritative outputs.
    """

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    def text(x: float, y: float, value: object, size: int = 12, anchor: str = "start", fill: str = "#1f2937", weight: str = "normal") -> str:
        return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" text-anchor="{anchor}" fill="{fill}" font-family="Arial,Helvetica,sans-serif" font-weight="{weight}">{esc(value)}</text>'

    width, height = 1200, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(30, 32, "Real-versus-synthetic two-sample classifier audit", 18, weight="bold"),
        text(30, 53, "Five group-held-out folds; AUC=0.5 is chance. Tables/JSON contain the exact values.", 11, fill="#4b5563"),
    ]
    colors = ["#1769aa", "#d97706"]
    for index, (model, color) in enumerate(zip(models, colors)):
        left = 45 + index * 565
        plot_left, plot_top, plot_width, plot_height = left + 58, 88, 465, 180
        y_min, y_max = 0.4, 1.0
        parts.append(f'<rect x="{left}" y="68" width="535" height="458" rx="8" fill="#f8fafc" stroke="#d1d5db"/>')
        parts.append(text(left + 16, 92, model["name"], 13, weight="bold"))
        for tick in np.arange(0.4, 1.01, 0.1):
            y = plot_top + plot_height - (float(tick) - y_min) / (y_max - y_min) * plot_height
            parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" stroke="#dbe2ea" stroke-width="1"/>')
            parts.append(text(plot_left - 9, y + 4, f"{tick:.1f}", 10, anchor="end", fill="#4b5563"))
        chance_y = plot_top + plot_height - (0.5 - y_min) / (y_max - y_min) * plot_height
        mean_y = plot_top + plot_height - (model["auc_mean"] - y_min) / (y_max - y_min) * plot_height
        parts.append(f'<line x1="{plot_left}" y1="{chance_y:.1f}" x2="{plot_left + plot_width}" y2="{chance_y:.1f}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5,4"/>')
        parts.append(f'<line x1="{plot_left}" y1="{mean_y:.1f}" x2="{plot_left + plot_width}" y2="{mean_y:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="7,4" opacity="0.55"/>')
        points = []
        for fold_index, row in enumerate(model["folds"]):
            x = plot_left + fold_index * plot_width / 4
            y = plot_top + plot_height - (row["auc"] - y_min) / (y_max - y_min) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            parts.append(text(x, plot_top + plot_height + 20, row["fold"], 10, anchor="middle", fill="#4b5563"))
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#374151"/>')
        parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#374151"/>')
        parts.append(text(plot_left + plot_width / 2, plot_top + plot_height + 40, "held-out group fold", 10, anchor="middle", fill="#4b5563"))
        parts.append(text(plot_left - 42, plot_top + plot_height / 2, "ROC AUC", 10, anchor="middle", fill="#4b5563"))
        parts.append(text(plot_left + plot_width - 4, chance_y - 5, "chance", 10, anchor="end", fill="#6b7280"))
        parts.append(text(plot_left + plot_width - 4, mean_y - 5, f"mean {model['auc_mean']:.3f}", 10, anchor="end", fill=color))

        if index == 1:
            top = model if model["name"] == "13 independent scalar proxies" else models[0]
            top = top["top_standardized_coefficients"][:8]
            parts.append(text(left + 16, 322, "scalar model: top mean |standardized coefficient|", 11, weight="bold"))
            max_value = max(row["mean_abs_standardized_coefficient"] for row in top)
            bar_left, bar_width = left + 205, 325
            for bar_index, row in enumerate(reversed(top)):
                y = 342 + bar_index * 20
                label = row["feature"].replace("_", " ")
                bar = row["mean_abs_standardized_coefficient"] / max_value * bar_width
                parts.append(text(bar_left - 8, y + 12, label, 9, anchor="end", fill="#4b5563"))
                parts.append(f'<rect x="{bar_left}" y="{y}" width="{bar:.1f}" height="14" rx="2" fill="#7c3aed"/>')
                parts.append(text(bar_left + bar + 5, y + 12, f"{row['mean_abs_standardized_coefficient']:.2f}", 9, fill="#4b5563"))
    parts.extend([
        text(45, 548, "Descriptive diagnostic only: a high AUC indicates distinguishability, not scientific preference or forecast validity.", 11, fill="#4b5563"),
        "</svg>",
    ])
    path.write_text("\n".join(parts))


def write_outputs(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, raw: np.ndarray, synthetic: np.ndarray, seed: int) -> dict:
    labels = np.r_[np.zeros(len(real_df), dtype=int), np.ones(len(synthetic_df), dtype=int)]
    groups = np.r_[real_df["region_group_id"].astype(str), synthetic_df["region_group_id"].astype(str)]
    scalar_features = np.vstack(
        [real_df[PROXY_COLUMNS].to_numpy(float), synthetic_df[PROXY_COLUMNS].to_numpy(float)]
    )
    coarse_features = np.vstack([block_mean(raw), block_mean(synthetic)])
    coarse_names = [f"block_{r:02d}_{c:02d}" for r in range(16) for c in range(16)]
    models = [
        (print("running scalar classifier", flush=True) or run_classifier("13 independent scalar proxies", scalar_features, PROXY_COLUMNS, labels, groups, seed)),
        (print("running coarse-field classifier", flush=True) or run_classifier("16x16 coarse field", coarse_features, coarse_names, labels, groups, seed)),
    ]
    print("classifiers complete; writing tables and figure", flush=True)
    fold_df = pd.DataFrame([row for model in models for row in model["folds"]])
    coefficient_df = pd.DataFrame(
        [row for model in models for row in model["all_standardized_coefficients"]]
    )
    fold_df.to_csv(TABLE_DIR / "two_sample_classifier_folds.csv", index=False)
    coefficient_df.to_csv(TABLE_DIR / "two_sample_classifier_coefficients.csv", index=False)
    summary_rows = []
    for model in models:
        summary_rows.append(
            {
                "model": model["name"],
                "feature_count": model["feature_count"],
                "auc_mean": model["auc_mean"],
                "auc_std": model["auc_std"],
                "auc_min": model["auc_min"],
                "auc_max": model["auc_max"],
                "balanced_accuracy_mean": model["balanced_accuracy_mean"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLE_DIR / "two_sample_classifier_summary.csv", index=False)

    with (TABLE_DIR / "two_sample_classifier_summary.tex").open("w") as handle:
        handle.write("\\scriptsize\n")
        handle.write("\\setlength{\\tabcolsep}{2pt}\n")
        handle.write("\\begin{longtable}{p{0.28\\textwidth}rrrrrr}\n")
        handle.write("\\caption{Group-held-out classifier two-sample audit.}\\label{tab:c2st}\\\\\n")
        handle.write("\\toprule\nModel & Features & AUC mean & AUC SD & AUC min & AUC max & Balanced accuracy\\\\\n")
        handle.write("\\midrule\n\\endfirsthead\\toprule\nModel & Features & AUC mean & AUC SD & AUC min & AUC max & Balanced accuracy\\\\\n\\midrule\\endhead\n")
        for row in summary_rows:
            label = str(row["model"]).replace("&", r"\&")
            handle.write(
                f"{label} & {row['feature_count']} & {row['auc_mean']:.4f} & {row['auc_std']:.4f} & "
                f"{row['auc_min']:.4f} & {row['auc_max']:.4f} & {row['balanced_accuracy_mean']:.4f}\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{longtable}\n\\normalsize\n")
        handle.write("\\setlength{\\tabcolsep}{6pt}\n")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("writing dependency-free classifier SVG", flush=True)
    write_classifier_svg(models, FIGURE_DIR / "two_sample_classifier.svg")

    result = {
        "status": "PASS",
        "method": "five-fold StratifiedGroupKFold logistic classifier; group-held-out and independent of forecast outcomes",
        "real_count": int(len(real_df)),
        "synthetic_count": int(len(synthetic_df)),
        "real_groups": int(real_df["region_group_id"].nunique()),
        "synthetic_groups": int(synthetic_df["region_group_id"].nunique()),
        "combined_groups": int(len(np.unique(groups))),
        "shared_grouping_note": "Real and generated records use the same connected-region ID when the synthetic sample was conditioned on that region; this is conservative against region memorization.",
        "seed": int(seed),
        "models": models,
        "outputs": [
            "tables/two_sample_classifier_folds.csv",
            "tables/two_sample_classifier_coefficients.csv",
            "tables/two_sample_classifier_summary.csv",
            "tables/two_sample_classifier_summary.tex",
            "figures/two_sample_classifier.svg",
        ],
        "limitations": [
            "A high AUC shows distinguishability, not which distribution is scientifically preferable.",
            "The classifier is a low-dimensional/coarse-field probe and cannot replace the declared independent descriptors.",
            "This audit uses the positive training cache and does not access validation/test outcomes.",
        ],
    }
    (ARTIFACT_DIR / "two_sample_classifier_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    real_df, synthetic_df, raw, synthetic = load_inputs()
    result = write_outputs(real_df, synthetic_df, raw, synthetic, args.seed)
    print(json.dumps({key: result[key] for key in ("status", "real_count", "synthetic_count", "real_groups", "synthetic_groups", "models")}, indent=2))


if __name__ == "__main__":
    main()
