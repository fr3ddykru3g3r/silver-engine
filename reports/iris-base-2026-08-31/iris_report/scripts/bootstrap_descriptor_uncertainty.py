#!/usr/bin/env python3
"""Cluster-bootstrap uncertainty for the fixed BASE descriptor audit.

Rows from the same connected region are not independent.  This script therefore
resamples region groups, not individual rows, and reports percentile intervals
for real/synthetic median ratios.  It is an uncertainty diagnostic, not a
replacement for the multivariate energy-distance gate.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
REPORT = Path(os.environ.get("IRIS_REPORT_DIR", str(Path(__file__).resolve().parents[1])))
RUN = ROOT / "work" / "runs" / "base_local_resume"
AUDIT = RUN / "audit"
ARTIFACT = REPORT / "artifacts"
TABLE = REPORT / "tables"
FIG = REPORT / "figures"


def group_series(df: pd.DataFrame, id_col: str, external_map: dict[str, str] | None = None) -> pd.Series:
    if "region_group_id" in df.columns:
        return df["region_group_id"].astype(str)
    if external_map is not None and id_col in df.columns:
        mapped = df[id_col].astype(str).map(external_map)
        if mapped.notna().all():
            return mapped.astype(str)
    if id_col not in df.columns and "synthetic_id" in df.columns:
        id_col = "synthetic_id"
    extracted = df[id_col].astype(str).str.extract(r"(RG\d+)", expand=False)
    return extracted.fillna(df[id_col].astype(str))


def cluster_median_samples(df: pd.DataFrame, group: pd.Series, feature: str,
                           rng: np.random.Generator, reps: int) -> np.ndarray:
    work = df[[feature]].copy()
    work["_group"] = group.to_numpy()
    clusters = {gid: g[feature].to_numpy(float) for gid, g in work.groupby("_group", sort=True)}
    gids = np.array(sorted(clusters), dtype=object)
    out = np.empty(reps, dtype=float)
    for i in range(reps):
        chosen = rng.choice(gids, size=len(gids), replace=True)
        values = np.concatenate([clusters[gid] for gid in chosen])
        out[i] = float(np.nanmedian(values))
    return out


def percentile(x: np.ndarray, q: float) -> float:
    finite = x[np.isfinite(x)]
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def main() -> None:
    reps = int(os.environ.get("IRIS_BOOTSTRAP_REPS", "2500"))
    seed = int(os.environ.get("IRIS_BOOTSTRAP_SEED", "2026"))
    rng = np.random.default_rng(seed)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    families = {
        "generic": ("real_generic.csv", "synthetic_generic.csv", "sample_id", [
            "log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction",
        ]),
        "geometry": ("real_hard_geometry.csv", "synthetic_hard_geometry.csv", "sample_id", [
            "log_sep", "log_strong_flux_density",
        ]),
        "pil": ("real_hard_pil.csv", "synthetic_hard_pil.csv", "sample_id", [
            "mean_grad", "rms_grad", "top10_grad", "frac_gt100", "pil_area_fraction",
        ]),
    }
    generic_real = pd.read_csv(AUDIT / "real_generic.csv")
    real_group_map = dict(
        zip(generic_real["sample_id"].astype(str), generic_real["region_group_id"].astype(str))
    )

    rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    for family, (real_name, synthetic_name, id_col, features) in families.items():
        real = pd.read_csv(AUDIT / real_name)
        synthetic = pd.read_csv(AUDIT / synthetic_name)
        real_groups = group_series(real, id_col, real_group_map)
        synthetic_groups = group_series(synthetic, id_col)
        for feature in features:
            r = real[feature].to_numpy(float)
            s = synthetic[feature].to_numpy(float)
            real_boot = cluster_median_samples(real, real_groups, feature, rng, reps)
            synthetic_boot = cluster_median_samples(synthetic, synthetic_groups, feature, rng, reps)
            real_median = float(np.nanmedian(r))
            synthetic_median = float(np.nanmedian(s))
            ratio_boot = np.divide(
                synthetic_boot,
                real_boot,
                out=np.full_like(synthetic_boot, np.nan),
                where=np.abs(real_boot) > 1e-12,
            )
            ratio = synthetic_median / real_median if abs(real_median) > 1e-12 else float("nan")
            row = {
                "family": family,
                "feature": feature,
                "real_median": real_median,
                "synthetic_median": synthetic_median,
                "median_ratio": ratio,
                "ratio_ci95_low": percentile(ratio_boot, 0.025),
                "ratio_ci95_high": percentile(ratio_boot, 0.975),
                "median_difference": synthetic_median - real_median,
                "real_groups": int(real_groups.nunique()),
                "synthetic_groups": int(synthetic_groups.nunique()),
                "real_rows": int(len(real)),
                "synthetic_rows": int(len(synthetic)),
            }
            rows.append(row)
            if np.isfinite(ratio) and np.isfinite(row["ratio_ci95_low"]) and np.isfinite(row["ratio_ci95_high"]):
                plot_rows.append(row)

    result = {
        "status": "PASS",
        "method": "cluster bootstrap of connected-region medians",
        "seed": seed,
        "replicates": reps,
        "note": "Intervals quantify descriptor median uncertainty only; they do not replace the multivariate energy-distance audit.",
        "families": {
            family: {
                "real_file": values[0],
                "synthetic_file": values[1],
                "real_groups": int(group_series(pd.read_csv(AUDIT / values[0]), values[2]).nunique()),
                "synthetic_groups": int(group_series(pd.read_csv(AUDIT / values[1]), values[2]).nunique()),
            }
            for family, values in families.items()
        },
        "rows": rows,
    }
    (ARTIFACT / "bootstrap_descriptor_uncertainty.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n"
    )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "bootstrap_descriptor_uncertainty.csv", index=False)

    def tex(value: object) -> str:
        return str(value).replace("_", r"\_")

    with (TABLE / "bootstrap_descriptor_uncertainty.tex").open("w") as f:
        f.write("\\scriptsize\n")
        f.write("\\begin{longtable}{llllrrr}\n")
        f.write("\\caption{Cluster-bootstrap uncertainty for descriptor medians.}\\label{tab:bootstrap}\\\\\n")
        f.write("\\toprule\nFamily & Feature & Real median & Synthetic median & Ratio & 95\\% low & 95\\% high\\\\\n")
        f.write("\\midrule\n\\endfirsthead\n\\toprule\nFamily & Feature & Real median & Synthetic median & Ratio & 95\\% low & 95\\% high\\\\\n\\midrule\n\\endhead\n")
        for row in rows:
            f.write(
                f"{tex(row['family'])} & {tex(row['feature'])} & {row['real_median']:.4f} & "
                f"{row['synthetic_median']:.4f} & {row['median_ratio']:.4f} & "
                f"{row['ratio_ci95_low']:.4f} & {row['ratio_ci95_high']:.4f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{longtable}\n\\normalsize\n")

    labels = [f"{row['family']}:{row['feature']}" for row in plot_rows]
    ratios = np.array([float(row["median_ratio"]) for row in plot_rows])
    low = np.array([float(row["ratio_ci95_low"]) for row in plot_rows])
    high = np.array([float(row["ratio_ci95_high"]) for row in plot_rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(labels) * 0.35)))
    ax.errorbar(ratios, y, xerr=[ratios - low, high - ratios], fmt="o", color="#1769aa",
                ecolor="#7b8794", elinewidth=1.3, capsize=3)
    ax.axvline(1.0, color="#d97706", linestyle="--", linewidth=1.3, label="equal median")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Synthetic / real median ratio (95% cluster-bootstrap interval)")
    ax.set_title("Descriptor-scale uncertainty: groups, not rows, were resampled", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "bootstrap_descriptor_uncertainty.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "bootstrap_descriptor_uncertainty.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"status": "PASS", "rows": len(rows), "figure": str(FIG / "bootstrap_descriptor_uncertainty.png")}, indent=2))


if __name__ == "__main__":
    main()
