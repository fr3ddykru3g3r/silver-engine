#!/usr/bin/env python3
"""Train-only duplicate, collapse, and nearest-neighbor diagnostics.

The audit uses the fixed 248-field real evaluation subset and the 128 already
generated BASE fields. Distances are measured in the declared 16x16 pooled
asinh representation. It is descriptive: no forecast outcomes, checkpoint
selection, or gate definition is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
REPORT = Path(os.environ.get("IRIS_REPORT_DIR", str(Path(__file__).resolve().parents[1])))
AUDIT = ROOT / "work" / "runs" / "base_local_resume" / "audit"
MANIFEST = ROOT / "work" / "runs" / "base_local_resume" / "samples" / "base" / "synthetic_manifest.csv"
PREPARED = ROOT / "work" / "prepared" / "positive_train_pergroup4"
TABLE = REPORT / "tables"
FIGURE = REPORT / "figures"
ARTIFACT = REPORT / "artifacts"


def distance(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    right = left if right is None else right
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    return np.sqrt(np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0))


def pooled(fields: np.ndarray) -> np.ndarray:
    transformed = np.arcsinh(fields / 300.0) / np.arcsinh(10.0)
    return transformed.reshape(len(fields), 16, 8, 16, 8).mean(axis=(2, 4)).reshape(len(fields), -1)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "p10": float(np.nanpercentile(values, 10)),
        "median": float(np.nanmedian(values)),
        "p90": float(np.nanpercentile(values, 90)),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
    }


def write_figure(distributions: dict[str, np.ndarray], path: Path) -> None:
    labels = ["real→real\nnearest", "synthetic→synthetic\nnearest", "synthetic→real\nnearest", "synthetic same\nsource group", "synthetic different\nsource group"]
    values = [distributions[key] for key in ["real_to_real", "synthetic_to_synthetic", "synthetic_to_real", "synthetic_same_group", "synthetic_different_group"]]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "#EAF2F8", "edgecolor": "#1769AA"}, medianprops={"color": "#B42318", "linewidth": 2})
    ax.set_ylabel("Euclidean distance in pooled asinh field space")
    ax.set_title("BASE nearest-neighbor and source-group distance audit")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    real_rows = pd.read_csv(AUDIT / "real_generic.csv")
    synthetic_rows = pd.read_csv(AUDIT / "synthetic_generic.csv")
    manifest = pd.read_csv(MANIFEST, usecols=["synthetic_id", "source_region_group_id", "array_path"])
    manifest_by_id = manifest.set_index("synthetic_id")
    synthetic_rows["region_group_id"] = synthetic_rows.synthetic_id.map(manifest_by_id.source_region_group_id)
    if synthetic_rows.region_group_id.isna().any():
        raise RuntimeError("Synthetic audit rows missing source connected-region IDs")
    synthetic_rows["region_group_id"] = synthetic_rows.region_group_id.astype(str)

    metadata = json.loads((PREPARED / "metadata.json").read_text())
    raw = np.load(PREPARED / "raw.npy", mmap_mode="r")
    if raw.ndim == 4:
        raw = raw[:, 0]
    real_index = {str(sample_id): index for index, sample_id in enumerate(metadata["sample_ids"])}
    real_fields = np.stack([raw[real_index[str(sample_id)]] for sample_id in real_rows.sample_id])
    synthetic_fields = []
    for value in synthetic_rows.synthetic_id.map(manifest_by_id.array_path):
        path = Path(str(value))
        if not path.exists():
            path = MANIFEST.parent / "arrays" / path.name
        normalized = np.load(path).astype(np.float32)
        synthetic_fields.append(250.0 * np.sinh(np.clip(normalized, -1.0, 1.0) * np.arcsinh(3000.0 / 250.0)))
    synthetic_fields = np.stack(synthetic_fields)

    real_vectors = pooled(real_fields)
    synthetic_vectors = pooled(synthetic_fields)
    real_distance = distance(real_vectors)
    synthetic_distance = distance(synthetic_vectors)
    cross_distance = distance(synthetic_vectors, real_vectors)
    np.fill_diagonal(real_distance, np.nan)
    np.fill_diagonal(synthetic_distance, np.nan)

    real_to_real = np.nanmin(real_distance, axis=1)
    synthetic_to_synthetic = np.nanmin(synthetic_distance, axis=1)
    synthetic_to_real = np.nanmin(cross_distance, axis=1)
    same_group = synthetic_rows.region_group_id.to_numpy()[:, None] == real_rows.region_group_id.astype(str).to_numpy()[None, :]
    same_group_values = np.where(same_group, cross_distance, np.nan)
    different_group_values = np.where(~same_group, cross_distance, np.nan)
    synthetic_same_group = np.nanmin(same_group_values, axis=1)
    synthetic_different_group = np.nanmin(different_group_values, axis=1)

    synthetic_ids = [hashlib.sha256(np.asarray(field, dtype=np.float32).tobytes()).hexdigest() for field in synthetic_fields]
    duplicate_count = len(synthetic_ids) - len(set(synthetic_ids))
    same_source_pairs = []
    cross_source_pairs = []
    for i in range(len(synthetic_rows)):
        for j in range(i + 1, len(synthetic_rows)):
            target = same_source_pairs if synthetic_rows.region_group_id.iloc[i] == synthetic_rows.region_group_id.iloc[j] else cross_source_pairs
            target.append(float(synthetic_distance[i, j]))

    distributions = {
        "real_to_real": real_to_real,
        "synthetic_to_synthetic": synthetic_to_synthetic,
        "synthetic_to_real": synthetic_to_real,
        "synthetic_same_group": synthetic_same_group,
        "synthetic_different_group": synthetic_different_group,
    }
    summary = pd.DataFrame([{"distribution": name, **quantiles(values)} for name, values in distributions.items()])
    summary.to_csv(TABLE / "memorization_distance_summary.csv", index=False)
    with (TABLE / "memorization_distance_summary.tex").open("w") as handle:
        handle.write("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\nDistribution & $n$ & p10 & median & p90 & minimum\\\\\n\\midrule\n")
        for row in summary.itertuples(index=False):
            label = str(row.distribution).replace("_", r"\_")
            handle.write(f"\\texttt{{{label}}} & {row.n} & {row.p10:.4f} & {row.median:.4f} & {row.p90:.4f} & {row.minimum:.4f}\\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\normalsize\n\\setlength{\\tabcolsep}{6pt}\n")
    write_figure(distributions, FIGURE / "memorization_nearest_neighbor.png")

    payload = {
        "status": "PASS",
        "method": "16x16 pooled asinh field-space nearest-neighbor and exact-hash audit",
        "seed": int(args.seed),
        "real_rows": int(len(real_rows)),
        "synthetic_rows": int(len(synthetic_rows)),
        "real_groups": int(real_rows.region_group_id.nunique()),
        "synthetic_groups": int(synthetic_rows.region_group_id.nunique()),
        "exact_normalized_array_duplicate_count": int(duplicate_count),
        "same_source_pair_count": int(len(same_source_pairs)),
        "cross_source_pair_count": int(len(cross_source_pairs)),
        "same_source_pair_distance": quantiles(np.asarray(same_source_pairs, dtype=float)),
        "cross_source_pair_distance": quantiles(np.asarray(cross_source_pairs, dtype=float)),
        "distributions": {name: quantiles(values) for name, values in distributions.items()},
        "outputs": [
            "tables/memorization_distance_summary.csv",
            "tables/memorization_distance_summary.tex",
            "figures/memorization_nearest_neighbor.png",
        ],
        "limitations": [
            "Nearest-neighbor distance is a low-dimensional similarity check, not proof of non-memorization or physical realism.",
            "The synthetic sample is conditioned on label and latitude; same-source comparisons are descriptive and may reflect shared conditioning strata.",
            "No validation/test flare outcomes are accessed and no model gate is changed.",
        ],
    }
    (ARTIFACT / "memorization_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
