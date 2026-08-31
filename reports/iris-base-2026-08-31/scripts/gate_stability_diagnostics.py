#!/usr/bin/env python3
"""Connected-group bootstrap stability audit for the corrected BASE gate.

This is a train-only uncertainty diagnostic. It resamples connected physical
regions, not individual images, from the already completed real and synthetic
audits. The corrected generic energy-distance ratio and all five gate
criteria are recomputed for each replicate. It does not change the frozen
gate, select a checkpoint, or inspect forecast outcomes.
"""

from __future__ import annotations

import argparse
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

CORE = ["log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction"]
CRITERIA = ["flux_median_ratio", "active_area_median_ratio", "diversity_ratio", "saturation_fraction", "generic_distance_ratio"]
LABELS = {
    "flux_median_ratio": "flux median ratio",
    "active_area_median_ratio": "active-area ratio",
    "diversity_ratio": "diversity ratio",
    "saturation_fraction": "saturation fraction",
    "generic_distance_ratio": "corrected distance ratio",
}


def pairwise_distance(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    right = left if right is None else right
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    squared = np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)
    return np.sqrt(squared)


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.asarray(sorted(set(groups.astype(str))), dtype=object)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups.astype(str) == group) for group in drawn]
    return np.concatenate(pieces).astype(int)


def inverse_log_mean(values: np.ndarray) -> np.ndarray:
    return 50.0 * np.expm1(values.astype(float))


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    ordered = list(summary.criterion) + ["joint_gate"]
    labels = [LABELS.get(value, "all criteria jointly") for value in ordered]
    values = list(summary.pass_fraction) + [float(summary.joint_pass_fraction.iloc[0])]
    colors = ["#18864B" if value >= 0.95 else "#D97706" if value >= 0.50 else "#B42318" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors, edgecolor="#17202A", linewidth=0.5)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of connected-group bootstrap replicates passing")
    ax.set_title("BASE gate criterion stability under connected-group bootstrap")
    ax.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, values):
        ax.text(min(value + 0.018, 1.01), position, f"{value:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("Use at least 100 bootstrap replicates for this diagnostic")

    real = pd.read_csv(AUDIT / "real_generic.csv")
    synthetic = pd.read_csv(AUDIT / "synthetic_generic.csv")
    manifest = pd.read_csv(MANIFEST, usecols=["synthetic_id", "source_region_group_id", "array_path"])
    manifest_by_id = manifest.set_index("synthetic_id")
    source_group = manifest_by_id["source_region_group_id"]
    synthetic["region_group_id"] = synthetic.synthetic_id.map(source_group)
    if synthetic.region_group_id.isna().any():
        raise RuntimeError("Synthetic audit rows missing source connected-region IDs")
    synthetic["region_group_id"] = synthetic.region_group_id.astype(str)

    metadata = json.loads((PREPARED / "metadata.json").read_text())
    raw = np.load(PREPARED / "raw.npy", mmap_mode="r")
    if raw.ndim == 4:
        raw = raw[:, 0]
    real_by_id = {str(sample_id): index for index, sample_id in enumerate(metadata["sample_ids"])}
    real_fields = np.stack([raw[real_by_id[str(sample_id)]] for sample_id in real.sample_id])
    synthetic_paths = synthetic.synthetic_id.map(manifest_by_id["array_path"])
    fields = []
    for value in synthetic_paths:
        path = Path(str(value))
        if not path.exists():
            path = MANIFEST.parent / "arrays" / path.name
        fields.append(np.load(path).astype(np.float32))
    synthetic_fields = np.stack(fields)
    # Synthetic arrays are normalized asinh values; invert the declared transform.
    synthetic_fields = 250.0 * np.sinh(np.clip(synthetic_fields, -1.0, 1.0) * np.arcsinh(3000.0 / 250.0))

    metrics = json.loads((AUDIT / "v2_manipulation_metrics.json").read_text())
    scaling = metrics["robust_scaling"]
    med = np.asarray(scaling["generic_core_median"], dtype=float)
    scale = np.asarray(scaling["generic_core_scale"], dtype=float)
    real_core = real[CORE].to_numpy(float)
    synthetic_core = synthetic[CORE].to_numpy(float)
    real_standardized = (real_core - med) / scale
    synthetic_standardized = (synthetic_core - med) / scale

    # Precompute all pairwise matrices once. Bootstrap replicates then only
    # index these matrices, keeping the audit fast and deterministic.
    real_distance = pairwise_distance(real_standardized)
    synthetic_distance = pairwise_distance(synthetic_standardized)
    cross_distance = pairwise_distance(real_standardized, synthetic_standardized)

    def pooled(fields: np.ndarray) -> np.ndarray:
        transformed = np.arcsinh(fields / 300.0) / np.arcsinh(10.0)
        return transformed.reshape(len(fields), 16, 8, 16, 8).mean(axis=(2, 4)).reshape(len(fields), -1)

    real_pooled_distance = pairwise_distance(pooled(real_fields))
    synthetic_pooled_distance = pairwise_distance(pooled(synthetic_fields))

    real_groups = real.region_group_id.astype(str).to_numpy()
    synthetic_groups = synthetic.region_group_id.astype(str).to_numpy()
    real_flux = inverse_log_mean(real.log_mean_abs.to_numpy(float))
    synthetic_flux = inverse_log_mean(synthetic.log_mean_abs.to_numpy(float))
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | bool | int]] = []

    for replicate in range(args.bootstrap_reps):
        real_index = bootstrap_indices(real_groups, rng)
        synthetic_index = bootstrap_indices(synthetic_groups, rng)
        real_flux_median = float(np.nanmedian(real_flux[real_index]))
        synthetic_flux_median = float(np.nanmedian(synthetic_flux[synthetic_index]))
        active_ratio = float(np.nanmedian(synthetic.active_fraction.to_numpy(float)[synthetic_index]) / max(np.nanmedian(real.active_fraction.to_numpy(float)[real_index]), 1e-9))
        diversity_real = float(real_pooled_distance[np.ix_(real_index, real_index)].mean())
        diversity_synthetic = float(synthetic_pooled_distance[np.ix_(synthetic_index, synthetic_index)].mean())
        distance = float(
            2.0 * cross_distance[np.ix_(real_index, synthetic_index)].mean()
            - real_distance[np.ix_(real_index, real_index)].mean()
            - synthetic_distance[np.ix_(synthetic_index, synthetic_index)].mean()
        )
        values = {
            "flux_median_ratio": synthetic_flux_median / max(real_flux_median, 1e-9),
            "active_area_median_ratio": active_ratio,
            "diversity_ratio": diversity_synthetic / max(diversity_real, 1e-9),
            "saturation_fraction": float(np.nanmean(synthetic.saturation_fraction.to_numpy(float)[synthetic_index])),
            "generic_distance_ratio": distance / max(float(metrics["generic_real_split_half_distance"]["p90"]), 1e-9),
        }
        passed = {
            "flux_median_ratio": 0.50 <= values["flux_median_ratio"] <= 2.00,
            "active_area_median_ratio": 0.50 <= values["active_area_median_ratio"] <= 2.00,
            "diversity_ratio": 0.40 <= values["diversity_ratio"] <= 2.50,
            "saturation_fraction": values["saturation_fraction"] < 0.01,
            "generic_distance_ratio": values["generic_distance_ratio"] <= 8.0,
        }
        rows.append({"replicate": replicate, **values, **{f"{key}_pass": value for key, value in passed.items()}, "joint_gate_pass": all(passed.values())})

    replicates = pd.DataFrame(rows)
    TABLE.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(TABLE / "gate_stability_replicates.csv", index=False)

    point = {
        "flux_median_ratio": float(metrics["synthetic_to_real_flux_median_ratio"]),
        "active_area_median_ratio": float(metrics["synthetic_to_real_active_area_median_ratio"]),
        "diversity_ratio": float(metrics["synthetic_to_real_diversity_ratio"]),
        "saturation_fraction": float(metrics["synthetic_saturation_fraction_abs_gt_2900G"]),
        "generic_distance_ratio": float(metrics["generic_distance_to_real_baseline_ratio"]),
    }
    summary_rows = []
    for criterion in CRITERIA:
        values = replicates[criterion].to_numpy(float)
        summary_rows.append({
            "criterion": criterion,
            "point_estimate": point[criterion],
            "bootstrap_median": float(np.nanmedian(values)),
            "ci_low": float(np.nanpercentile(values, 2.5)),
            "ci_high": float(np.nanpercentile(values, 97.5)),
            "pass_fraction": float(replicates[f"{criterion}_pass"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    joint_fraction = float(replicates.joint_gate_pass.mean())
    summary["joint_pass_fraction"] = joint_fraction
    summary.to_csv(TABLE / "gate_stability_summary.csv", index=False)
    with (TABLE / "gate_stability_summary.tex").open("w") as handle:
        handle.write("\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
        handle.write("\\begin{tabular}{lrrrrr}\n\\toprule\nCriterion & point & bootstrap median & 2.5\\% & 97.5\\% & pass fraction\\\\\n\\midrule\n")
        for row in summary.itertuples(index=False):
            criterion_label = str(row.criterion).replace("_", r"\_")
            handle.write(f"\\texttt{{{criterion_label}}} & {row.point_estimate:.4f} & {row.bootstrap_median:.4f} & {row.ci_low:.4f} & {row.ci_high:.4f} & {row.pass_fraction:.3f}\\\\\n")
        handle.write(f"\\texttt{{joint gate}} & -- & -- & -- & -- & {joint_fraction:.3f}\\\\\n\\bottomrule\n\\end{{tabular}}\n\\normalsize\n\\setlength{{\\tabcolsep}}{{6pt}}\n")
    write_figure(summary.assign(joint_pass_fraction=joint_fraction), FIGURE / "gate_stability.png")

    payload = {
        "status": "PASS",
        "method": "connected-region bootstrap with replacement; corrected five-criterion BASE gate",
        "seed": int(args.seed),
        "bootstrap_replicates": int(args.bootstrap_reps),
        "real_rows": int(len(real)),
        "synthetic_rows": int(len(synthetic)),
        "real_groups": int(real.region_group_id.nunique()),
        "synthetic_groups": int(synthetic.region_group_id.nunique()),
        "joint_gate_pass_fraction": joint_fraction,
        "criteria": summary.to_dict(orient="records"),
        "gate_definition": metrics["generic_fidelity_gate_definition"],
        "outputs": [
            "tables/gate_stability_replicates.csv",
            "tables/gate_stability_summary.csv",
            "tables/gate_stability_summary.tex",
            "figures/gate_stability.png",
        ],
        "limitations": [
            "This is uncertainty around the finite train-only audit sample, not a probability that the generator is scientifically valid.",
            "It resamples the completed 128 synthetic outputs and does not add new model samples.",
            "It does not inspect validation/test outcomes or alter the frozen gate definition.",
        ],
    }
    (ARTIFACT / "gate_stability_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
