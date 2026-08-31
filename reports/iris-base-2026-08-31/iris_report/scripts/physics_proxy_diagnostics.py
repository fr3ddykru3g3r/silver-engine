#!/usr/bin/env python3
"""Compute independent line-of-sight magnetic-structure diagnostics.

This analysis is deliberately separate from the declared BASE gate and from
the differentiable HJ/PIL training losses.  It uses the prepared positive
training cache (489 real fields) and the completed BASE sample set (128
synthetic fields) to measure field moments, signed-flux balance, spatial
gradients, strong-polarity component structure, centroid separation, and
Fourier scale distribution.  The potential-energy quantity is a normalized
half-space spectral proxy, not a coronal energy measurement: only B_los is
available, so no vector-field or volume extrapolation is claimed.
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
from scipy import ndimage


ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
REPORT = Path(os.environ.get("IRIS_REPORT_DIR", str(Path(__file__).resolve().parents[1])))
EVIDENCE = ROOT / "evidence"
PREPARED = ROOT / "work" / "prepared" / "positive_train_pergroup4"
SYNTHETIC_MANIFEST = ROOT / "work" / "runs" / "base_local_resume" / "samples" / "base" / "synthetic_manifest.csv"
FIG = REPORT / "figures"
TABLE = REPORT / "tables"
ARTIFACT = REPORT / "artifacts"

PIXEL_MM = 2.0
STRONG_G = 150.0
BOOTSTRAP_REPS = 2000
SEED = 2026

FEATURE_LABELS = {
    "unsigned_flux_proxy": r"mean $|B|$ [G]",
    "signed_flux_imbalance": r"$|\langle B\rangle|/\langle|B|\rangle$",
    "rms_field": r"RMS $B$ [G]",
    "field_energy_proxy": r"$\langle B^2\rangle$ [G$^2$]",
    "gradient_rms": r"RMS $|\nabla B|$ [G Mm$^{-1}$]",
    "high_gradient_fraction": r"fraction $|\nabla B|>100$ G Mm$^{-1}$",
    "positive_components": "positive strong-field components",
    "negative_components": "negative strong-field components",
    "strong_component_count": "total strong-field components",
    "strong_centroid_separation": "strong-polarity centroid separation [Mm]",
    "spectral_centroid": "Fourier wavenumber centroid [Mm$^{-1}$]",
    "spectral_high_fraction": "high-wavenumber power fraction",
    "spectral_potential_proxy": "half-space spectral potential proxy",
}


def denormalize(x: np.ndarray) -> np.ndarray:
    denom = np.arcsinh(3000.0 / 250.0)
    return 250.0 * np.sinh(np.clip(x, -1.0, 1.0) * denom)


def load_real() -> tuple[np.ndarray, pd.DataFrame]:
    metadata = json.loads((PREPARED / "metadata.json").read_text())
    raw = np.load(PREPARED / "raw.npy").astype(np.float32)
    if raw.ndim == 4:
        raw = raw[:, 0]
    sample_ids = np.asarray(metadata["sample_ids"], dtype=str)
    if len(sample_ids) != len(raw):
        raise RuntimeError(f"Prepared metadata/raw mismatch: {len(sample_ids)} vs {len(raw)}")
    manifest = pd.read_csv(
        EVIDENCE / "data" / "derived" / "training_manifest.csv.gz",
        usecols=["sample_id", "region_group_id", "latitude_deg"],
        low_memory=False,
    )
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    rows = pd.DataFrame({"sample_id": sample_ids}).merge(manifest, on="sample_id", how="left")
    if rows.region_group_id.isna().any():
        missing = rows.loc[rows.region_group_id.isna(), "sample_id"].head(3).tolist()
        raise RuntimeError(f"Prepared samples missing from manifest: {missing}")
    rows["region_group_id"] = rows["region_group_id"].astype(str)
    return raw, rows


def load_synthetic() -> tuple[np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(SYNTHETIC_MANIFEST)
    arrays = []
    for value in manifest.array_path:
        path = Path(str(value))
        if not path.exists():
            path = SYNTHETIC_MANIFEST.parent / "arrays" / path.name
        arrays.append(denormalize(np.load(path).astype(np.float32)))
    fields = np.stack(arrays)
    rows = manifest.copy()
    rows["region_group_id"] = rows["source_region_group_id"].astype(str)
    return fields, rows


def field_features(fields: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if fields.ndim != 3:
        raise ValueError(f"Expected [N,H,W] fields, got {fields.shape}")
    n, height, width = fields.shape
    gy, gx = np.gradient(fields, PIXEL_MM, PIXEL_MM, axis=(1, 2), edge_order=1)
    grad = np.sqrt(gx * gx + gy * gy)
    abs_field = np.abs(fields)
    signed_imbalance = np.abs(fields.mean(axis=(1, 2))) / np.maximum(abs_field.mean(axis=(1, 2)), 1e-9)

    positive_components = np.zeros(n, dtype=float)
    negative_components = np.zeros(n, dtype=float)
    separations = np.full(n, np.nan, dtype=float)
    structure = np.ones((3, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width), dtype=float)
    for i, field in enumerate(fields):
        pos = field >= STRONG_G
        neg = field <= -STRONG_G
        positive_components[i] = ndimage.label(pos, structure=structure)[1]
        negative_components[i] = ndimage.label(neg, structure=structure)[1]
        pos_weight = np.where(pos, field, 0.0)
        neg_weight = np.where(neg, -field, 0.0)
        pos_total = pos_weight.sum()
        neg_total = neg_weight.sum()
        if pos_total > 0 and neg_total > 0:
            px = float((pos_weight * xx).sum() / pos_total)
            py = float((pos_weight * yy).sum() / pos_total)
            nx = float((neg_weight * xx).sum() / neg_total)
            ny = float((neg_weight * yy).sum() / neg_total)
            separations[i] = float(np.hypot(px - nx, py - ny) * PIXEL_MM)

    freq_y = np.fft.fftfreq(height, d=PIXEL_MM)
    freq_x = np.fft.rfftfreq(width, d=PIXEL_MM)
    wave_number = np.hypot(freq_y[:, None], freq_x[None, :])
    valid = wave_number > 0
    power = np.abs(np.fft.rfft2(fields, axes=(1, 2))) ** 2
    total_power = np.maximum(power[:, valid].sum(axis=1), 1e-20)
    high = wave_number >= 0.125
    spectral_centroid = (power[:, valid] * wave_number[valid]).sum(axis=1) / total_power
    spectral_high_fraction = power[:, high].sum(axis=1) / total_power
    spectral_potential_proxy = (
        (power[:, valid] / np.maximum(2.0 * np.pi * wave_number[valid], 1e-12)).sum(axis=1)
        / float((height * width) ** 2)
    )

    values = {
        "unsigned_flux_proxy": abs_field.mean(axis=(1, 2)),
        "signed_flux_imbalance": signed_imbalance,
        "rms_field": np.sqrt(np.mean(fields * fields, axis=(1, 2))),
        "field_energy_proxy": np.mean(fields * fields, axis=(1, 2)),
        "gradient_rms": np.sqrt(np.mean(grad * grad, axis=(1, 2))),
        "high_gradient_fraction": np.mean(grad > 100.0, axis=(1, 2)),
        "positive_components": positive_components,
        "negative_components": negative_components,
        "strong_component_count": positive_components + negative_components,
        "strong_centroid_separation": separations,
        "spectral_centroid": spectral_centroid,
        "spectral_high_fraction": spectral_high_fraction,
        "spectral_potential_proxy": spectral_potential_proxy,
    }
    return pd.DataFrame(values), power


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    def mean_distance(a: np.ndarray, b: np.ndarray) -> float:
        delta = a[:, None, :] - b[None, :, :]
        return float(np.sqrt(np.sum(delta * delta, axis=2)).mean())

    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return 2.0 * mean_distance(x, y) - mean_distance(x, x) - mean_distance(y, y)


def robust_standardize(real: np.ndarray, synthetic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(real, axis=0)
    q1 = np.nanpercentile(real, 25, axis=0)
    q3 = np.nanpercentile(real, 75, axis=0)
    scale = q3 - q1
    scale = np.where(scale > 1e-8, scale, np.nanstd(real, axis=0) + 1e-6)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return (real - median) / scale, (synthetic - median) / scale


def split_half_baseline(real: np.ndarray, seed: int, repeats: int = 64) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        order = rng.permutation(len(real))
        half = len(real) // 2
        values.append(energy_distance(real[order[:half]], real[order[half:]]))
    return {
        "median": float(np.nanmedian(values)),
        "p90": float(np.nanpercentile(values, 90)),
        "repeats": repeats,
    }


def bootstrap_medians(values: np.ndarray, groups: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    result = np.empty(reps, dtype=float)
    for i in range(reps):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[j] for j in selected])
        result[i] = np.nanmedian(values[indices])
    return result


def summarize(real: pd.DataFrame, synthetic: pd.DataFrame, seed: int, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(seed)
    real_groups = real["region_group_id"].to_numpy(str)
    synthetic_groups = synthetic["region_group_id"].to_numpy(str)
    feature_names = list(FEATURE_LABELS)
    real_matrix = real[feature_names].to_numpy(float)
    synthetic_matrix = synthetic[feature_names].to_numpy(float)
    real_standardized, synthetic_standardized = robust_standardize(real_matrix, synthetic_matrix)
    distance = energy_distance(real_standardized, synthetic_standardized)
    baseline = split_half_baseline(real_standardized, seed + 71)
    for name in feature_names:
        r = real[name].to_numpy(float)
        s = synthetic[name].to_numpy(float)
        r_boot = bootstrap_medians(r, real_groups, reps, rng)
        s_boot = bootstrap_medians(s, synthetic_groups, reps, rng)
        ratio_boot = s_boot / np.maximum(np.abs(r_boot), 1e-12)
        r_med = float(np.nanmedian(r))
        s_med = float(np.nanmedian(s))
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS[name],
            "real_mean": float(np.nanmean(r)),
            "real_median": r_med,
            "real_p10": float(np.nanpercentile(r, 10)),
            "real_p90": float(np.nanpercentile(r, 90)),
            "synthetic_mean": float(np.nanmean(s)),
            "synthetic_median": s_med,
            "synthetic_p10": float(np.nanpercentile(s, 10)),
            "synthetic_p90": float(np.nanpercentile(s, 90)),
            "median_ratio": s_med / max(abs(r_med), 1e-12),
            "median_difference": s_med - r_med,
        })
        bootstrap_rows.append({
            "feature": name,
            "label": FEATURE_LABELS[name],
            "real_groups": int(len(np.unique(real_groups))),
            "synthetic_groups": int(len(np.unique(synthetic_groups))),
            "real_median_boot_low": float(np.nanpercentile(r_boot, 2.5)),
            "real_median_boot_high": float(np.nanpercentile(r_boot, 97.5)),
            "synthetic_median_boot_low": float(np.nanpercentile(s_boot, 2.5)),
            "synthetic_median_boot_high": float(np.nanpercentile(s_boot, 97.5)),
            "ratio_boot_low": float(np.nanpercentile(ratio_boot, 2.5)),
            "ratio_boot_high": float(np.nanpercentile(ratio_boot, 97.5)),
        })
    return (
        pd.DataFrame(rows),
        pd.DataFrame(bootstrap_rows),
        {
            "feature_names": feature_names,
            "real_groups": int(len(np.unique(real_groups))),
            "synthetic_groups": int(len(np.unique(synthetic_groups))),
            "standardized_energy_distance": float(distance),
            "real_split_half_baseline": baseline,
            "distance_to_real_p90_ratio": float(distance / max(baseline["p90"], baseline["median"], 1e-12)),
            "note": "Secondary line-of-sight proxy audit; not a replacement for the predeclared BASE gate or a vector-field/volume-physics result.",
        },
    )


def tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_tables(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    summary.to_csv(TABLE / "physics_proxy_statistics.csv", index=False)
    bootstrap.to_csv(TABLE / "physics_proxy_bootstrap.csv", index=False)
    with (TABLE / "physics_proxy_statistics.tex").open("w") as f:
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{2pt}\n")
        f.write("\\begin{longtable}{p{0.25\\textwidth}rrrrrr}\n")
        f.write("\\caption{Independent line-of-sight magnetic proxy statistics.}\\label{tab:physics-proxy}\\\\\n")
        f.write("\\toprule\nFeature & Real median & Synthetic median & Ratio & Real p10 & Real p90 & Synthetic p90\\\\\n")
        f.write("\\midrule\n\\endfirsthead\n\\toprule\nFeature & Real median & Synthetic median & Ratio & Real p10 & Real p90 & Synthetic p90\\\\\n\\midrule\n\\endhead\n")
        for row in summary.itertuples():
            f.write(
                f"{tex_escape(row.label)} & {row.real_median:.4g} & {row.synthetic_median:.4g} & "
                f"{row.median_ratio:.4g} & {row.real_p10:.4g} & {row.real_p90:.4g} & {row.synthetic_p90:.4g}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{longtable}\n\\setlength{\\tabcolsep}{6pt}\n\\normalsize\n")
    with (TABLE / "physics_proxy_bootstrap.tex").open("w") as f:
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{2pt}\n")
        f.write("\\begin{longtable}{p{0.25\\textwidth}rrrrrr}\n")
        f.write("\\caption{Connected-region bootstrap intervals for independent magnetic proxies.}\\label{tab:physics-proxy-bootstrap}\\\\\n")
        f.write("\\toprule\nFeature & Real low & Real high & Synthetic low & Synthetic high & Ratio low & Ratio high\\\\\n")
        f.write("\\midrule\n\\endfirsthead\n\\toprule\nFeature & Real low & Real high & Synthetic low & Synthetic high & Ratio low & Ratio high\\\\\n\\midrule\n\\endhead\n")
        for row in bootstrap.itertuples():
            f.write(
                f"{tex_escape(row.label)} & {row.real_median_boot_low:.4g} & {row.real_median_boot_high:.4g} & "
                f"{row.synthetic_median_boot_low:.4g} & {row.synthetic_median_boot_high:.4g} & "
                f"{row.ratio_boot_low:.4g} & {row.ratio_boot_high:.4g}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{longtable}\n\\setlength{\\tabcolsep}{6pt}\n\\normalsize\n")


def plot_distributions(real: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    names = list(FEATURE_LABELS)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        r = real[name].to_numpy(float)
        s = synthetic[name].to_numpy(float)
        combined = np.concatenate([r[np.isfinite(r)], s[np.isfinite(s)]])
        if len(combined) == 0:
            continue
        lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 24)
        ax.hist(r, bins=bins, density=True, alpha=0.45, color="#1769aa", label="real")
        ax.hist(s, bins=bins, density=True, alpha=0.45, color="#d97706", label="synthetic")
        ax.axvline(np.nanmedian(r), color="#1769aa", linewidth=1.2)
        ax.axvline(np.nanmedian(s), color="#d97706", linewidth=1.2)
        ax.set_title(FEATURE_LABELS[name], fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Independent magnetic proxy distributions: real versus BASE synthetic", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "physics_proxy_distributions.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "physics_proxy_distributions.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def radial_spectrum(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width_rfft = power.shape[1:]
    fy = np.fft.fftfreq(height, d=PIXEL_MM)
    fx = np.fft.rfftfreq((width_rfft - 1) * 2, d=PIXEL_MM)
    k = np.hypot(fy[:, None], fx[None, :])
    bins = np.linspace(0.0, float(k.max()), 24)
    centers = (bins[:-1] + bins[1:]) / 2.0
    spectra = []
    for sample in power:
        normalized = sample / max(float(sample[k > 0].sum()), 1e-20)
        sums, _ = np.histogram(k, bins=bins, weights=normalized)
        counts, _ = np.histogram(k, bins=bins)
        spectra.append(sums / np.maximum(counts, 1))
    spectra = np.asarray(spectra)
    return centers, np.nanmean(spectra, axis=0), np.nanstd(spectra, axis=0)


def plot_spectral(real_power: np.ndarray, synthetic_power: np.ndarray) -> None:
    rk, rmean, rstd = radial_spectrum(real_power)
    sk, smean, sstd = radial_spectrum(synthetic_power)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(rk[1:], rmean[1:], color="#1769aa", label="real")
    axes[0].fill_between(rk[1:], np.maximum(rmean[1:] - rstd[1:], 1e-12), rmean[1:] + rstd[1:], color="#1769aa", alpha=0.18)
    axes[0].plot(sk[1:], smean[1:], color="#d97706", label="synthetic")
    axes[0].fill_between(sk[1:], np.maximum(smean[1:] - sstd[1:], 1e-12), smean[1:] + sstd[1:], color="#d97706", alpha=0.18)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("radial spatial frequency [cycles Mm$^{-1}$]")
    axes[0].set_ylabel("mean normalized Fourier power per annulus")
    axes[0].set_title("Scale distribution", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    for name, color, frame in [
        ("real", "#1769aa", real_power),
        ("synthetic", "#d97706", synthetic_power),
    ]:
        feature, _ = field_features_from_power(frame)
        axes[1].hist(feature, bins=24, density=True, alpha=0.45, color=color, label=name)
    axes[1].set_xlabel("high-wavenumber power fraction")
    axes[1].set_ylabel("density")
    axes[1].set_title("Small-scale power proxy", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Fourier-scale diagnostics (secondary, line-of-sight only)", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "physics_proxy_spectral.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "physics_proxy_spectral.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def field_features_from_power(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width_rfft = power.shape[1:]
    fy = np.fft.fftfreq(height, d=PIXEL_MM)
    fx = np.fft.rfftfreq((width_rfft - 1) * 2, d=PIXEL_MM)
    k = np.hypot(fy[:, None], fx[None, :])
    valid = k > 0
    total = np.maximum(power[:, valid].sum(axis=1), 1e-20)
    return power[:, k >= 0.125].sum(axis=1) / total, total


def plot_topology(real: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    names = ["strong_component_count", "signed_flux_imbalance", "strong_centroid_separation"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, name in zip(axes, names):
        ax.hist(real[name], bins=24, density=True, alpha=0.45, color="#1769aa", label="real")
        ax.hist(synthetic[name], bins=24, density=True, alpha=0.45, color="#d97706", label="synthetic")
        ax.set_title(FEATURE_LABELS[name], fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Strong-polarity topology and balance proxies", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "physics_proxy_topology.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "physics_proxy_topology.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPORT))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    global FIG, TABLE, ARTIFACT
    out = Path(args.out_dir)
    FIG = out / "figures"
    TABLE = out / "tables"
    ARTIFACT = out / "artifacts"
    for directory in (FIG, TABLE, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)

    real_fields, real_rows = load_real()
    synthetic_fields, synthetic_rows = load_synthetic()
    real_values, real_power = field_features(real_fields)
    synthetic_values, synthetic_power = field_features(synthetic_fields)
    real = pd.concat([real_rows.reset_index(drop=True), real_values], axis=1)
    synthetic = pd.concat([synthetic_rows.reset_index(drop=True), synthetic_values], axis=1)
    real.to_csv(TABLE / "physics_proxy_real.csv", index=False)
    synthetic.to_csv(TABLE / "physics_proxy_synthetic.csv", index=False)
    summary, bootstrap, audit = summarize(real, synthetic, args.seed, args.bootstrap_reps)
    write_tables(summary, bootstrap)
    plot_distributions(real, synthetic)
    plot_spectral(real_power, synthetic_power)
    plot_topology(real, synthetic)
    result = {
        "status": "PASS",
        "method": "independent line-of-sight magnetic proxy diagnostics",
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_reps,
        "real_count": int(len(real)),
        "synthetic_count": int(len(synthetic)),
        "real_groups": int(real.region_group_id.nunique()),
        "synthetic_groups": int(synthetic.region_group_id.nunique()),
        "pixel_size_mm": PIXEL_MM,
        "strong_field_threshold_gauss": STRONG_G,
        "summary": summary.to_dict(orient="records"),
        "cluster_bootstrap": bootstrap.to_dict(orient="records"),
        "secondary_multivariate_audit": audit,
        "limitations": [
            "B_los-only diagnostics; no vector magnetic field or coronal volume extrapolation.",
            "The half-space spectral potential quantity is a normalized scale proxy, not a calibrated energy.",
            "These results are descriptive secondary diagnostics and do not modify the frozen BASE gate.",
        ],
        "outputs": [
            "tables/physics_proxy_real.csv",
            "tables/physics_proxy_synthetic.csv",
            "tables/physics_proxy_statistics.csv",
            "tables/physics_proxy_bootstrap.csv",
            "figures/physics_proxy_distributions.png",
            "figures/physics_proxy_spectral.png",
            "figures/physics_proxy_topology.png",
        ],
    }
    (ARTIFACT / "physics_proxy_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "real_count": result["real_count"],
        "synthetic_count": result["synthetic_count"],
        "real_groups": result["real_groups"],
        "synthetic_groups": result["synthetic_groups"],
        "energy_distance_ratio": audit["distance_to_real_p90_ratio"],
        "outputs": result["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
