#!/usr/bin/env python3
"""Evaluate generated counterfactual destruction controls without forecasts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

RUN_ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
MODEL_SRC = RUN_ROOT / "source" / "iris-model"
COMMON_SRC = RUN_ROOT / "source" / "common"
sys.path.insert(0, str(MODEL_SRC))
sys.path.insert(0, str(COMMON_SRC))

from evaluate_generator_v2 import (  # noqa: E402
    energy_distance,
    generic_descriptor,
    hard_geometry_descriptor,
    hard_pil_descriptor,
    load_synthetic,
    robust_standardize,
    split_half_baseline,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-audit", required=True)
    ap.add_argument("--control-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    primary = Path(args.primary_audit)
    control_root = Path(args.control_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_gen = pd.read_csv(primary / "real_generic.csv")
    real_geom = pd.read_csv(primary / "real_hard_geometry.csv")
    real_pil = pd.read_csv(primary / "real_hard_pil.csv")
    parent_manifest = control_root.parent / "samples" / "base" / "synthetic_manifest.csv"
    parent, parent_lat, parent_m = load_synthetic(parent_manifest)
    parent_g = generic_descriptor(parent)
    parent_geom = hard_geometry_descriptor(parent, parent_lat)
    parent_pil = hard_pil_descriptor(parent)
    gen_cols = ["log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction", "saturation_fraction"]
    geom_cols = ["hemi_ux", "hemi_uy", "log_sep", "log_strong_flux_density", "has_bipole"]
    pil_cols = ["mean_grad", "rms_grad", "top10_grad", "frac_gt100", "frac_gt250", "frac_gt500", "pil_area_fraction", "has_pil"]
    core = [0, 1, 2, 3, 4]
    real_g = real_gen[gen_cols].to_numpy(np.float32)
    real_geom_a = real_geom[geom_cols].to_numpy(np.float32)
    real_pil_a = real_pil[pil_cols].to_numpy(np.float32)
    primary_metrics = json.loads((primary / "v2_manipulation_metrics.json").read_text())
    names = ["BASE"]
    rows = []

    def one(name: str, arrays: torch.Tensor, lat: np.ndarray, g: np.ndarray, geom: np.ndarray, pil: np.ndarray) -> dict:
        rg, sg, _, _ = robust_standardize(real_g[:, core], g[:, core])
        rgeom, sgeom, _, _ = robust_standardize(real_geom_a, geom)
        rpil, spil, _, _ = robust_standardize(real_pil_a, pil)
        dgen = energy_distance(torch.from_numpy(sg), torch.from_numpy(rg))
        ref = split_half_baseline(rg, 2026 + 71)
        ratio = dgen / max(ref["p90"], ref["median"], 1e-6)
        dg = energy_distance(torch.from_numpy(sgeom), torch.from_numpy(rgeom))
        dp = energy_distance(torch.from_numpy(spil), torch.from_numpy(rpil))
        return {
            "control": name,
            "generic_core_distance": float(dgen),
            "generic_core_ratio_to_real_p90": float(ratio),
            "hard_geometry_distance": float(dg),
            "hard_pil_distance": float(dp),
            "parent_pairwise_generic_change": float(np.mean(np.abs(g - parent_g))),
            "parent_pairwise_geometry_change": float(np.mean(np.abs(geom - parent_geom))),
            "parent_pairwise_pil_change": float(np.mean(np.abs(pil - parent_pil))),
            "parent_field_mae_gauss": float(torch.mean(torch.abs(arrays - parent))),
            "parent_saturation_fraction": float((arrays.abs() > 2900).float().mean()),
        }

    rows.append(one("BASE", parent, parent_lat, parent_g, parent_geom, parent_pil))
    for control in ["pil_blur", "geometry_flip", "block_shuffle"]:
        manifest = control_root / control / "synthetic_manifest.csv"
        arrays, lat, sman = load_synthetic(manifest)
        g = generic_descriptor(arrays)
        geom = hard_geometry_descriptor(arrays, lat)
        pil = hard_pil_descriptor(arrays)
        names.append(control)
        rows.append(one(control, arrays, lat, g, geom, pil))
        pd.DataFrame(g, columns=gen_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / f"{control}_generic.csv", index=False)
        pd.DataFrame(geom, columns=geom_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / f"{control}_geometry.csv", index=False)
        pd.DataFrame(pil, columns=pil_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / f"{control}_pil.csv", index=False)
    result = {
        "purpose": "counterfactual generated-field destruction controls; no forecasting outputs",
        "controls": rows,
        "base_reference": {
            "generic_core_distance": primary_metrics["generic_standardized_energy_distance"],
            "hard_geometry_distance": primary_metrics["hard_geometry_standardized_energy_distance"],
            "hard_pil_distance": primary_metrics["hard_pil_standardized_energy_distance"],
        },
        "interpretation": "A control is useful only if it changes its targeted descriptor while preserving nuisance descriptors sufficiently; these controls are diagnostic and are not downstream evidence.",
    }
    (out / "destruction_controls.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(out / "destruction_control_summary.csv", index=False)

    # A compact figure for the report. The distance panels use their own scale.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7))
    x = np.arange(len(table))
    colors = ["#1769aa", "#d97706", "#18864b", "#7c3aed"]
    panels = [
        ("generic core / real p90", "generic_core_ratio_to_real_p90"),
        ("hard geometry distance", "hard_geometry_distance"),
        ("hard PIL distance", "hard_pil_distance"),
    ]
    for ax, (title, col) in zip(axes, panels):
        bars = ax.bar(x, table[col], color=colors, width=0.62)
        ax.set_title(title, loc="left", fontweight="bold", color="#17202a")
        ax.set_xticks(x, table.control.tolist(), rotation=30, ha="right")
        ax.grid(axis="y", color="#d9e2ec", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, table[col]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(table[col]) * 0.025, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("Executed destruction controls: changing structure without forecasts", x=0.04, ha="left", fontsize=15, fontweight="bold", color="#17202a")
    fig.tight_layout()
    report_dir = Path(os.environ.get("IRIS_REPORT_DIR", str(Path(__file__).resolve().parents[1])))
    report_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / "figures" / "destruction_controls.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(report_dir / "figures" / "destruction_controls.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
