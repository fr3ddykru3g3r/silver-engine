#!/usr/bin/env python3
"""Audit an alternate sampler seed against the fixed primary real reference.

The primary evaluator re-selects real rows by seed and therefore may request
FITS records that are not in the stage-specific cache. This script deliberately
keeps the 248-row real descriptor reference fixed and changes only the
generated sample seed. It is an inference-level robustness check, not a
replacement for an independent training replicate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    pooled_vectors,
    robust_standardize,
    split_half_baseline,
)


def scalar(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-audit", required=True)
    ap.add_argument("--synthetic-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    primary = Path(args.primary_audit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_gen = pd.read_csv(primary / "real_generic.csv")
    real_geom = pd.read_csv(primary / "real_hard_geometry.csv")
    real_pil = pd.read_csv(primary / "real_hard_pil.csv")
    syn, slat, sman = load_synthetic(args.synthetic_manifest)
    syn_gen_a = generic_descriptor(syn)
    syn_geom_a = hard_geometry_descriptor(syn, slat)
    syn_pil_a = hard_pil_descriptor(syn)

    gen_cols = ["log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction", "saturation_fraction"]
    geom_cols = ["hemi_ux", "hemi_uy", "log_sep", "log_strong_flux_density", "has_bipole"]
    pil_cols = ["mean_grad", "rms_grad", "top10_grad", "frac_gt100", "frac_gt250", "frac_gt500", "pil_area_fraction", "has_pil"]
    pd.DataFrame(syn_gen_a, columns=gen_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / "synthetic_generic.csv", index=False)
    pd.DataFrame(syn_geom_a, columns=geom_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / "synthetic_hard_geometry.csv", index=False)
    pd.DataFrame(syn_pil_a, columns=pil_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out / "synthetic_hard_pil.csv", index=False)

    rg = real_gen[gen_cols].to_numpy(np.float32)
    sg = syn_gen_a.astype(np.float32)
    core = [0, 1, 2, 3, 4]
    rgs, sgs, gmed, gscale = robust_standardize(rg[:, core], sg[:, core])
    generic_dist = energy_distance(torch.from_numpy(sgs), torch.from_numpy(rgs))
    split = split_half_baseline(rgs, args.seed + 71)
    ratio = generic_dist / max(split["p90"], split["median"], 1e-6)
    orig_r, orig_s, _, _ = robust_standardize(rg, sg)
    orig_dist = energy_distance(torch.from_numpy(orig_s), torch.from_numpy(orig_r))
    orig_split = split_half_baseline(orig_r, args.seed + 71)
    orig_ratio = orig_dist / max(orig_split["p90"], orig_split["median"], 1e-6)

    rgeom = real_geom[geom_cols].to_numpy(np.float32)
    rpill = real_pil[pil_cols].to_numpy(np.float32)
    g_r, g_s, _, _ = robust_standardize(rgeom, syn_geom_a)
    p_r, p_s, _, _ = robust_standardize(rpill, syn_pil_a)

    syn_flux = syn.abs().mean((1, 2, 3)).numpy()
    real_flux = 50.0 * (np.exp(real_gen.log_mean_abs.to_numpy(float)) - 1.0)
    syn_active = syn_gen_a[:, 3]
    real_active = real_gen.active_fraction.to_numpy(float)
    real_diversity = np.nan
    # The primary audit stores the reference diversity in its JSON. This script
    # loads it below; the synthetic distance is computed using the same pooled
    # representation and compared to that fixed reference.
    primary_metrics = json.loads((primary / "v2_manipulation_metrics.json").read_text())
    real_diversity = float(primary_metrics["real_coarse_pairwise_diversity"])
    syn_diversity = float(torch.pdist(pooled_vectors(syn)).mean())
    flux_ratio = float(np.median(syn_flux) / max(np.median(real_flux), 1e-9))
    active_ratio = float(np.median(syn_active) / max(np.median(real_active), 1e-9))
    diversity_ratio = float(syn_diversity / max(real_diversity, 1e-9))
    sat = float((syn.abs() > 2900).float().mean())
    passed = bool(
        0.50 <= flux_ratio <= 2.00
        and 0.50 <= active_ratio <= 2.00
        and 0.40 <= diversity_ratio <= 2.50
        and sat < 0.01
        and ratio <= 8.0
    )
    report = {
        "evaluator": "alternate sampler seed against fixed primary real descriptor reference",
        "seed": args.seed,
        "real_reference_count": len(real_gen),
        "synthetic_count": len(sg),
        "generic_standardized_energy_distance": float(generic_dist),
        "generic_real_split_half_distance": split,
        "generic_distance_to_real_baseline_ratio": float(ratio),
        "generic_standardized_energy_distance_original_6d": float(orig_dist),
        "generic_real_split_half_distance_original_6d": orig_split,
        "generic_distance_to_real_baseline_ratio_original_6d": float(orig_ratio),
        "generic_distance_features": [gen_cols[i] for i in core],
        "generic_distance_excluded_feature": "saturation_fraction (retained as separate absolute gate)",
        "hard_geometry_standardized_energy_distance": float(energy_distance(torch.from_numpy(g_s), torch.from_numpy(g_r))),
        "hard_pil_standardized_energy_distance": float(energy_distance(torch.from_numpy(p_s), torch.from_numpy(p_r))),
        "real_unsigned_flux_proxy": scalar(real_flux),
        "synthetic_unsigned_flux_proxy": scalar(syn_flux),
        "synthetic_to_real_flux_median_ratio": flux_ratio,
        "real_strong_field_area_fraction": scalar(real_active),
        "synthetic_strong_field_area_fraction": scalar(syn_active),
        "synthetic_to_real_active_area_median_ratio": active_ratio,
        "real_coarse_pairwise_diversity": real_diversity,
        "synthetic_coarse_pairwise_diversity": syn_diversity,
        "synthetic_to_real_diversity_ratio": diversity_ratio,
        "synthetic_saturation_fraction_abs_gt_2900G": sat,
        "generic_fidelity_gate_pass": passed,
        "reference_note": "Real descriptor rows are fixed to the primary seed-2026 audit so the alternate run changes only sampling randomness.",
        "training_replicate": False,
    }
    (out / "v2_manipulation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
