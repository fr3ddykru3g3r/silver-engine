from __future__ import annotations

"""Corrected v2 generator audit.

This wrapper preserves the original independent evaluator outputs but repairs one
numerical degeneracy in the *generic* multivariate fidelity gate: saturation
fraction is almost identically zero in the real positive training set, so robust
standardisation gives it an approximately 1e-6 scale. Tiny, scientifically
acceptable synthetic saturation then dominates Euclidean/energy distance by
orders of magnitude. Saturation already has an explicit absolute gate, so it is
excluded from the multivariate generic distance and retained as that separate
absolute criterion.

No forecasting labels/metrics are accessed here. The correction is applied
before coefficient selection and is therefore outcome-blind with respect to the
downstream forecasting experiment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_generator_v2 import energy_distance, robust_standardize, split_half_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--synthetic-manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--real-per-group', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    base = Path(__file__).with_name('evaluate_generator_v2.py')
    cmd = [
        sys.executable, str(base),
        '--evidence-dir', args.evidence_dir,
        '--cache-dir', args.cache_dir,
        '--synthetic-manifest', args.synthetic_manifest,
        '--out-dir', args.out_dir,
        '--real-per-group', str(args.real_per_group),
        '--seed', str(args.seed),
    ]
    subprocess.run(cmd, check=True)

    out = Path(args.out_dir)
    report_path = out / 'v2_manipulation_metrics.json'
    report = json.loads(report_path.read_text())

    r = pd.read_csv(out / 'real_generic.csv')
    s = pd.read_csv(out / 'synthetic_generic.csv')
    core = ['log_mean_abs', 'log_p90_abs', 'log_p99_abs', 'active_fraction', 'strong_fraction']
    ra = r[core].to_numpy(np.float32)
    sa = s[core].to_numpy(np.float32)
    rs, ss, med, scale = robust_standardize(ra, sa)
    dist = energy_distance(torch.from_numpy(ss), torch.from_numpy(rs))
    rr = split_half_baseline(rs, args.seed + 71)
    ref = max(rr['p90'], rr['median'], 1e-6)
    ratio = dist / ref

    flux_ratio = float(report['synthetic_to_real_flux_median_ratio'])
    active_ratio = float(report['synthetic_to_real_active_area_median_ratio'])
    diversity_ratio = float(report['synthetic_to_real_diversity_ratio'])
    sat = float(report['synthetic_saturation_fraction_abs_gt_2900G'])
    passed = bool(
        0.50 <= flux_ratio <= 2.00 and
        0.50 <= active_ratio <= 2.00 and
        0.40 <= diversity_ratio <= 2.50 and
        sat < 0.01 and
        ratio <= 8.0
    )

    report.update({
        'evaluator': 'v2 independent hard thresholds; corrected generic-core distance',
        'generic_standardized_energy_distance_original_6d': report['generic_standardized_energy_distance'],
        'generic_real_split_half_distance_original_6d': report['generic_real_split_half_distance'],
        'generic_distance_to_real_baseline_ratio_original_6d': report['generic_distance_to_real_baseline_ratio'],
        'generic_standardized_energy_distance': dist,
        'generic_real_split_half_distance': rr,
        'generic_distance_to_real_baseline_ratio': ratio,
        'generic_fidelity_gate_pass': passed,
        'generic_distance_features': core,
        'generic_distance_excluded_feature': 'saturation_fraction (retained as separate absolute gate)',
        'generic_distance_correction_reason': 'real saturation has near-zero dispersion, causing pathological robust standardisation; absolute saturation gate remains unchanged',
    })
    report['robust_scaling']['generic_core_median'] = med.tolist()
    report['robust_scaling']['generic_core_scale'] = scale.tolist()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + '\n')
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
