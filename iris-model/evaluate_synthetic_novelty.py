from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_generator_v2 import load_real, load_synthetic, pooled_vectors


EPS = 1e-9
NOVELTY_RATIO_MIN = 0.25
EXACT_DUPLICATE_TOL = 1e-7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--synthetic-manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--real-per-group', type=int, default=2)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    # Training-positive data only. No validation/test manifests or flare outcomes are read.
    real, _, rrec = load_real(args.evidence_dir, args.cache_dir, args.real_per_group, args.seed)
    syn, _, sman = load_synthetic(args.synthetic_manifest)
    rv = pooled_vectors(real).float()
    sv = pooled_vectors(syn).float()

    if len(rv) < 2 or len(sv) < 1:
        raise RuntimeError('Need at least two real samples and one synthetic sample for novelty audit')

    rg = rrec.region_group_id.astype(str).to_numpy()
    # Natural reference scale: for each real magnetogram, distance to the closest
    # positive magnetogram from a DIFFERENT connected active-region group. This
    # avoids counting nearby frames from one region as independent novelty.
    rr = torch.cdist(rv, rv)
    same_group = torch.from_numpy(rg[:, None] == rg[None, :]).to(rr.device)
    rr = rr.masked_fill(same_group, float('inf'))
    real_cross_nn = rr.min(dim=1).values
    finite = torch.isfinite(real_cross_nn)
    if not finite.any():
        raise RuntimeError('No cross-region real nearest-neighbour distances available')
    real_cross_nn = real_cross_nn[finite]

    # Synthetic novelty is measured against ALL real positive training images,
    # including its source. A near-copy therefore receives a near-zero score.
    sr = torch.cdist(sv, rv)
    syn_nn, syn_nn_idx = sr.min(dim=1)

    real_ref_median = float(torch.median(real_cross_nn).item())
    syn_nn_median = float(torch.median(syn_nn).item())
    novelty_ratio = syn_nn_median / max(real_ref_median, EPS)
    exact_duplicate_fraction = float((syn_nn <= EXACT_DUPLICATE_TOL).float().mean().item())

    # Extra audit: where source-region metadata exists, measure distance to the
    # nearest real frame from that source region. This is reported, not tuned on.
    source_nn = []
    if 'source_region_group_id' in sman.columns:
        for i, src in enumerate(sman.source_region_group_id.astype(str)):
            idx = np.where(rg == src)[0]
            if len(idx):
                source_nn.append(float(sr[i, torch.as_tensor(idx, dtype=torch.long)].min().item()))
    source_nn_median = float(np.median(source_nn)) if source_nn else float('nan')

    gate = bool(novelty_ratio >= NOVELTY_RATIO_MIN and exact_duplicate_fraction == 0.0)
    report = {
        'phase': 'TRAIN-ONLY synthetic novelty audit; forecast validation/test not accessed',
        'descriptor': '16x16 pooled asinh-normalized magnetogram vectors',
        'real_reference': 'nearest positive training magnetogram from a different connected region_group_id',
        'synthetic_reference': 'nearest positive real training magnetogram, source included',
        'real_count': int(len(rv)),
        'synthetic_count': int(len(sv)),
        'real_cross_region_nn_median': real_ref_median,
        'synthetic_to_real_nn_median': syn_nn_median,
        'novelty_ratio_to_real_nn_median': float(novelty_ratio),
        'source_region_nn_median': source_nn_median,
        'exact_duplicate_fraction': exact_duplicate_fraction,
        'novelty_gate_definition': {
            'novelty_ratio_min': NOVELTY_RATIO_MIN,
            'exact_duplicate_fraction_required': 0.0,
            'exact_duplicate_tolerance': EXACT_DUPLICATE_TOL,
        },
        'novelty_gate_pass': gate,
        'forecast_outcomes_accessed': False,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
