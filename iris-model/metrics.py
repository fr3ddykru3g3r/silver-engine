from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def threshold_metrics(y, p, threshold: float = 0.5):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    z = (p >= threshold).astype(int)
    tp = int(((z == 1) & (y == 1)).sum())
    tn = int(((z == 0) & (y == 0)).sum())
    fp = int(((z == 1) & (y == 0)).sum())
    fn = int(((z == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    tss = tpr - fpr if np.isfinite(tpr) and np.isfinite(fpr) else np.nan
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fn * fp) / den if den else np.nan
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'tss': float(tss), 'hss': float(hss)}


def expected_calibration_error(y, p, bins: int = 10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any(): ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def all_metrics(y, p, threshold: float = 0.5):
    out = threshold_metrics(y, p, threshold)
    try: out['auroc'] = float(roc_auc_score(y, p))
    except Exception: out['auroc'] = np.nan
    out['brier'] = float(brier_score_loss(y, p))
    out['ece10'] = expected_calibration_error(y, p, 10)
    return out


def region_bootstrap(frame: pd.DataFrame, n_boot: int = 1000, seed: int = 2026, threshold: float = 0.5):
    """Cluster bootstrap by connected physical-region group, not by image row."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted(frame.region_group_id.unique()))
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        pieces = []
        for j, g in enumerate(draw):
            x = frame[frame.region_group_id.eq(g)].copy()
            x['_boot_group'] = j
            pieces.append(x)
        z = pd.concat(pieces, ignore_index=True)
        vals.append(all_metrics(z.y.values, z.p.values, threshold))
    keys = ['tss', 'hss', 'auroc', 'brier', 'ece10']
    return {k: {'median': float(np.nanmedian([v[k] for v in vals])),
                'lo95': float(np.nanpercentile([v[k] for v in vals], 2.5)),
                'hi95': float(np.nanpercentile([v[k] for v in vals], 97.5))} for k in keys}
