from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import all_metrics


SCIENCE_TO_IMPLEMENTATION = {
    "R": "real",
    "Rw": "real_weighted",
    "D": "duplicate",
    "L0": "base",
    "L2": "hj",
    "L3": "hj_pil",
}
SCIENCE_ARMS = tuple(SCIENCE_TO_IMPLEMENTATION)
METRICS = [
    "tss",
    "hss",
    "recall",
    "fpr",
    "precision",
    "auroc",
    "auprc",
    "brier",
    "bss",
    "ece10",
]


def _find(root: Path, implementation: str, filename: str) -> Path:
    hits = sorted(root.rglob(f"{implementation}/{filename}"))
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one {implementation}/{filename}, found {len(hits)}: {hits}"
        )
    return hits[0]


def load(root: Path, implementation: str):
    metrics_path = _find(root, implementation, "metrics.json")
    predictions_path = metrics_path.parent / "test_predictions.csv"
    if not predictions_path.is_file():
        raise RuntimeError(f"missing test predictions for {implementation}: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    metrics = json.loads(metrics_path.read_text())
    threshold = float(metrics["validation_threshold"])
    return predictions, threshold, metrics


def paired(a, b, ta, tb, nboot, seed):
    required = {"sample_id", "region_group_id", "y"}
    if not required.issubset(a.columns) or not required.issubset(b.columns):
        raise RuntimeError("test predictions must contain sample_id, region_group_id, and y")
    if a.sample_id.duplicated().any() or b.sample_id.duplicated().any():
        raise RuntimeError("test prediction sample IDs must be unique within each arm")
    z = a.merge(
        b,
        on=["sample_id", "region_group_id", "y"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(z) != len(a) or len(z) != len(b):
        raise RuntimeError("test prediction sample identities do not match")
    groups = np.asarray(sorted(z.region_group_id.unique()))
    rng = np.random.default_rng(seed)
    values = {key: [] for key in METRICS}
    point_a = all_metrics(z.y, z.p_a, ta)
    point_b = all_metrics(z.y, z.p_b, tb)
    for _ in range(nboot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        pieces = [z[z.region_group_id.eq(group)] for group in draw]
        sample = pd.concat(pieces, ignore_index=True)
        metrics_a = all_metrics(sample.y, sample.p_a, ta)
        metrics_b = all_metrics(sample.y, sample.p_b, tb)
        for key in METRICS:
            values[key].append(metrics_a[key] - metrics_b[key])
    result = {}
    for key, raw in values.items():
        finite = np.asarray(raw, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            result[key] = {
                "point_delta": float(point_a[key] - point_b[key]),
                "median_delta": None,
                "lo95": None,
                "hi95": None,
                "p_two_sided": None,
            }
            continue
        result[key] = {
            "point_delta": float(point_a[key] - point_b[key]),
            "median_delta": float(np.median(finite)),
            "lo95": float(np.percentile(finite, 2.5)),
            "hi95": float(np.percentile(finite, 97.5)),
            "p_two_sided": float(
                min(1.0, 2 * min(np.mean(finite <= 0), np.mean(finite >= 0)))
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")

    data = {}
    summary = {}
    rows = []
    for science_arm, implementation in SCIENCE_TO_IMPLEMENTATION.items():
        predictions, threshold, metrics = load(root, implementation)
        data[science_arm] = (predictions, threshold)
        summary[science_arm] = {
            "implementation_arm": implementation,
            "threshold": threshold,
            "test": metrics["test"],
            "test_region_bootstrap": metrics["test_region_bootstrap"],
            "test_items": metrics.get("test_items"),
            "test_groups": metrics.get("test_groups"),
            "class_weighting": metrics.get("class_weighting"),
            "added_positive_rows": metrics.get("added_positive_rows"),
        }
        rows.append(
            {
                "science_arm": science_arm,
                "implementation_arm": implementation,
                "threshold": threshold,
                "class_weighting": metrics.get("class_weighting"),
                "added_positive_rows": metrics.get("added_positive_rows"),
                "test_items": metrics.get("test_items"),
                "test_groups": metrics.get("test_groups"),
                **{f"test_{key}": metrics["test"].get(key) for key in METRICS},
            }
        )

    additions = [rows[index]["added_positive_rows"] for index in (2, 3, 4, 5)]
    if any(value is None for value in additions) or len(set(additions)) != 1:
        raise RuntimeError(f"D/L0/L2/L3 positive additions are not matched: {additions}")
    if additions[0] <= 0:
        raise RuntimeError("D/L0/L2/L3 must contain a positive matched addition")

    comparisons = {}
    comparison_pairs = (("Rw", "R"), ("D", "R"), ("L0", "D"), ("L2", "D"), ("L3", "D"))
    seed = 260826
    for first, second in comparison_pairs:
        comparisons[f"{first}_minus_{second}"] = paired(
            data[first][0],
            data[second][0],
            data[first][1],
            data[second][1],
            args.bootstrap,
            seed,
        )
        seed += 1

    report = {
        "protocol": "V2 exact primary six-arm matrix",
        "science_arms": list(SCIENCE_ARMS),
        "implementation_mapping": SCIENCE_TO_IMPLEMENTATION,
        "primary_metric": "tss",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "connected region",
        "arms": summary,
        "paired_comparisons": comparisons,
        "interpretation": "delta = first science arm minus comparator; lower is better for FPR, Brier, and ECE10",
    }
    (out / "primary_metrics.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    (out / "primary_paired_metric_bootstrap.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n"
    )
    tss_report = {
        "protocol": report["protocol"],
        "primary_metric": "tss",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": report["bootstrap_unit"],
        "comparisons": {
            name: values["tss"] for name, values in comparisons.items()
        },
    }
    (out / "primary_paired_tss_bootstrap.json").write_text(
        json.dumps(tss_report, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
