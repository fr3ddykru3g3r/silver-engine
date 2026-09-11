"""Compare frozen Phase II predictions with the released SEPNET-PRISM architecture.

Both inputs must already exist as immutable development artifacts.  This module
only aligns rows, reuses each model's threshold-role decision threshold, and
computes paired physical-unit uncertainty.  It does not retrain or retune either
model and cannot authorize a universal SOTA claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs

COMPARATOR = "SEPNET_PRISM_RELEASED_ARCHITECTURE"
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
ROLES = ("score", "monitor")
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_REPLICATES = 10000


def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return [finite(v) for v in value.tolist()]
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalized_times(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise")).as_unit("ns")


def evaluate(y, p, threshold, roles, role, prevalence):
    mask = roles == role
    return {
        **v1.threshold_metrics(y[mask], p[mask], threshold),
        **v1.probability_metrics(y[mask], p[mask], prevalence),
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
    }


def run(phase2_root: Path, released_root: Path, output: Path):
    output = Path(output)
    if output.exists(): raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    phase2_summary = json.loads(Path(phase2_root, "summary.json").read_text())
    released_summary = json.loads(Path(released_root, "summary.json").read_text())
    phase2 = pd.read_csv(Path(phase2_root, "predictions.csv"))
    released = pd.read_csv(Path(released_root, "predictions.csv"))

    selected = phase2_summary["architecture_selection"]["selected"]
    if selected not in phase2.columns:
        raise ValueError("selected Phase II candidate missing from predictions")
    if COMPARATOR not in released.columns:
        raise ValueError("released comparator missing from predictions")
    if len(phase2) != len(released):
        raise ValueError("row count mismatch")
    if not np.array_equal(phase2["label"].to_numpy(dtype=int), released["label"].to_numpy(dtype=int)):
        raise ValueError("label mismatch")
    if not np.array_equal(phase2["role"].astype(str).to_numpy(), released["role"].astype(str).to_numpy()):
        raise ValueError("role mismatch")
    if not np.array_equal(phase2["unit_id"].fillna("").astype(str).to_numpy(), released["unit_id"].fillna("").astype(str).to_numpy()):
        raise ValueError("physical-unit mismatch")
    if not normalized_times(phase2["issue_time"]).equals(normalized_times(released["issue_time"])):
        raise ValueError("issue-time mismatch")

    y = phase2["label"].to_numpy(dtype=int)
    roles = phase2["role"].astype(str).to_numpy()
    units = phase2["unit_id"].fillna("").astype(str).to_numpy()
    candidate_p = phase2[selected].to_numpy(dtype=float)
    comparator_p = released[COMPARATOR].to_numpy(dtype=float)
    fit = roles == "fit"
    prevalence = float(np.mean(y[fit]))

    candidate_thresholds = phase2_summary["models"][selected]["thresholds"]
    comparator_thresholds = released_summary["models"][COMPARATOR]["thresholds"]

    summary = {
        "status": "COMPLETED_PHASE2_VS_RELEASED_ARCHITECTURE_DEVELOPMENT_COMPARISON",
        "scope": "DEVELOPMENT_ONLY_SAME_COHORT",
        "locked_test_accessed": False,
        "protected_post_2025_outcomes_accessed": False,
        "candidate": selected,
        "comparator": COMPARATOR,
        "comparator_description": "released architecture / fixed recipe under IRIS chronology; not exact paper random-IID reproduction",
        "models": {},
        "paired_comparisons": {},
    }
    for name, p, thresholds in (
        (selected, candidate_p, candidate_thresholds),
        (COMPARATOR, comparator_p, comparator_thresholds),
    ):
        summary["models"][name] = {"thresholds": thresholds, "policies": {}}
        for policy in POLICIES:
            threshold = float(thresholds[policy])
            summary["models"][name]["policies"][policy] = {
                role: evaluate(y, p, threshold, roles, role, prevalence) for role in ROLES
            }

    for policy in POLICIES:
        for role in ROLES:
            key = f"{selected}_minus_{COMPARATOR}_{policy}_{role}"
            summary["paired_comparisons"][key] = cs.bootstrap_difference(
                y,
                candidate_p, float(candidate_thresholds[policy]),
                comparator_p, float(comparator_thresholds[policy]),
                units, roles, role,
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
            )

    c = summary["models"][selected]["policies"]["MAX_TSS"]["score"]
    b = summary["models"][COMPARATOR]["policies"]["MAX_TSS"]["score"]
    paired = summary["paired_comparisons"][f"{selected}_minus_{COMPARATOR}_MAX_TSS_score"]
    checks = {
        "TSS_point_gt_comparator": bool(float(c["TSS"]) > float(b["TSS"])),
        "paired_TSS_ci95_lower_gt_zero": bool(float(paired["ci_lower_95"]) > 0.0),
        "HSS_gt_comparator": bool(float(c["HSS"]) > float(b["HSS"])),
        "FAR_lt_comparator": bool(float(c["FAR"]) < float(b["FAR"])),
    }
    summary["same_cohort_released_architecture_gate"] = {
        "policy": "MAX_TSS",
        "role": "score",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "point_TSS_delta": float(c["TSS"] - b["TSS"]),
        "bootstrap": paired,
    }

    aligned = pd.DataFrame({
        "issue_time": phase2["issue_time"],
        "role": roles,
        "unit_id": units,
        "label": y,
        selected: candidate_p,
        COMPARATOR: comparator_p,
    })
    aligned.to_csv(output / "aligned_predictions.csv", index=False, float_format="%.17g")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": summary["status"],
        "candidate": selected,
        "comparator": COMPARATOR,
        "locked_test_accessed": False,
        "protected_post_2025_outcomes_accessed": False,
        "models_retrained_or_retuned_by_comparison": False,
        "same_cohort_required_and_verified": True,
        "universal_state_of_the_art_claim_authorized": False,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--released-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.phase2_root, args.released_root, args.output)


if __name__ == "__main__":
    main()
