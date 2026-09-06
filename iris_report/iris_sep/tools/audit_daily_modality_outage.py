"""Independent recomputation audit for the preregistered daily outage experiment.

This tool deliberately does not import the outage runner or its metric helpers.
It reconstructs scenario exposure from saved block timestamps and recomputes
threshold/probability metrics directly from the immutable predictions CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPECTED_MODALITIES = ("XRS", "PROTON", "XRS_AND_PROTON")
EXPECTED_DURATIONS = (24, 72, 168)
EXPECTED_ARMS = ("MASK_AWARE_NO_FILL", "TRAIN_FIT_MEDIAN", "CAUSAL_FORWARD_FILL")
EXPECTED_POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
MATCHED_PODS = (0.6, 0.7, 0.8, 0.9)
TOL = 1e-10


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value):
    if isinstance(value, dict):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, object]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    tss = pod - fpr
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "POD": pod,
        "FPR": fpr,
        "FAR": far,
        "TSS": tss,
        "HSS": hss,
    }


def ece(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    value = 0.0
    for i in range(10):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        if np.any(mask):
            value += float(np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask])))
    return float(value)


def minimum_far_at_pod(y: np.ndarray, p: np.ndarray, target_pod: float):
    best = None
    for threshold in np.unique(np.r_[0.0, p, 1.0]):
        metrics = threshold_metrics(y, p, float(threshold))
        if metrics["POD"] >= target_pod and np.isfinite(metrics["FAR"]):
            row = (metrics["FAR"], -float(threshold), metrics)
            if best is None or row[:2] < best[:2]:
                best = row
    if best is None:
        return None
    return {
        "target_POD": float(target_pod),
        "achieved_POD": float(best[2]["POD"]),
        "FAR": float(best[0]),
        "threshold": float(-best[1]),
    }


def score(y: np.ndarray, p: np.ndarray, threshold: float, prevalence: float) -> dict[str, object] | None:
    if len(y) == 0:
        return None
    out = threshold_metrics(y, p, threshold)
    brier = float(np.mean((p - y) ** 2))
    reference_brier = float(np.mean((prevalence - y) ** 2))
    out.update(
        {
            "BRIER": brier,
            "BRIER_SKILL": float(1 - brier / reference_brier) if reference_brier > 0 else None,
            "ECE": ece(y, p),
            "AUPRC": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
            "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "matched_detection": (
                {str(pod): minimum_far_at_pod(y, p, pod) for pod in MATCHED_PODS}
                if len(np.unique(y)) == 2
                else None
            ),
            "rows": int(len(y)),
            "positives": int(np.sum(y)),
        }
    )
    return finite_or_none(out)


def close_enough(a, b, path: str, mismatches: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                mismatches.append(f"{path}.{key}: key missing")
            else:
                close_enough(a[key], b[key], f"{path}.{key}", mismatches)
        return
    if a is None or b is None:
        if a is not None or b is not None:
            mismatches.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        if int(a) != int(b):
            mismatches.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (float, int, np.floating, np.integer)) and isinstance(b, (float, int, np.floating, np.integer)):
        if not np.isclose(float(a), float(b), rtol=0.0, atol=TOL):
            mismatches.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        mismatches.append(f"{path}: {a!r} != {b!r}")


def subset_for_compare(metrics: dict[str, object] | None) -> dict[str, object] | None:
    if metrics is None:
        return None
    keys = (
        "TP", "FN", "FP", "TN", "POD", "FPR", "FAR", "TSS", "HSS",
        "BRIER", "BRIER_SKILL", "ECE", "AUPRC", "AUROC", "matched_detection",
    )
    return {k: metrics.get(k) for k in keys if k in metrics}


def affected_mask(frame: pd.DataFrame, scenario: dict[str, object]) -> np.ndarray:
    issue = frame["issue_time"]
    score_role = frame["role"].astype(str).to_numpy() == "score"
    mask = np.zeros(len(frame), dtype=bool)
    for block in scenario["blocks"]:
        start = pd.Timestamp(block["issue_start"])
        end = pd.Timestamp(block["issue_end_exclusive"])
        mask |= score_role & (issue >= start).to_numpy() & (issue < end).to_numpy()
    return mask


def audit(predictions: Path, summary_path: Path, output: Path) -> dict[str, object]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions)
    required = {"issue_time", "role", "unit_id", "label", "reference"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions missing columns: {sorted(required - set(frame.columns))}")
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True, errors="raise")
    y = frame["label"].to_numpy(dtype=int)
    reference = frame["reference"].to_numpy(dtype=float)
    score_role = frame["role"].astype(str).to_numpy() == "score"
    fit_role = frame["role"].astype(str).to_numpy() == "fit"
    if not score_role.any() or not fit_role.any():
        raise ValueError("fit/score roles must be nonempty")
    prevalence = float(np.mean(y[fit_role]))

    expected_scenarios = {
        f"{modality}_{duration}H"
        for modality in EXPECTED_MODALITIES
        for duration in EXPECTED_DURATIONS
    }
    actual_scenarios = set(summary.get("scenarios", {}))
    if actual_scenarios != expected_scenarios:
        raise ValueError(f"scenario registry mismatch: {sorted(actual_scenarios ^ expected_scenarios)}")
    thresholds = summary["model"]["thresholds"]
    if set(thresholds) != set(EXPECTED_POLICIES):
        raise ValueError("threshold policy registry mismatch")

    mismatches: list[str] = []
    audited = {}
    for scenario_key in sorted(expected_scenarios):
        scenario = summary["scenarios"][scenario_key]
        mask = affected_mask(frame, scenario)
        if int(mask.sum()) != int(scenario["affected_score_rows"]):
            mismatches.append(
                f"{scenario_key}.affected_score_rows: recomputed {int(mask.sum())} != stored {scenario['affected_score_rows']}"
            )
        if int(y[mask].sum()) != int(scenario["affected_score_positives"]):
            mismatches.append(
                f"{scenario_key}.affected_score_positives: recomputed {int(y[mask].sum())} != stored {scenario['affected_score_positives']}"
            )
        positive_units = len(np.unique(frame.loc[mask & (y == 1), "unit_id"].astype(str)))
        if int(positive_units) != int(scenario["affected_positive_event_units"]):
            mismatches.append(
                f"{scenario_key}.affected_positive_event_units: recomputed {positive_units} != stored {scenario['affected_positive_event_units']}"
            )

        scenario_audit = {"affected_rows": int(mask.sum()), "affected_positives": int(y[mask].sum()), "arms": {}}
        if set(scenario["arms"]) != set(EXPECTED_ARMS):
            mismatches.append(f"{scenario_key}.arms: registry mismatch")
        for arm in EXPECTED_ARMS:
            column = f"p_{scenario_key}_{arm}"
            if column not in frame.columns:
                mismatches.append(f"{scenario_key}.{arm}: missing prediction column")
                continue
            p = frame[column].to_numpy(dtype=float)
            if not np.isfinite(p).all():
                mismatches.append(f"{scenario_key}.{arm}: nonfinite saved probability")
                continue
            arm_audit = {"policies": {}}
            for policy in EXPECTED_POLICIES:
                threshold = float(thresholds[policy])
                whole = score(y[score_role], p[score_role], threshold, prevalence)
                affected = score(y[mask], p[mask], threshold, prevalence)
                reference_whole = score(y[score_role], reference[score_role], threshold, prevalence)
                reference_affected = score(y[mask], reference[mask], threshold, prevalence)
                stored = scenario["arms"][arm]["policies"][policy]
                close_enough(
                    subset_for_compare(whole),
                    subset_for_compare(stored["whole_score_candidate"]),
                    f"{scenario_key}.{arm}.{policy}.whole_candidate",
                    mismatches,
                )
                close_enough(
                    subset_for_compare(reference_whole),
                    subset_for_compare(stored["whole_score_reference"]),
                    f"{scenario_key}.{arm}.{policy}.whole_reference",
                    mismatches,
                )
                close_enough(
                    subset_for_compare(affected),
                    subset_for_compare(stored["affected_rows_candidate"]),
                    f"{scenario_key}.{arm}.{policy}.affected_candidate",
                    mismatches,
                )
                close_enough(
                    subset_for_compare(reference_affected),
                    subset_for_compare(stored["affected_rows_reference"]),
                    f"{scenario_key}.{arm}.{policy}.affected_reference",
                    mismatches,
                )
                arm_audit["policies"][policy] = {
                    "whole_score": whole,
                    "affected_rows": affected,
                }
            scenario_audit["arms"][arm] = arm_audit
        audited[scenario_key] = scenario_audit

    result = {
        "status": "PASSED" if not mismatches else "FAILED",
        "independent_implementation": True,
        "imports_outage_runner": False,
        "predictions_sha256": digest(predictions),
        "summary_sha256": digest(summary_path),
        "scenario_count": len(expected_scenarios),
        "arm_count": len(expected_scenarios) * len(EXPECTED_ARMS),
        "policy_evaluations": len(expected_scenarios) * len(EXPECTED_ARMS) * len(EXPECTED_POLICIES),
        "mismatches": mismatches,
        "audited": audited,
    }
    Path(output).write_text(json.dumps(finite_or_none(result), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if mismatches:
        raise ValueError(f"independent outage audit failed with {len(mismatches)} mismatches")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.predictions, args.summary, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "audited"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
