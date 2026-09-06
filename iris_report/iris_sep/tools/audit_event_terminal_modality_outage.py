"""Independent recomputation audit for event-terminal outage evidence.

This module intentionally does not import the event-terminal runner.
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity_sha(times) -> str:
    ordered = sorted(pd.Timestamp(t).isoformat() for t in times)
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def threshold_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= float(threshold)
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred)); tn = int(np.sum((y == 0) & ~pred))
    pod = None if tp + fn == 0 else tp / (tp + fn)
    fpr = None if fp + tn == 0 else fp / (fp + tn)
    far = None if tp + fp == 0 else fp / (tp + fp)
    tss = None if pod is None or fpr is None else pod - fpr
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "POD": pod, "FPR": fpr, "FAR": far, "TSS": tss}


def ece(y, p):
    edges = np.linspace(0.0, 1.0, 11); value = 0.0
    for i in range(10):
        member = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        if np.any(member):
            value += float(np.mean(member) * abs(np.mean(p[member]) - np.mean(y[member])))
    return float(value)


def metrics(y, p, mask, threshold, prevalence):
    mask = np.asarray(mask, dtype=bool)
    yy = np.asarray(y, dtype=int)[mask]; pp = np.asarray(p, dtype=float)[mask]
    if len(yy) == 0:
        return None
    out = threshold_metrics(yy, pp, threshold)
    out["BRIER"] = float(np.mean((pp - yy) ** 2))
    out["ECE"] = ece(yy, pp)
    if len(np.unique(yy)) == 2:
        baseline = float(np.mean((yy - float(prevalence)) ** 2))
        out["BRIER_SKILL"] = None if baseline == 0 else float(1.0 - out["BRIER"] / baseline)
        out["AUPRC"] = float(average_precision_score(yy, pp))
        out["AUROC"] = float(roc_auc_score(yy, pp))
    else:
        out["BRIER_SKILL"] = None; out["AUPRC"] = None; out["AUROC"] = None
    return out


def same(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def compare_metric_set(expected, actual, prefix, mismatches):
    for key in ("TP", "FN", "FP", "TN", "POD", "FPR", "FAR", "TSS", "BRIER", "ECE", "BRIER_SKILL", "AUPRC", "AUROC"):
        if key in expected and not same(expected.get(key), actual.get(key)):
            mismatches.append({"path": f"{prefix}.{key}", "saved": expected.get(key), "recomputed": actual.get(key)})


def audit(predictions: Path, summary_path: Path, output: Path):
    frame = pd.read_csv(predictions)
    summary = json.loads(Path(summary_path).read_text())
    y = frame["label"].to_numpy(dtype=int)
    reference = frame["reference"].to_numpy(dtype=float)
    score = frame["role"].astype(str).to_numpy() == "score"
    prevalence = float(summary["model"]["fit_prevalence"])
    mismatches = []
    evaluated = 0

    if int(score.sum()) != int(summary["score_rows"]) or int(y[score].sum()) != int(summary["score_positives"]):
        mismatches.append({"path": "score_support", "saved": [summary["score_rows"], summary["score_positives"]], "recomputed": [int(score.sum()), int(y[score].sum())]})

    for scenario, spec in summary["scenarios"].items():
        event = frame[f"event_{scenario}"].to_numpy(dtype=int) == 1
        quiet = frame[f"quiet_{scenario}"].to_numpy(dtype=int) == 1
        affected = event | quiet
        issue = pd.to_datetime(frame["issue_time"], utc=True)
        if int(event.sum()) != int(spec["eligible_positive_terminal_rows"]):
            mismatches.append({"path": f"{scenario}.event_count", "saved": spec["eligible_positive_terminal_rows"], "recomputed": int(event.sum())})
        if int(quiet.sum()) != int(spec["quiet_control_terminal_rows"]):
            mismatches.append({"path": f"{scenario}.quiet_count", "saved": spec["quiet_control_terminal_rows"], "recomputed": int(quiet.sum())})
        if identity_sha(issue[event]) != spec["positive_terminal_identity_sha256"]:
            mismatches.append({"path": f"{scenario}.event_identity_sha256"})
        if identity_sha(issue[quiet]) != spec["quiet_control_identity_sha256"]:
            mismatches.append({"path": f"{scenario}.quiet_identity_sha256"})
        if np.any(y[event] != 1) or np.any(y[quiet] != 0):
            mismatches.append({"path": f"{scenario}.label_stratification"})

        for arm, result in spec["arms"].items():
            candidate = frame[f"p_{scenario}_{arm}"].to_numpy(dtype=float)
            event_ref = reference[event]; event_p = candidate[event]
            checks = {
                "event_probability_mean_reference": float(np.mean(event_ref)),
                "event_probability_mean_candidate": float(np.mean(event_p)),
                "event_probability_absolute_drift": float(np.mean(np.abs(event_p - event_ref))),
                "event_probability_signed_delta": float(np.mean(event_p - event_ref)),
                "whole_score_probability_absolute_drift": float(np.mean(np.abs(candidate[score] - reference[score]))),
            }
            if quiet.any():
                checks["quiet_probability_absolute_drift"] = float(np.mean(np.abs(candidate[quiet] - reference[quiet])))
            for key, val in checks.items():
                if not same(result.get(key), val):
                    mismatches.append({"path": f"{scenario}.{arm}.{key}", "saved": result.get(key), "recomputed": val})

            for policy, policy_result in result["policies"].items():
                threshold = float(summary["model"]["thresholds"][policy])
                subsets = {
                    "event_rows_reference": (reference, event),
                    "event_rows_candidate": (candidate, event),
                    "quiet_rows_reference": (reference, quiet),
                    "quiet_rows_candidate": (candidate, quiet),
                    "combined_rows_reference": (reference, affected),
                    "combined_rows_candidate": (candidate, affected),
                    "whole_score_reference": (reference, score),
                    "whole_score_candidate": (candidate, score),
                }
                for name, (prob, mask) in subsets.items():
                    actual = metrics(y, prob, mask, threshold, prevalence)
                    expected = policy_result[name]
                    if expected is None or actual is None:
                        if expected is not None or actual is not None:
                            mismatches.append({"path": f"{scenario}.{arm}.{policy}.{name}"})
                    else:
                        compare_metric_set(expected, actual, f"{scenario}.{arm}.{policy}.{name}", mismatches)
                evaluated += 1

    report = {
        "status": "PASSED" if not mismatches else "FAILED",
        "independent_implementation": True,
        "imports_event_terminal_runner": False,
        "predictions_sha256": digest(predictions),
        "summary_sha256": digest(summary_path),
        "scenario_count": len(summary["scenarios"]),
        "arm_threshold_evaluations": evaluated,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "audited_fields": ["identity hashes", "support counts", "event/quiet probability drift", "TP", "FN", "FP", "TN", "POD", "FPR", "FAR", "TSS", "Brier", "Brier skill", "ECE", "AUPRC", "AUROC"],
    }
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if mismatches:
        raise SystemExit(f"event-terminal audit failed with {len(mismatches)} mismatches")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.predictions, args.summary, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
