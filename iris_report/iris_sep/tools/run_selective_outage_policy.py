"""Apply preregistered operator-handling policies to a verified daily-outage artifact.

No model is trained here. Probabilities, thresholds, outage placements and
recovery arms are inherited immutably from the daily outage experiment. The
only intervention is whether an outage-affected forecast is exposed normally,
withheld (ABSTAIN), or exposed with an explicit DEGRADED status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import audit_daily_modality_outage as audit

PREREG = "config/selective_outage_policy_preregistration_2026-09-06.json"
HANDLING_POLICIES = (
    "ALWAYS_EXPOSE_NORMAL",
    "ABSTAIN_ON_DECLARED_OUTAGE",
    "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE",
)


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
    if isinstance(value, np.ndarray):
        return [finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def exposure_mask(score_mask: np.ndarray, affected_mask: np.ndarray, handling_policy: str) -> np.ndarray:
    score_mask = np.asarray(score_mask, dtype=bool)
    affected_mask = np.asarray(affected_mask, dtype=bool)
    if handling_policy == "ABSTAIN_ON_DECLARED_OUTAGE":
        return score_mask & ~affected_mask
    if handling_policy in ("ALWAYS_EXPOSE_NORMAL", "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE"):
        return score_mask.copy()
    raise ValueError(f"unknown handling policy: {handling_policy}")


def full_cohort_accounting(
    y: np.ndarray,
    p: np.ndarray,
    score_mask: np.ndarray,
    affected_mask: np.ndarray,
    threshold: float,
    handling_policy: str,
) -> dict[str, object]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    score_mask = np.asarray(score_mask, dtype=bool)
    affected_mask = np.asarray(affected_mask, dtype=bool) & score_mask
    exposed = exposure_mask(score_mask, affected_mask, handling_policy)
    abstained = score_mask & ~exposed
    total_positives = int(np.sum(y[score_mask] == 1))
    covered_positives = int(np.sum((y == 1) & exposed))
    abstained_positives = int(np.sum((y == 1) & abstained))
    covered_negatives = int(np.sum((y == 0) & exposed))
    abstained_negatives = int(np.sum((y == 0) & abstained))
    alert = p >= float(threshold)
    true_alerts = int(np.sum(exposed & (y == 1) & alert))
    false_alerts = int(np.sum(exposed & (y == 0) & alert))
    covered_fn = int(np.sum(exposed & (y == 1) & ~alert))
    missed = covered_fn + abstained_positives
    return {
        "coverage_fraction": float(np.mean(exposed[score_mask])) if np.any(score_mask) else None,
        "covered_rows": int(np.sum(exposed)),
        "abstained_rows": int(np.sum(abstained)),
        "total_positives": total_positives,
        "covered_positives": covered_positives,
        "abstained_positives": abstained_positives,
        "covered_negative_rows": covered_negatives,
        "abstained_negative_rows": abstained_negatives,
        "true_alerts": true_alerts,
        "false_alerts": false_alerts,
        "covered_false_negatives": covered_fn,
        "coverage_adjusted_missed_positives": int(missed),
        "coverage_adjusted_detection_fraction": (
            float(true_alerts / total_positives) if total_positives else None
        ),
        "false_alert_ratio_among_exposed_alerts": (
            float(false_alerts / (true_alerts + false_alerts))
            if true_alerts + false_alerts
            else None
        ),
        "degraded_rows": int(np.sum(affected_mask)) if handling_policy == "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE" else 0,
        "degraded_positives": int(np.sum(affected_mask & (y == 1))) if handling_policy == "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE" else 0,
    }


def probability_metrics_on_exposed(
    y: np.ndarray,
    p: np.ndarray,
    exposed: np.ndarray,
    threshold: float,
    prevalence: float,
):
    exposed = np.asarray(exposed, dtype=bool)
    return audit.score(y[exposed], p[exposed], threshold, prevalence) if exposed.any() else None


def run(
    predictions: Path,
    outage_summary: Path,
    independent_audit: Path,
    output: Path,
) -> dict[str, object]:
    verified = json.loads(Path(independent_audit).read_text(encoding="utf-8"))
    if verified.get("status") != "PASSED":
        raise ValueError("daily outage independent audit must be PASSED")
    if verified.get("predictions_sha256") != digest(predictions):
        raise ValueError("predictions hash does not match independent audit")
    if verified.get("summary_sha256") != digest(outage_summary):
        raise ValueError("summary hash does not match independent audit")

    summary = json.loads(Path(outage_summary).read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions)
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True, errors="raise")
    y = frame["label"].to_numpy(dtype=int)
    roles = frame["role"].astype(str).to_numpy()
    score_mask = roles == "score"
    fit_mask = roles == "fit"
    if not score_mask.any() or not fit_mask.any():
        raise ValueError("fit and score roles must be nonempty")
    prevalence = float(np.mean(y[fit_mask]))
    thresholds = summary["model"]["thresholds"]

    project_root = Path(__file__).resolve().parents[1]
    prereg_path = project_root / PREREG
    if not prereg_path.exists():
        raise ValueError("selective policy preregistration missing")

    result = {
        "status": "COMPLETED_SELECTIVE_OUTAGE_POLICY_DEVELOPMENT_ONLY",
        "target": summary["target"],
        "preregistration": PREREG,
        "preregistration_sha256": digest(prereg_path),
        "daily_outage_summary_sha256": digest(outage_summary),
        "daily_outage_predictions_sha256": digest(predictions),
        "daily_outage_independent_audit_sha256": digest(independent_audit),
        "daily_outage_independent_audit_passed": True,
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_role_already_inspected": True,
        "no_model_training": True,
        "no_probability_transform": True,
        "no_rethresholding": True,
        "scenarios": {},
    }

    evaluation_count = 0
    for scenario_key in sorted(summary["scenarios"]):
        scenario = summary["scenarios"][scenario_key]
        affected = audit.affected_mask(frame, scenario)
        matched = score_mask & ~affected
        scenario_result = {
            "affected_score_rows": int(np.sum(affected)),
            "affected_score_positives": int(np.sum(y[affected])),
            "matched_coverage_rows": int(np.sum(matched)),
            "arms": {},
        }
        for arm in audit.EXPECTED_ARMS:
            column = f"p_{scenario_key}_{arm}"
            if column not in frame.columns:
                raise ValueError(f"missing probability column: {column}")
            p = frame[column].to_numpy(dtype=float)
            if not np.isfinite(p).all():
                raise ValueError(f"nonfinite probability column: {column}")
            arm_result = {"threshold_policies": {}}
            for threshold_policy in audit.EXPECTED_POLICIES:
                threshold = float(thresholds[threshold_policy])
                matched_metrics = audit.score(y[matched], p[matched], threshold, prevalence) if matched.any() else None
                policy_result = {}
                for handling_policy in HANDLING_POLICIES:
                    exposed = exposure_mask(score_mask, affected, handling_policy)
                    policy_result[handling_policy] = {
                        "full_cohort_accounting": full_cohort_accounting(
                            y, p, score_mask, affected, threshold, handling_policy
                        ),
                        "probability_metrics_on_exposed_rows": probability_metrics_on_exposed(
                            y, p, exposed, threshold, prevalence
                        ),
                        "matched_coverage_probability_metrics": matched_metrics,
                        "status_semantics": (
                            "ABSTAIN_ON_AFFECTED_ROWS"
                            if handling_policy == "ABSTAIN_ON_DECLARED_OUTAGE"
                            else "DEGRADED_ON_AFFECTED_ROWS"
                            if handling_policy == "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE"
                            else "NORMAL_ON_ALL_ROWS"
                        ),
                    }
                    evaluation_count += 1
                arm_result["threshold_policies"][threshold_policy] = policy_result
            scenario_result["arms"][arm] = arm_result
        result["scenarios"][scenario_key] = scenario_result

    result["policy_evaluations"] = int(evaluation_count)
    expected = len(audit.EXPECTED_MODALITIES) * len(audit.EXPECTED_DURATIONS) * len(audit.EXPECTED_ARMS) * len(audit.EXPECTED_POLICIES) * len(HANDLING_POLICIES)
    if evaluation_count != expected:
        raise ValueError(f"policy evaluation count {evaluation_count} != preregistered {expected}")
    result["all_preregistered_policy_evaluations_reported"] = True
    result["claim_boundary"] = (
        "Development-only handling-policy diagnostic on already-inspected score identities. "
        "It quantifies coverage and alert tradeoffs but does not establish operator benefit, "
        "prospective causality, locked-test robustness, operational readiness or superiority."
    )

    Path(output).write_text(
        json.dumps(finite_or_none(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outage-summary", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.predictions, args.outage_summary, args.independent_audit, args.output)
    print(json.dumps({
        "status": result["status"],
        "policy_evaluations": result["policy_evaluations"],
        "all_preregistered_policy_evaluations_reported": result["all_preregistered_policy_evaluations_reported"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
