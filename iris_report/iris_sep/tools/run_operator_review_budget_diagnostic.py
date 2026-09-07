"""Development-only operator review-budget diagnostic for IRIS-SEP.

Consumes immutable availability-fallback predictions. No model is retrained and
no threshold is tuned. The primary 5% review budget is frozen in
config/operator_review_budget_policy_v1.json before any locked-test access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

POLICY = "config/operator_review_budget_policy_v1.json"
STATE_COLUMNS = {
    "FULL": "p_FULL",
    "NO_XRS": "p_NO_XRS",
    "NO_PROTON": "p_NO_PROTON",
    "NO_XRS_OR_PROTON": "p_NO_XRS_OR_PROTON",
}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def budget_metrics(labels, probabilities, issue_times, fraction: float) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    t = pd.to_datetime(pd.Series(issue_times), utc=True, errors="raise")
    if not (len(y) == len(p) == len(t)) or len(y) == 0:
        raise ValueError("review-budget arrays must be non-empty and aligned")
    if not np.isfinite(p).all():
        raise ValueError("review-budget probabilities must be finite")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("review fraction must be in (0, 1]")

    n = len(y)
    positives = int(y.sum())
    k = int(math.ceil(float(fraction) * n))
    order = np.lexsort((t.astype("int64").to_numpy(), -p))
    selected = order[:k]
    captured = int(y[selected].sum())
    prevalence = float(positives / n) if n else float("nan")
    precision = float(captured / k)
    expected_random = float(positives * k / n)
    enrichment = None if prevalence == 0 else float(precision / prevalence)
    capture = None if positives == 0 else float(captured / positives)

    cutoff = float(p[selected[-1]])
    above = p > cutoff
    tied = p == cutoff
    above_pos = int(y[above].sum())
    tie_rows = int(tied.sum())
    tie_pos = int(y[tied].sum())
    slots_from_tie = int(k - above.sum())
    min_tie_pos = max(0, slots_from_tie - (tie_rows - tie_pos))
    max_tie_pos = min(tie_pos, slots_from_tie)
    min_capture_count = above_pos + min_tie_pos
    max_capture_count = above_pos + max_tie_pos

    return {
        "eligible_rows": int(n),
        "positive_rows": positives,
        "review_fraction_requested": float(fraction),
        "review_rows": k,
        "review_fraction_realized": float(k / n),
        "review_days_per_365_issue_days": float(365.0 * k / n),
        "captured_positive_rows": captured,
        "event_capture_fraction": None if positives == 0 else float(captured / positives),
        "precision_within_review_budget": precision,
        "enrichment_vs_random_review": enrichment,
        "expected_positive_rows_under_random_review": expected_random,
        "cutoff_probability": cutoff,
        "cutoff_tie_rows": tie_rows,
        "cutoff_tie_positive_rows": tie_pos,
        "cutoff_slots_selected": slots_from_tie,
        "tie_robust_event_capture_min": None if positives == 0 else float(min_capture_count / positives),
        "tie_robust_event_capture_max": None if positives == 0 else float(max_capture_count / positives),
    }


def run(result_dir: Path, output: Path) -> dict:
    result_dir = Path(result_dir)
    predictions_path = result_dir / "predictions.csv"
    fallback_summary_path = result_dir / "summary.json"
    if not predictions_path.exists() or not fallback_summary_path.exists():
        raise ValueError("availability fallback predictions and summary are required")

    root = Path(__file__).resolve().parents[1]
    policy_path = root / POLICY
    policy = _load(policy_path)
    fallback = _load(fallback_summary_path)
    predictions = pd.read_csv(predictions_path)
    required = {"issue_time", "role", "label", *STATE_COLUMNS.values()}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")

    score = predictions["role"].astype(str).eq("score").to_numpy()
    if not score.any():
        raise ValueError("score rows required")
    labels = predictions.loc[score, "label"].to_numpy(dtype=int)
    issue_times = predictions.loc[score, "issue_time"]
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("event-bearing score role required")

    fractions = [policy["primary_review_fraction"], *policy["secondary_review_fractions"]]
    fractions = list(dict.fromkeys(float(x) for x in fractions))
    state_permissions = {"FULL": {"permission": "NORMAL", "normal_allowed": True}}
    state_permissions.update(fallback.get("operator_state_decision", {}))

    out = {
        "status": "COMPLETED_OPERATOR_REVIEW_BUDGET_DEVELOPMENT_ONLY",
        "target": fallback.get("target"),
        "policy": POLICY,
        "policy_sha256": digest(policy_path),
        "predictions_sha256": digest(predictions_path),
        "fallback_summary_sha256": digest(fallback_summary_path),
        "locked_test_accessed": False,
        "development_score_already_inspected": True,
        "primary_review_fraction": float(policy["primary_review_fraction"]),
        "score_rows": int(score.sum()),
        "score_positives": positives,
        "score_prevalence": float(positives / score.sum()),
        "states": {},
        "claim_boundary": policy["claim_boundary"],
    }
    for state, column in STATE_COLUMNS.items():
        p = predictions.loc[score, column].to_numpy(dtype=float)
        permission = state_permissions.get(state, {"permission": "UNKNOWN", "normal_allowed": False})
        out["states"][state] = {
            "probability_column": column,
            "permission": permission.get("permission", "UNKNOWN"),
            "normal_allowed": bool(permission.get("normal_allowed", False)),
            "budgets": {str(f): budget_metrics(labels, p, issue_times, f) for f in fractions},
        }

    primary_key = str(float(policy["primary_review_fraction"]))
    out["judge_headline"] = {
        state: {
            "permission": detail["permission"],
            "event_capture_fraction_at_5pct_review": detail["budgets"][primary_key]["event_capture_fraction"],
            "captured_positive_rows": detail["budgets"][primary_key]["captured_positive_rows"],
            "positive_rows": detail["budgets"][primary_key]["positive_rows"],
            "review_days_per_365_issue_days": detail["budgets"][primary_key]["review_days_per_365_issue_days"],
            "enrichment_vs_random_review": detail["budgets"][primary_key]["enrichment_vs_random_review"],
        }
        for state, detail in out["states"].items()
    }

    output = Path(output)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.result_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
