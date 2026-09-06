"""Preregisterable daily forecast-input modality outage benchmark.

The model-ready SEPNET-PRISM table contains one UTC issue row per day, each
summarizing the preceding 24 hours. Therefore a 24/72/168-hour outage on this
*forecast-input interface* is represented by 1/3/7 consecutive daily issue
cycles. This is deliberately distinct from a raw 5-minute instrument-stream
outage, which requires upstream reaggregation and is not claimed here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.missingness_experiment import (
    recover_causal_forward_fill,
    recover_train_median,
)
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_contiguous_modality_outage_v2 as v2
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer


PREREG = "config/daily_modality_outage_preregistration_2026-09-06.json"
DURATION_HOURS = (24, 72, 168)
DURATION_DAYS = {24: 1, 72: 3, 168: 7}
MODALITIES = ("XRS", "PROTON", "XRS_AND_PROTON")
ARMS = transfer.ARMS
START_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
SEED = 20260906
DAY = pd.Timedelta(days=1)


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


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _contiguous_daily_starts(
    source_times: pd.Series,
    duration_days: int,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> list[pd.Timestamp]:
    if duration_days <= 0:
        raise ValueError("duration_days must be positive")
    times = pd.Series(source_times).reset_index(drop=True)
    times = times[(times >= lower) & (times <= upper)].reset_index(drop=True)
    if len(times) < duration_days:
        raise ValueError("source clock too short for daily outage")
    ns = times.astype("int64").to_numpy(dtype=np.int64)
    day_ns = int(DAY.value)
    starts: list[pd.Timestamp] = []
    for i in range(0, len(times) - duration_days + 1):
        segment = ns[i:i + duration_days]
        if duration_days == 1 or np.all(np.diff(segment) == day_ns):
            start = times.iloc[i]
            end_exclusive = start + pd.Timedelta(days=duration_days)
            if end_exclusive - DAY <= upper:
                starts.append(start)
    return starts


def deterministic_daily_blocks(
    source_times: pd.Series,
    duration_hours: int,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> list[dict]:
    if duration_hours not in DURATION_DAYS:
        raise ValueError("unsupported outage duration")
    duration_days = DURATION_DAYS[duration_hours]
    admissible = _contiguous_daily_starts(source_times, duration_days, lower, upper)
    if len(admissible) < len(START_FRACTIONS):
        raise ValueError(
            f"too few daily source starts for {duration_hours}-hour span: {len(admissible)}"
        )
    positions = [int(round(f * (len(admissible) - 1))) for f in START_FRACTIONS]
    starts = [admissible[p] for p in positions]
    if len(set(starts)) != len(starts):
        raise ValueError("deterministic daily starts collide")
    ordered = sorted(starts)
    width = pd.Timedelta(days=duration_days)
    if any(b < a + width for a, b in zip(ordered, ordered[1:])):
        raise ValueError("daily outage blocks overlap")
    return [
        {
            "issue_start": start,
            "issue_end_exclusive": start + width,
            "duration_hours": int(duration_hours),
            "duration_daily_cycles": int(duration_days),
        }
        for start in starts
    ]


def scenario_holdout(
    observed: np.ndarray,
    score_rows: np.ndarray,
    eligible_times: pd.Series,
    source_times: pd.Series,
    feature_indices: np.ndarray,
    duration_hours: int,
):
    score_rows = np.asarray(score_rows, dtype=bool)
    times = pd.Series(eligible_times).reset_index(drop=True)
    if observed.ndim != 2 or observed.shape[0] != len(times) or len(score_rows) != len(times):
        raise ValueError("outage inputs must align")
    if not score_rows.any():
        raise ValueError("score role empty")
    lower = times[score_rows].min()
    upper = times[score_rows].max()
    blocks = deterministic_daily_blocks(source_times, duration_hours, lower, upper)
    holdout = np.zeros_like(observed, dtype=bool)
    report = []
    for spec in blocks:
        start = spec["issue_start"]
        end_exclusive = spec["issue_end_exclusive"]
        affected = score_rows & (times >= start).to_numpy() & (times < end_exclusive).to_numpy()
        rows = np.flatnonzero(affected)
        block = np.zeros_like(observed, dtype=bool)
        if len(rows):
            block[np.ix_(rows, feature_indices)] = observed[np.ix_(rows, feature_indices)]
        holdout |= block
        report.append(
            {
                "issue_start": start.isoformat(),
                "issue_end_exclusive": end_exclusive.isoformat(),
                "duration_hours": int(duration_hours),
                "duration_daily_cycles": int(spec["duration_daily_cycles"]),
                "eligible_score_rows_affected": int(len(rows)),
                "hidden_cells": int(block.sum()),
                "evaluable": bool(len(rows) and block.any()),
            }
        )
    return holdout, report


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    source_times = v2.load_source_clock(features)
    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    score = roles == "score"
    if not score.any():
        raise ValueError("score role empty")

    # Model-ready source documentation says one UTC day per row. Fail closed if
    # the actual hash-pinned table does not contain daily transitions.
    diffs = source_times.diff().dropna()
    daily_fraction = float(np.mean(diffs == DAY)) if len(diffs) else 0.0
    if daily_fraction == 0.0:
        raise ValueError("hash-pinned source table contains no consecutive daily issue transitions")

    feature_names = list(base) + list(xrs) + list(proton)
    raw_values = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    observed = np.isfinite(raw_values)
    structural = ~observed
    fit_rows = roles == "fit"
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    modality_indices = {
        "XRS": np.asarray([name_to_index[n] for n in xrs], dtype=int),
        "PROTON": np.asarray([name_to_index[n] for n in proton], dtype=int),
        "XRS_AND_PROTON": np.asarray([name_to_index[n] for n in list(xrs) + list(proton)], dtype=int),
    }
    if any(len(v) == 0 for v in modality_indices.values()):
        raise ValueError("empty modality family")

    clean = transfer.build_clean_model(frame, y, roles, units, base, xrs, proton)
    reference = clean["clean_probability"]
    clean_metrics = {
        policy: transfer.score_metrics(y, reference, roles, threshold, clean["fit_prevalence"])
        for policy, threshold in clean["thresholds"].items()
    }

    prereg_path = Path(__file__).resolve().parents[1] / PREREG
    summary = {
        "status": "COMPLETED_DAILY_MODALITY_OUTAGE_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": v2.digest(prereg_path),
        "runner_sha256": v2.digest(Path(__file__)),
        "outage_semantics": (
            "Complete modality unavailable at consecutive daily forecast-input issue cycles in the "
            "model-ready table. 24/72/168 hours map to 1/3/7 daily cycles. This is not a raw "
            "5-minute sensor-stream outage experiment."
        ),
        "daily_source_transition_fraction": daily_fraction,
        "v1_v2_failures_preserved": True,
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_block_prior_inspection_disclosed": True,
        "feature_table_sha256": v2.digest(features),
        "event_catalogue_sha256": v2.digest(events),
        "source_clock_rows": int(len(source_times)),
        "source_clock_start": source_times.iloc[0].isoformat(),
        "source_clock_end": source_times.iloc[-1].isoformat(),
        "score_rows": int(score.sum()),
        "score_positives": int(y[score].sum()),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "model": {
            "id": "IRIS_CROSSFIT_EVIDENCE_STACK_V1",
            "stack_diagnostics": clean["stack"].diagnostics(),
            "calibration_intercept": clean["calibration_intercept"],
            "thresholds": clean["thresholds"],
            "clean_score_metrics": clean_metrics,
        },
        "scenarios": {},
    }

    prediction_columns = {
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
        "reference": reference,
    }

    for modality in MODALITIES:
        cols = modality_indices[modality]
        for duration_hours in DURATION_HOURS:
            holdout, blocks = scenario_holdout(
                observed,
                score,
                frame["window_end"],
                source_times,
                cols,
                duration_hours,
            )
            experimental = observed & ~holdout
            no_fill = raw_values.copy()
            no_fill[holdout] = np.nan
            median = recover_train_median(raw_values, observed, structural, holdout, fit_rows=fit_rows)
            forward = recover_causal_forward_fill(raw_values, observed, structural, holdout)
            values_by_arm = {
                "MASK_AWARE_NO_FILL": no_fill,
                "TRAIN_FIT_MEDIAN": median.values,
                "CAUSAL_FORWARD_FILL": forward.values,
            }
            available_by_arm = {
                "MASK_AWARE_NO_FILL": experimental,
                "TRAIN_FIT_MEDIAN": median.available_mask,
                "CAUSAL_FORWARD_FILL": forward.available_mask,
            }
            affected_rows = holdout.any(axis=1)
            scenario_key = f"{modality}_{duration_hours}H"
            scenario = {
                "modality": modality,
                "duration_hours": int(duration_hours),
                "duration_daily_cycles": int(DURATION_DAYS[duration_hours]),
                "blocks": blocks,
                "hidden_cells": int(holdout.sum()),
                "affected_score_rows": int(affected_rows.sum()),
                "evaluable_blocks": int(sum(bool(b["evaluable"]) for b in blocks)),
                "arms": {},
            }
            for arm in ARMS:
                vals = values_by_arm[arm].copy()
                vals[~available_by_arm[arm]] = np.nan
                candidate_frame = frame.copy()
                for j, name in enumerate(feature_names):
                    candidate_frame[name] = vals[:, j]
                p = transfer.candidate_probability(clean, candidate_frame, base, xrs, proton)
                result = {
                    "unresolved_hidden_cells": int((holdout & ~available_by_arm[arm]).sum()),
                    "probability_abs_drift_score": float(np.mean(np.abs(p[score] - reference[score]))),
                    "probability_abs_drift_affected_rows": (
                        float(np.mean(np.abs(p[affected_rows] - reference[affected_rows])))
                        if affected_rows.any()
                        else None
                    ),
                    "probability_bootstrap": transfer.bootstrap_probability_differences(
                        y, reference, p, units, roles
                    ),
                    "policies": {},
                }
                role_mask = np.where(score, "score", "outside")
                for policy, threshold in clean["thresholds"].items():
                    cand = transfer.score_metrics(y, p, roles, threshold, clean["fit_prevalence"])
                    ref = clean_metrics[policy]
                    tss_boot = cs.bootstrap_difference(
                        y,
                        p,
                        threshold,
                        reference,
                        threshold,
                        units,
                        role_mask,
                        "score",
                        seed=SEED,
                        replicates=10000,
                    )
                    result["policies"][policy] = {
                        "reference": ref,
                        "candidate": cand,
                        "delta_candidate_minus_reference": {
                            "TSS": float(cand["TSS"] - ref["TSS"]),
                            "BRIER": float(cand["BRIER"] - ref["BRIER"]),
                            "ECE": float(cand["ECE"] - ref["ECE"]),
                        },
                        "paired_TSS_bootstrap": tss_boot,
                    }
                scenario["arms"][arm] = result
                prediction_columns[f"p_{scenario_key}_{arm}"] = p
            summary["scenarios"][scenario_key] = scenario

    predictions = pd.DataFrame(prediction_columns)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = v2.digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(
        output / "receipt.json",
        {
            "status": summary["status"],
            "preregistration": PREREG,
            "preregistration_sha256": summary["preregistration_sha256"],
            "runner_sha256": summary["runner_sha256"],
            "predictions_sha256": summary["predictions_sha256"],
            "all_predeclared_scenarios_reported": True,
            "all_predeclared_arms_reported": True,
            "v1_v2_failures_preserved": True,
            "locked_test_accessed": False,
            "monitor_used": False,
            "retraining_after_outage": False,
            "recalibration_after_outage": False,
            "rethresholding_after_outage": False,
            "claim_boundary": (
                "Development-only daily forecast-input outage diagnostic. It cannot independently validate "
                "a raw-sensor outage policy, locked-test robustness, operational readiness or superiority."
            ),
        },
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.features, a.events, a.output)


if __name__ == "__main__":
    main()
