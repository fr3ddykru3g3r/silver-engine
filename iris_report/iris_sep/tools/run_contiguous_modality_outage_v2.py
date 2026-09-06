"""V2 contiguous-modality outage benchmark on the promoted IRIS-SEP stack.

V1 correctly refused to invent a contiguous outage inside the filtered score
identities, but that revealed a semantics bug: an instrument outage occurs on
the source clock, not on the subset of forecast rows that remain eligible for
scoring. V2 therefore places each outage on the complete hourly feature-table
issue clock first, then projects the interval onto eligible score rows.

The forecast model, calibration, thresholds, durations, modality scenarios,
recovery arms, number of blocks, block fractions, bootstrap and evaluation
roles remain frozen. This is development-only; locked test and monitor are not
used.

This historical hourly-interface prototype is retained for audit continuity and
has been superseded scientifically by the daily aggregate-interface outage
experiment. Timestamp mechanics remain tested and portable across pandas
storage resolutions.
"""
from __future__ import annotations

import argparse
import hashlib
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
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer


PREREG = "config/contiguous_modality_outage_preregistration_v2_2026-09-06.json"
DURATIONS = (24, 72, 168)
MODALITIES = ("XRS", "PROTON", "XRS_AND_PROTON")
ARMS = transfer.ARMS
START_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
SEED = 20260906
HOUR = pd.Timedelta(hours=1)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
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


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_source_clock(features: Path) -> pd.Series:
    """Return the complete unique feature-table issue clock before eligibility filters."""
    clock = pd.read_csv(features, usecols=["window_begin", "window_end"])
    clock["window_begin"] = pd.to_datetime(clock["window_begin"], utc=True, errors="raise")
    clock["window_end"] = pd.to_datetime(clock["window_end"], utc=True, errors="raise")
    if not ((clock["window_end"] - clock["window_begin"]) == pd.Timedelta(hours=24)).all():
        raise ValueError("source predictor windows must be exactly 24 hours")
    clock = (
        clock.sort_values(["window_end", "window_begin"])
        .drop_duplicates(["window_begin", "window_end"])
        .reset_index(drop=True)
    )
    times = clock["window_end"]
    if times.empty or times.duplicated().any() or np.any(times.diff().dropna() <= pd.Timedelta(0)):
        raise ValueError("source issue clock must be nonempty, unique and strictly increasing")
    return times


def _contiguous_source_starts(
    source_times: pd.Series,
    duration_hours: int,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> list[pd.Timestamp]:
    """All starts with every hourly source timestamp present in [start,end)."""
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    times = pd.Series(pd.to_datetime(source_times, utc=True, errors="raise")).reset_index(drop=True)
    times = times[(times >= lower) & (times <= upper)].reset_index(drop=True)
    if len(times) < duration_hours:
        raise ValueError("source clock too short for requested outage")
    starts: list[pd.Timestamp] = []
    for i in range(0, len(times) - duration_hours + 1):
        segment = times.iloc[i:i + duration_hours]
        if duration_hours == 1 or (segment.diff().dropna() == HOUR).all():
            start = segment.iloc[0]
            end_exclusive = start + pd.Timedelta(hours=duration_hours)
            if end_exclusive - HOUR <= upper:
                starts.append(start)
    return starts


def deterministic_source_blocks(
    source_times: pd.Series,
    duration_hours: int,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> list[dict]:
    """Choose the preregistered five label-blind blocks on the full source clock."""
    admissible = _contiguous_source_starts(source_times, duration_hours, lower, upper)
    if len(admissible) < len(START_FRACTIONS):
        raise ValueError(
            f"too few source-clock starts for {duration_hours}-hour outage: {len(admissible)}"
        )
    positions = [int(round(f * (len(admissible) - 1))) for f in START_FRACTIONS]
    starts = [admissible[p] for p in positions]
    if len(set(starts)) != len(starts):
        raise ValueError("deterministic source-clock starts collide")
    ordered = sorted(starts)
    duration = pd.Timedelta(hours=duration_hours)
    if any(b < a + duration for a, b in zip(ordered, ordered[1:])):
        raise ValueError("deterministic source-clock outage blocks overlap")
    return [
        {
            "issue_start": start,
            "issue_end_exclusive": start + duration,
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
    """Place outage on source clock, then project it onto eligible score rows."""
    score_rows = np.asarray(score_rows, dtype=bool)
    if observed.ndim != 2 or len(score_rows) != observed.shape[0]:
        raise ValueError("observed and score_rows must align")
    times = pd.Series(pd.to_datetime(eligible_times, utc=True, errors="raise")).reset_index(drop=True)
    if len(times) != observed.shape[0]:
        raise ValueError("eligible_times must align with observed rows")
    if not score_rows.any():
        raise ValueError("score role empty")
    lower = times[score_rows].min()
    upper = times[score_rows].max()
    source_blocks = deterministic_source_blocks(source_times, duration_hours, lower, upper)

    holdout = np.zeros_like(observed, dtype=bool)
    blocks = []
    for spec in source_blocks:
        start = spec["issue_start"]
        end_exclusive = spec["issue_end_exclusive"]
        affected = score_rows & (times >= start).to_numpy() & (times < end_exclusive).to_numpy()
        rows = np.flatnonzero(affected)
        block = np.zeros_like(observed, dtype=bool)
        if len(rows):
            block[np.ix_(rows, feature_indices)] = observed[np.ix_(rows, feature_indices)]
        holdout |= block
        blocks.append(
            {
                "issue_start": start.isoformat(),
                "issue_end_exclusive": end_exclusive.isoformat(),
                "source_duration_hours": int(duration_hours),
                "eligible_score_rows_affected": int(len(rows)),
                "first_eligible_global_row": int(rows[0]) if len(rows) else None,
                "last_eligible_global_row": int(rows[-1]) if len(rows) else None,
                "hidden_cells": int(block.sum()),
                "evaluable": bool(len(rows) and block.any()),
            }
        )
    return holdout, blocks


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    source_times = load_source_clock(features)
    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    score = roles == "score"
    if not score.any():
        raise ValueError("score role empty")

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
        "status": "COMPLETED_CONTIGUOUS_MODALITY_OUTAGE_V2_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": digest(prereg_path),
        "runner_sha256": digest(Path(__file__)),
        "outage_semantics": (
            "Complete modality forecast-input outage is placed on the full hourly public feature-table "
            "issue clock before target-eligibility/purge filtering; the interval is then projected onto "
            "eligible score decisions. This does not claim minute-scale raw sensor reaggregation."
        ),
        "v1_failure_preserved": {
            "run": "33987778788",
            "head": "4c70b5f089cd826f37425f8f7e87db72b466076b",
            "reason": "V1 incorrectly required eligible score identities themselves to form an hourly-contiguous outage clock.",
        },
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_block_prior_inspection_disclosed": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
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
        for duration in DURATIONS:
            holdout, blocks = scenario_holdout(
                observed,
                score,
                frame["window_end"],
                source_times,
                cols,
                duration,
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
            scenario_key = f"{modality}_{duration}H"
            scenario = {
                "modality": modality,
                "duration_hours": int(duration),
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
                affected_drift = (
                    float(np.mean(np.abs(p[affected_rows] - reference[affected_rows])))
                    if affected_rows.any()
                    else None
                )
                result = {
                    "unresolved_hidden_cells": int((holdout & ~available_by_arm[arm]).sum()),
                    "probability_abs_drift_score": float(np.mean(np.abs(p[score] - reference[score]))),
                    "probability_abs_drift_affected_rows": affected_drift,
                    "probability_bootstrap": transfer.bootstrap_probability_differences(y, reference, p, units, roles),
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
    summary["predictions_sha256"] = digest(output / "predictions.csv")
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
            "v1_failed_run_preserved": True,
            "locked_test_accessed": False,
            "monitor_used": False,
            "retraining_after_outage": False,
            "recalibration_after_outage": False,
            "rethresholding_after_outage": False,
            "claim_boundary": (
                "Development-only whole-modality forecast-input outage diagnostic on already-inspected score identities; "
                "may motivate but cannot independently validate an abstention policy or superiority claim."
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
