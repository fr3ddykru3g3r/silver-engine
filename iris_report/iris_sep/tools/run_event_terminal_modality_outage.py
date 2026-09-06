"""Event-bearing terminal modality-outage stress test for IRIS-SEP.

This runner exists only after the preregistration in
config/event_terminal_modality_outage_preregistration_2026-09-06.json.
It is a development-only extension motivated by the zero-positive support of the
original label-blind daily-outage blocks.  It never uses model predictions to
select event or quiet-control issue rows.
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
from iris_report.iris_sep.tools import run_contiguous_modality_outage_v2 as v2
from iris_report.iris_sep.tools import run_daily_modality_outage as daily
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1

PREREG = "config/event_terminal_modality_outage_preregistration_2026-09-06.json"
PROVENANCE_CONTRACT = "config/source_provenance_contract_v1.json"
DURATION_HOURS = (24, 72, 168)
DURATION_DAYS = {24: 1, 72: 3, 168: 7}
MODALITIES = ("XRS", "PROTON", "XRS_AND_PROTON")
ARMS = ("MASK_AWARE_NO_FILL", "TRAIN_FIT_MEDIAN", "CAUSAL_FORWARD_FILL")
SEED = 20260906
DAY = pd.Timedelta(days=1)


def _finite_or_none(value):
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(_finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity_sha256(times: pd.Series | list[pd.Timestamp]) -> str:
    ordered = sorted(pd.Timestamp(t).isoformat() for t in list(times))
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def _source_has_history(issue: pd.Timestamp, days: int, source_set: set[pd.Timestamp]) -> bool:
    issue = pd.Timestamp(issue)
    return all(issue - i * DAY in source_set for i in range(days))


def select_event_rows(
    times: pd.Series,
    labels: np.ndarray,
    score: np.ndarray,
    terminal_finite: np.ndarray,
    source_times: pd.Series,
    duration_days: int,
) -> np.ndarray:
    """All positive score terminals with declared source history and exposure."""
    t = pd.Series(pd.to_datetime(times, utc=True, errors="raise")).reset_index(drop=True)
    source_set = set(pd.to_datetime(source_times, utc=True, errors="raise"))
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=bool)
    finite = np.asarray(terminal_finite, dtype=bool)
    if not (len(t) == len(labels) == len(score) == len(finite)):
        raise ValueError("event-selection arrays must align")
    out = np.zeros(len(t), dtype=bool)
    for i in np.flatnonzero(score & (labels == 1) & finite):
        if _source_has_history(t.iloc[i], duration_days, source_set):
            out[i] = True
    return out


def select_quiet_controls(
    times: pd.Series,
    labels: np.ndarray,
    score: np.ndarray,
    terminal_finite: np.ndarray,
    source_times: pd.Series,
    duration_days: int,
    event_rows: np.ndarray,
) -> np.ndarray:
    """One deterministic, unused, label-only quiet match per selected event."""
    t = pd.Series(pd.to_datetime(times, utc=True, errors="raise")).reset_index(drop=True)
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=bool)
    finite = np.asarray(terminal_finite, dtype=bool)
    event_rows = np.asarray(event_rows, dtype=bool)
    source_set = set(pd.to_datetime(source_times, utc=True, errors="raise"))
    if not (len(t) == len(labels) == len(score) == len(finite) == len(event_rows)):
        raise ValueError("quiet-selection arrays must align")
    all_positive_times = list(t[score & (labels == 1)])
    candidates = []
    for i in np.flatnonzero(score & (labels == 0) & finite):
        issue = t.iloc[i]
        if not _source_has_history(issue, duration_days, source_set):
            continue
        if all(abs(issue - pt) >= 8 * DAY for pt in all_positive_times):
            candidates.append(i)
    used: set[int] = set()
    chosen: list[int] = []
    for event_i in np.flatnonzero(event_rows):
        event_t = t.iloc[event_i]
        remaining = [i for i in candidates if i not in used]
        same_year = [i for i in remaining if t.iloc[i].year == event_t.year]
        pool = same_year or remaining
        if not pool:
            continue
        best = min(pool, key=lambda i: (abs(t.iloc[i] - event_t), t.iloc[i]))
        used.add(best)
        chosen.append(best)
    mask = np.zeros(len(t), dtype=bool)
    mask[chosen] = True
    return mask


def _terminal_and_block_holdout(
    values: np.ndarray,
    times: pd.Series,
    terminal_rows: np.ndarray,
    cols: np.ndarray,
    duration_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return terminal truth mask plus complete preceding gap mask for causal fill."""
    finite = np.isfinite(values)
    t = pd.Series(pd.to_datetime(times, utc=True, errors="raise")).reset_index(drop=True)
    terminals = np.asarray(terminal_rows, dtype=bool)
    terminal = np.zeros_like(finite, dtype=bool)
    block = np.zeros_like(finite, dtype=bool)
    for row in np.flatnonzero(terminals):
        end = t.iloc[row]
        start = end - (duration_days - 1) * DAY
        inside = (t >= start).to_numpy() & (t <= end).to_numpy()
        inside_rows = np.flatnonzero(inside)
        block[np.ix_(inside_rows, cols)] |= finite[np.ix_(inside_rows, cols)]
        terminal[row, cols] = finite[row, cols]
    return terminal, block


def _subset_summary(y, p, mask, threshold, prevalence):
    return daily.metrics_on_mask(y, p, mask, threshold, prevalence)


def _handling_accounting(y, p, score, affected, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    score = np.asarray(score, dtype=bool)
    affected = np.asarray(affected, dtype=bool) & score
    covered = score & ~affected
    pred = p >= threshold
    total_pos = int(np.sum(score & (y == 1)))
    covered_tp = int(np.sum(covered & (y == 1) & pred))
    covered_fn = int(np.sum(covered & (y == 1) & ~pred))
    abstained_pos = int(np.sum(affected & (y == 1)))
    false_alerts = int(np.sum(covered & (y == 0) & pred))
    exposed_alerts = covered_tp + false_alerts
    return {
        "coverage_fraction": float(np.sum(covered) / np.sum(score)),
        "abstained_rows": int(np.sum(affected)),
        "abstained_positives": abstained_pos,
        "coverage_adjusted_missed_positives": covered_fn + abstained_pos,
        "coverage_adjusted_detection_fraction": None if total_pos == 0 else float(covered_tp / total_pos),
        "false_alerts_among_exposed_rows": false_alerts,
        "false_alert_ratio_among_exposed_alerts": None if exposed_alerts == 0 else float(false_alerts / exposed_alerts),
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    source_times = v2.load_source_clock(features)
    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    score = roles == "score"
    fit_rows = roles == "fit"
    if not score.any() or int(y[score].sum()) == 0:
        raise ValueError("event-bearing score role required")

    feature_names = list(base) + list(xrs) + list(proton)
    raw = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    structural = ~finite
    name_to_i = {n: i for i, n in enumerate(feature_names)}
    modality_indices = {
        "XRS": np.asarray([name_to_i[n] for n in xrs], dtype=int),
        "PROTON": np.asarray([name_to_i[n] for n in proton], dtype=int),
        "XRS_AND_PROTON": np.asarray([name_to_i[n] for n in list(xrs) + list(proton)], dtype=int),
    }

    clean = transfer.build_clean_model(frame, y, roles, units, base, xrs, proton)
    reference = clean["clean_probability"]
    clean_metrics = {
        policy: transfer.score_metrics(y, reference, roles, threshold, clean["fit_prevalence"])
        for policy, threshold in clean["thresholds"].items()
    }
    root = Path(__file__).resolve().parents[1]
    prereg = root / PREREG
    provenance = root / PROVENANCE_CONTRACT
    if not prereg.exists() or not provenance.exists():
        raise ValueError("required preregistration/provenance contract missing")

    summary = {
        "status": "COMPLETED_EVENT_TERMINAL_MODALITY_OUTAGE_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": v2.digest(prereg),
        "runner_sha256": v2.digest(Path(__file__)),
        "provenance_contract_sha256": v2.digest(provenance),
        "feature_table_sha256": v2.digest(features),
        "event_catalogue_sha256": v2.digest(events),
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_role_already_inspected": True,
        "prior_daily_outage_results_inspected": True,
        "score_rows": int(score.sum()),
        "score_positives": int(y[score].sum()),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "model": {
            "id": "IRIS_CROSSFIT_EVIDENCE_STACK_V1",
            "fit_prevalence": float(clean["fit_prevalence"]),
            "calibration_intercept": float(clean["calibration_intercept"]),
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

    times = frame["window_end"].reset_index(drop=True)
    for modality in MODALITIES:
        cols = modality_indices[modality]
        terminal_finite = np.any(finite[:, cols], axis=1)
        for hours in DURATION_HOURS:
            days = DURATION_DAYS[hours]
            event_rows = select_event_rows(times, y, score, terminal_finite, source_times, days)
            quiet_rows = select_quiet_controls(times, y, score, terminal_finite, source_times, days, event_rows)
            affected = event_rows | quiet_rows
            if not event_rows.any():
                raise ValueError(f"no eligible positive terminal rows for {modality} {hours}h")
            terminal_holdout, block_holdout = _terminal_and_block_holdout(raw, times, affected, cols, days)
            if not terminal_holdout.any():
                raise ValueError("terminal outage contains no finite interface cells")

            # Median is fitted only on the historical fit role and applied only to
            # terminal truth cells.  Forward fill sees a complete preceding block
            # as missing, but only its terminal reconstruction is injected into the
            # candidate forecast so the stress cohort remains exactly event+control.
            median_recovery = recover_train_median(raw, finite, structural, terminal_holdout, fit_rows=fit_rows)
            forward_recovery = recover_causal_forward_fill(raw, finite, structural, block_holdout)

            no_fill = raw.copy()
            no_fill[terminal_holdout] = np.nan
            median_values = raw.copy()
            median_values[terminal_holdout] = median_recovery.values[terminal_holdout]
            forward_values = raw.copy()
            terminal_forward_resolved = terminal_holdout & forward_recovery.available_mask
            forward_values[terminal_holdout] = np.nan
            forward_values[terminal_forward_resolved] = forward_recovery.values[terminal_forward_resolved]

            values_by_arm = {
                "MASK_AWARE_NO_FILL": no_fill,
                "TRAIN_FIT_MEDIAN": median_values,
                "CAUSAL_FORWARD_FILL": forward_values,
            }
            unresolved_by_arm = {
                "MASK_AWARE_NO_FILL": int(terminal_holdout.sum()),
                "TRAIN_FIT_MEDIAN": int(np.sum(terminal_holdout & ~median_recovery.available_mask)),
                "CAUSAL_FORWARD_FILL": int(np.sum(terminal_holdout & ~forward_recovery.available_mask)),
            }
            key = f"{modality}_{hours}H"
            scenario = {
                "modality": modality,
                "duration_hours": int(hours),
                "duration_daily_cycles": int(days),
                "eligible_positive_terminal_rows": int(event_rows.sum()),
                "excluded_score_positive_terminal_rows": int(y[score].sum() - event_rows.sum()),
                "positive_terminal_identity_sha256": identity_sha256(times[event_rows]),
                "quiet_control_terminal_rows": int(quiet_rows.sum()),
                "quiet_control_identity_sha256": identity_sha256(times[quiet_rows]),
                "hidden_terminal_cells": int(terminal_holdout.sum()),
                "affected_terminal_rows": int(affected.sum()),
                "arms": {},
            }
            prediction_columns[f"event_{key}"] = event_rows.astype(np.int8)
            prediction_columns[f"quiet_{key}"] = quiet_rows.astype(np.int8)

            for arm, vals in values_by_arm.items():
                candidate_frame = frame.copy()
                for j, name in enumerate(feature_names):
                    candidate_frame[name] = vals[:, j]
                prob = transfer.candidate_probability(clean, candidate_frame, base, xrs, proton)
                event_ref = reference[event_rows]
                event_prob = prob[event_rows]
                result = {
                    "unresolved_hidden_cells": unresolved_by_arm[arm],
                    "event_probability_mean_reference": float(np.mean(event_ref)),
                    "event_probability_mean_candidate": float(np.mean(event_prob)),
                    "event_probability_absolute_drift": float(np.mean(np.abs(event_prob - event_ref))),
                    "event_probability_signed_delta": float(np.mean(event_prob - event_ref)),
                    "quiet_probability_absolute_drift": float(np.mean(np.abs(prob[quiet_rows] - reference[quiet_rows]))) if quiet_rows.any() else None,
                    "whole_score_probability_absolute_drift": float(np.mean(np.abs(prob[score] - reference[score]))),
                    "policies": {},
                }
                role_mask = np.where(score, "score", "outside")
                affected_role = np.where(affected, "affected", "outside")
                for policy, threshold in clean["thresholds"].items():
                    whole = transfer.score_metrics(y, prob, roles, threshold, clean["fit_prevalence"])
                    ref = clean_metrics[policy]
                    boot = cs.bootstrap_difference(
                        y, prob, threshold, reference, threshold, units, role_mask, "score",
                        seed=SEED, replicates=10000,
                    )
                    affected_boot = cs.bootstrap_difference(
                        y, prob, threshold, reference, threshold, units, affected_role, "affected",
                        seed=SEED, replicates=10000,
                    ) if quiet_rows.any() else None
                    result["policies"][policy] = {
                        "event_rows_reference": _subset_summary(y, reference, event_rows, threshold, clean["fit_prevalence"]),
                        "event_rows_candidate": _subset_summary(y, prob, event_rows, threshold, clean["fit_prevalence"]),
                        "quiet_rows_reference": _subset_summary(y, reference, quiet_rows, threshold, clean["fit_prevalence"]) if quiet_rows.any() else None,
                        "quiet_rows_candidate": _subset_summary(y, prob, quiet_rows, threshold, clean["fit_prevalence"]) if quiet_rows.any() else None,
                        "combined_rows_reference": _subset_summary(y, reference, affected, threshold, clean["fit_prevalence"]),
                        "combined_rows_candidate": _subset_summary(y, prob, affected, threshold, clean["fit_prevalence"]),
                        "whole_score_reference": ref,
                        "whole_score_candidate": whole,
                        "whole_score_delta_candidate_minus_reference": {
                            "TSS": float(whole["TSS"] - ref["TSS"]),
                            "BRIER": float(whole["BRIER"] - ref["BRIER"]),
                            "ECE": float(whole["ECE"] - ref["ECE"]),
                        },
                        "whole_score_paired_TSS_bootstrap": boot,
                        "affected_rows_paired_TSS_bootstrap": affected_boot,
                        "handling": {
                            "ALWAYS_EXPOSE_NORMAL": {"coverage_fraction": 1.0, "status": "NORMAL"},
                            "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE": {"coverage_fraction": 1.0, "status": "DEGRADED", "degraded_rows": int(affected.sum()), "degraded_positives": int(y[event_rows].sum())},
                            "ABSTAIN_ON_DECLARED_OUTAGE": _handling_accounting(y, prob, score, affected, threshold),
                        },
                    }
                scenario["arms"][arm] = result
                prediction_columns[f"p_{key}_{arm}"] = prob
            summary["scenarios"][key] = scenario

    predictions = pd.DataFrame(prediction_columns)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = v2.digest(output / "predictions.csv")
    _save_json(output / "summary.json", summary)
    _save_json(output / "receipt.json", {
        "status": summary["status"],
        "preregistration": PREREG,
        "preregistration_sha256": summary["preregistration_sha256"],
        "runner_sha256": summary["runner_sha256"],
        "predictions_sha256": summary["predictions_sha256"],
        "all_predeclared_scenarios_reported": len(summary["scenarios"]) == 9,
        "all_predeclared_arms_reported": all(set(v["arms"]) == set(ARMS) for v in summary["scenarios"].values()),
        "event_bearing_support_required": True,
        "locked_test_accessed": False,
        "monitor_used": False,
        "retraining_after_outage": False,
        "recalibration_after_outage": False,
        "rethresholding_after_outage": False,
        "claim_boundary": "Development-only event-stratified aggregate-interface outage stress test; not raw-sensor reconstruction accuracy, locked-test robustness or operational certification."
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_finite_or_none(run(args.features, args.events, args.output)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
