"""Preregistered contiguous-modality outage benchmark on the promoted stack.

The model/calibration/thresholds are frozen on clean development roles before
any outage. Five deterministic, label-blind, non-overlapping outage blocks are
applied per modality/duration scenario on the already-inspected score role.
Development-only; locked test and monitor are never used.
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
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer


PREREG = "config/contiguous_modality_outage_preregistration_2026-09-06.json"
DURATIONS = (24, 72, 168)
MODALITIES = ("XRS", "PROTON", "XRS_AND_PROTON")
ARMS = transfer.ARMS
START_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
SEED = 20260906


def finite_or_none(value):
    if isinstance(value, dict): return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray): return [finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic): return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def deterministic_block_starts(n_rows: int, length: int) -> list[int]:
    if length <= 0 or n_rows <= length:
        raise ValueError("outage length incompatible with score rows")
    max_start = n_rows - length
    starts = [int(round(f * max_start)) for f in START_FRACTIONS]
    if len(set(starts)) != len(starts):
        raise ValueError("deterministic block starts collide")
    ordered = sorted(starts)
    if any(b < a + length for a, b in zip(ordered, ordered[1:])):
        raise ValueError("deterministic outage blocks overlap")
    return starts


def scenario_holdout(observed, score_rows, feature_indices, length):
    score_idx = np.flatnonzero(score_rows)
    starts_local = deterministic_block_starts(len(score_idx), length)
    holdout = np.zeros_like(observed, dtype=bool)
    blocks = []
    for local_start in starts_local:
        local_end = local_start + length
        rows = score_idx[local_start:local_end]
        block = np.zeros_like(observed, dtype=bool)
        block[np.ix_(rows, feature_indices)] = observed[np.ix_(rows, feature_indices)]
        if not block.any():
            raise ValueError("outage block contains no observed modality cells")
        holdout |= block
        blocks.append({
            "score_local_start": int(local_start),
            "score_local_end_exclusive": int(local_end),
            "global_row_start": int(rows[0]),
            "global_row_end": int(rows[-1]),
            "hidden_cells": int(block.sum()),
        })
    return holdout, blocks


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists(): raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    score = roles == "score"
    if not score.any(): raise ValueError("score role empty")
    score_times = frame.loc[score, "window_end"].reset_index(drop=True)
    diffs = score_times.diff().dropna()
    if len(diffs) == 0 or not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError("contiguous modality benchmark requires exact 1-hour score cadence")

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
        "status": "COMPLETED_CONTIGUOUS_MODALITY_OUTAGE_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": transfer.digest(prereg_path),
        "locked_test_accessed": False,
        "monitor_used": False,
        "score_block_prior_inspection_disclosed": True,
        "feature_table_sha256": transfer.digest(features),
        "event_catalogue_sha256": transfer.digest(events),
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
            holdout, blocks = scenario_holdout(observed, score, cols, duration)
            experimental = observed & ~holdout
            no_fill = raw_values.copy(); no_fill[holdout] = np.nan
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
                    "probability_abs_drift_affected_rows": float(np.mean(np.abs(p[affected_rows] - reference[affected_rows]))),
                    "probability_bootstrap": transfer.bootstrap_probability_differences(y, reference, p, units, roles),
                    "policies": {},
                }
                role_mask = np.where(score, "score", "outside")
                for policy, threshold in clean["thresholds"].items():
                    cand = transfer.score_metrics(y, p, roles, threshold, clean["fit_prevalence"])
                    ref = clean_metrics[policy]
                    tss_boot = cs.bootstrap_difference(
                        y, p, threshold, reference, threshold,
                        units, role_mask, "score", seed=SEED, replicates=10000,
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
    summary["predictions_sha256"] = transfer.digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": summary["status"],
        "preregistration": PREREG,
        "preregistration_sha256": summary["preregistration_sha256"],
        "predictions_sha256": summary["predictions_sha256"],
        "all_predeclared_scenarios_reported": True,
        "all_predeclared_arms_reported": True,
        "locked_test_accessed": False,
        "monitor_used": False,
        "retraining_after_outage": False,
        "recalibration_after_outage": False,
        "rethresholding_after_outage": False,
        "claim_boundary": "Development-only contiguous modality-outage realism diagnostic; may motivate but cannot validate an abstention policy."
    })
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); run(a.features, a.events, a.output)


if __name__ == "__main__": main()
