"""Preregistered availability-conditioned fallback experiment for IRIS-SEP.

No missing measurement is reconstructed or imputed. Specialist models and small
fusion rules are trained in advance for the information families that remain
available. Runtime feed loss selects the corresponding frozen fallback.

Development-only: score data are already inspected and no locked test is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.availability_fallback import (
    STATE_EXPERTS,
    degraded_candidate_gate,
    fit_availability_stacks,
    stacked_raw_probability,
)
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_contiguous_modality_outage_v2 as source_clock
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_event_terminal_modality_outage as event_outage
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1


PREREG = "config/availability_conditioned_fallback_preregistration_2026-09-07.json"
DURATION_HOURS = (24, 72, 168)
DURATION_DAYS = {24: 1, 72: 3, 168: 7}
MODALITY_TO_STATE = {
    "XRS": "NO_XRS",
    "PROTON": "NO_PROTON",
    "XRS_AND_PROTON": "NO_XRS_OR_PROTON",
}
SEED = 20260907
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
PROMOTED_REPRO_TOLERANCE = 1e-6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value):
    if isinstance(value, dict): return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray): return [finite_or_none(v) for v in value.tolist()]
    if isinstance(value, np.generic): return finite_or_none(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _subset_metrics(y, p, mask, threshold, prevalence):
    return event_outage._subset_summary(y, p, mask, threshold, prevalence)


def _fit_all_states(frame, y, roles, units, base, xrs, proton):
    fit = roles == "fit"
    fit_prevalence = float(np.mean(y[fit]))
    xrs_rel = cf.family_reliability(frame, xrs)
    proton_rel = cf.family_reliability(frame, proton)

    folds = cf.build_inner_folds(frame, y, roles, units)
    oof_rows, oof_evidence, oof_labels, fold_receipts = [], [], [], []
    for fold in folds:
        tr, sc, prev = fold["train_idx"], fold["score_idx"], fold["train_prevalence"]
        ps = cf.specialist_predict(frame, base, y, tr, sc)
        px = cf.specialist_predict(frame, xrs, y, tr, sc)
        pp = cf.specialist_predict(frame, proton, y, tr, sc)
        oof_rows.append(sc)
        oof_labels.append(y[sc])
        oof_evidence.append(np.column_stack([
            cf.centered_evidence(ps, prev),
            cf.centered_evidence(px, prev, xrs_rel[sc]),
            cf.centered_evidence(pp, prev, proton_rel[sc]),
        ]))
        fold_receipts.append({k: val for k, val in fold.items() if k not in ("train_idx", "score_idx")})

    rows = np.concatenate(oof_rows)
    evidence = np.concatenate(oof_evidence, axis=0)
    labels = np.concatenate(oof_labels)
    order = np.argsort(rows)
    rows, evidence, labels = rows[order], evidence[order], labels[order]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("OOF rows overlap")
    stacks = fit_availability_stacks(evidence, labels)

    family_models = {
        "solar": transfer.fit_family_models(frame, base, y, fit),
        "xrs": transfer.fit_family_models(frame, xrs, y, fit),
        "proton": transfer.fit_family_models(frame, proton, y, fit),
    }
    raw = {
        "solar": transfer.predict_family(family_models["solar"], frame, base),
        "xrs": transfer.predict_family(family_models["xrs"], frame, xrs),
        "proton": transfer.predict_family(family_models["proton"], frame, proton),
    }
    full_evidence = np.column_stack([
        cf.centered_evidence(raw["solar"], fit_prevalence),
        cf.centered_evidence(raw["xrs"], fit_prevalence, xrs_rel),
        cf.centered_evidence(raw["proton"], fit_prevalence, proton_rel),
    ])

    raw_state = {}
    probability = {}
    calibration = {}
    thresholds = {}
    for state in STATE_EXPERTS:
        raw_state[state] = stacked_raw_probability(
            state=state,
            full_evidence=full_evidence,
            raw_solar_probability=raw["solar"],
            stacks=stacks,
            sigmoid=v1.sigmoid,
        )
        probability[state], calibration[state] = cf.calibrated_probability(raw_state[state], y, roles)
        thresholds[state] = cf.thresholds(y, probability[state], roles)

    return {
        "fit_prevalence": fit_prevalence,
        "stacks": stacks,
        "family_models": family_models,
        "raw_family_probability": raw,
        "probability": probability,
        "calibration_intercept": calibration,
        "thresholds": thresholds,
        "oof_rows": int(len(rows)),
        "oof_positives": int(labels.sum()),
        "inner_folds": fold_receipts,
    }


def _whole_score_metrics(y, p, roles, threshold, prevalence):
    score = roles == "score"
    return {
        **v1.threshold_metrics(y[score], p[score], threshold),
        **v1.probability_metrics(y[score], p[score], prevalence),
        "matched_detection": {
            str(pod): v1.minimum_far_at_pod(y[score], p[score], pod)
            for pod in (0.6, 0.7, 0.8, 0.9)
        },
        "rows": int(score.sum()),
        "positives": int(y[score].sum()),
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    root = Path(__file__).resolve().parents[1]
    prereg = root / PREREG
    if not prereg.exists():
        raise ValueError("availability fallback preregistration missing")

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)

    # The preregistration forbids use of the previously inspected 2023-2025
    # monitor. build_scope_roles labels those rows so that other diagnostics can
    # evaluate them. Here we remove them completely before any specialist fit,
    # calibration, thresholding, prediction export, scenario selection, or
    # metric calculation. This preserves the exact pre-monitor role identities
    # while making monitor_used=false literal rather than merely aspirational.
    monitor_mask = roles == "monitor"
    monitor_rows_excluded = int(monitor_mask.sum())
    if monitor_rows_excluded:
        keep = ~monitor_mask
        frame = frame.loc[keep].reset_index(drop=True)
        y = np.asarray(y)[keep]
        event_ids = np.asarray(event_ids)[keep]
        roles = np.asarray(roles)[keep]
        units = np.asarray(units)[keep]
    if np.any(roles == "monitor"):
        raise ValueError("monitor exclusion failed")

    score = roles == "score"
    if not score.any() or int(y[score].sum()) == 0:
        raise ValueError("event-bearing score role required")

    model = _fit_all_states(frame, y, roles, units, base, xrs, proton)
    probability = model["probability"]
    clean = probability["FULL"]
    clean_metrics = {
        policy: _whole_score_metrics(y, clean, roles, model["thresholds"]["FULL"][policy], model["fit_prevalence"])
        for policy in POLICIES
    }

    # Verify that the FULL state is the promoted architecture, not a new clean-
    # data candidate. Two independent XGBoost fits can differ by tiny floating-
    # point amounts, so require both a tight probability tolerance and identical
    # score decisions under both frozen policies rather than impossible bitwise
    # equality.
    promoted = transfer.build_clean_model(frame, y, roles, units, base, xrs, proton)
    max_abs_full_diff = float(np.max(np.abs(promoted["clean_probability"] - clean)))
    if max_abs_full_diff > PROMOTED_REPRO_TOLERANCE:
        raise ValueError(f"FULL state does not reproduce promoted stack within tolerance: {max_abs_full_diff}")
    promoted_decision_mismatches = {}
    for policy in POLICIES:
        here = clean[score] >= float(model["thresholds"]["FULL"][policy])
        reference = promoted["clean_probability"][score] >= float(promoted["thresholds"][policy])
        mismatches = int(np.sum(here != reference))
        promoted_decision_mismatches[policy] = mismatches
        if mismatches:
            raise ValueError(f"FULL state changes {policy} score decisions: {mismatches}")

    source_times = source_clock.load_source_clock(features)
    raw_values = frame.loc[:, list(base) + list(xrs) + list(proton)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw_values)
    n_base, n_xrs, n_proton = len(base), len(xrs), len(proton)
    modality_cols = {
        "XRS": np.arange(n_base, n_base + n_xrs, dtype=int),
        "PROTON": np.arange(n_base + n_xrs, n_base + n_xrs + n_proton, dtype=int),
        "XRS_AND_PROTON": np.arange(n_base, n_base + n_xrs + n_proton, dtype=int),
    }
    times = frame["window_end"].reset_index(drop=True)

    summary = {
        "status": "COMPLETED_AVAILABILITY_CONDITIONED_FALLBACK_DEVELOPMENT_ONLY",
        "target": v1.TARGET,
        "preregistration": PREREG,
        "preregistration_sha256": digest(prereg),
        "runner_sha256": digest(Path(__file__)),
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "score_role_already_inspected": True,
        "imputation_used": False,
        "reconstruction_used": False,
        "retraining_at_outage_time": False,
        "full_state_max_abs_difference_vs_promoted_stack": max_abs_full_diff,
        "full_state_reproduction_tolerance": PROMOTED_REPRO_TOLERANCE,
        "full_state_score_decision_mismatches": promoted_decision_mismatches,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "oof_rows": model["oof_rows"],
        "oof_positives": model["oof_positives"],
        "inner_folds": model["inner_folds"],
        "states": {},
        "scenarios": {},
    }
    for state in STATE_EXPERTS:
        summary["states"][state] = {
            "experts": list(STATE_EXPERTS[state]),
            "calibration_intercept": float(model["calibration_intercept"][state]),
            "thresholds": model["thresholds"][state],
            "stack_diagnostics": None if state == "NO_XRS_OR_PROTON" else model["stacks"][state].diagnostics(),
            "whole_score": {
                policy: _whole_score_metrics(y, probability[state], roles, model["thresholds"][state][policy], model["fit_prevalence"])
                for policy in POLICIES
            },
        }

    prediction_columns = {
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
        "p_FULL": probability["FULL"],
        "p_NO_XRS": probability["NO_XRS"],
        "p_NO_PROTON": probability["NO_PROTON"],
        "p_NO_XRS_OR_PROTON": probability["NO_XRS_OR_PROTON"],
    }

    state_passes: dict[str, list[bool]] = {state: [] for state in MODALITY_TO_STATE.values()}
    for modality, state in MODALITY_TO_STATE.items():
        cols = modality_cols[modality]
        terminal_finite = np.any(finite[:, cols], axis=1)
        for hours in DURATION_HOURS:
            days = DURATION_DAYS[hours]
            event_rows = event_outage.select_event_rows(times, y, score, terminal_finite, source_times, days)
            quiet_rows = event_outage.select_quiet_controls(times, y, score, terminal_finite, source_times, days, event_rows)
            affected = event_rows | quiet_rows
            if not event_rows.any() or not quiet_rows.any():
                raise ValueError(f"event/control support missing for {modality} {hours}h")
            candidate = probability[state]
            scenario_key = f"{modality}_{hours}H"
            result = {
                "modality": modality,
                "availability_state": state,
                "duration_hours": int(hours),
                "event_rows": int(event_rows.sum()),
                "quiet_control_rows": int(quiet_rows.sum()),
                "event_identity_sha256": event_outage.identity_sha256(times[event_rows]),
                "quiet_identity_sha256": event_outage.identity_sha256(times[quiet_rows]),
                "probability_abs_drift_on_affected": float(np.mean(np.abs(candidate[affected] - clean[affected]))),
                "policies": {},
            }
            for policy in POLICIES:
                candidate_threshold = model["thresholds"][state][policy]
                reference_threshold = model["thresholds"]["FULL"][policy]
                candidate_affected = _subset_metrics(y, candidate, affected, candidate_threshold, model["fit_prevalence"])
                reference_affected = _subset_metrics(y, clean, affected, reference_threshold, model["fit_prevalence"])
                candidate_whole = summary["states"][state]["whole_score"][policy]
                reference_whole = clean_metrics[policy]
                affected_role = np.where(affected, "affected", "outside")
                score_role = np.where(score, "score", "outside")
                affected_boot = cs.bootstrap_difference(
                    y, candidate, candidate_threshold, clean, reference_threshold,
                    units, affected_role, "affected", seed=SEED, replicates=10000,
                )
                whole_boot = cs.bootstrap_difference(
                    y, candidate, candidate_threshold, clean, reference_threshold,
                    units, score_role, "score", seed=SEED, replicates=10000,
                )
                result["policies"][policy] = {
                    "candidate_affected": candidate_affected,
                    "reference_affected": reference_affected,
                    "candidate_whole_score": candidate_whole,
                    "reference_whole_score": reference_whole,
                    "affected_paired_TSS_bootstrap": affected_boot,
                    "whole_score_paired_TSS_bootstrap": whole_boot,
                    "delta_candidate_minus_reference": {
                        "affected_TSS": float(candidate_affected["TSS"] - reference_affected["TSS"]),
                        "whole_score_TSS": float(candidate_whole["TSS"] - reference_whole["TSS"]),
                        "whole_score_BRIER": float(candidate_whole["BRIER"] - reference_whole["BRIER"]),
                        "whole_score_ECE": float(candidate_whole["ECE"] - reference_whole["ECE"]),
                    },
                }
            primary = result["policies"]["MAX_TSS"]
            gate = degraded_candidate_gate(
                affected_tss=primary["candidate_affected"]["TSS"],
                affected_pod=primary["candidate_affected"]["POD"],
                whole_score_brier_delta=primary["delta_candidate_minus_reference"]["whole_score_BRIER"],
                whole_score_ece_delta=primary["delta_candidate_minus_reference"]["whole_score_ECE"],
                probabilities_finite=bool(np.isfinite(candidate).all()),
            )
            result["operator_gate"] = gate
            state_passes[state].append(bool(gate["passed"]))
            summary["scenarios"][scenario_key] = result
            prediction_columns[f"event_{scenario_key}"] = event_rows.astype(np.int8)
            prediction_columns[f"quiet_{scenario_key}"] = quiet_rows.astype(np.int8)

    summary["operator_state_decision"] = {}
    for state, passes in state_passes.items():
        all_pass = len(passes) == 3 and all(passes)
        summary["operator_state_decision"][state] = {
            "all_three_durations_pass_degraded_candidate_gate": bool(all_pass),
            "permission": "DEGRADED" if all_pass else "ABSTAIN",
            "normal_allowed": False,
        }

    predictions = pd.DataFrame(prediction_columns)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": summary["status"],
        "preregistration": PREREG,
        "preregistration_sha256": summary["preregistration_sha256"],
        "runner_sha256": summary["runner_sha256"],
        "predictions_sha256": summary["predictions_sha256"],
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "imputation_used": False,
        "reconstruction_used": False,
        "runtime_retraining": False,
        "normal_status_promotable": False,
        "all_availability_states_reported": set(summary["states"]) == set(STATE_EXPERTS),
        "all_nine_scenarios_reported": len(summary["scenarios"]) == 9,
        "claim_boundary": "Development-only availability fallback experiment. A pass grants only DEGRADED-candidate status; fresh independent evaluation is required for NORMAL or superiority claims."
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finite_or_none(run(args.features, args.events, args.output)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
