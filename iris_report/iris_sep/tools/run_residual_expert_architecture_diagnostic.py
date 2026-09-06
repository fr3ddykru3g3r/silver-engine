"""One-shot development diagnostic for IRIS residual expert architecture.

Preregistered in config/residual_expert_architecture_preregistration_2026-09-06.json.
The score and 2023-2025 monitor are already-inspected development evidence.
The locked test is never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.residual_logit_fusion import ResidualLogitFusion
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs


POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
MODELS = ("BASE_SOLAR", "LATE_FUSION_SOLAR_XRS_PROTON", "IRIS_RESIDUAL_EXPERT_V1")


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


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def family_reliability(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Observed fraction only; no target, score, or high-dimensional learned gate."""
    values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    reliability = np.mean(np.isfinite(values), axis=1)
    if ((reliability < 0) | (reliability > 1) | ~np.isfinite(reliability)).any():
        raise ValueError("invalid family reliability")
    return reliability.astype(np.float64)


def calibrated_solar(raw: np.ndarray, y: np.ndarray, roles: np.ndarray):
    cal = roles == "calibration"
    intercept = v1.fit_intercept(raw[cal], y[cal])
    return v1.sigmoid(v1.logit(raw) + intercept), {"calibration_intercept": float(intercept)}


def thresholds(y: np.ndarray, p: np.ndarray, roles: np.ndarray) -> dict[str, float]:
    mask = roles == "threshold"
    max_tss = float(v1.select_threshold(y[mask], p[mask]))
    pod80 = v1.minimum_far_at_pod(y[mask], p[mask], 0.8)
    if pod80 is None:
        raise ValueError("POD80 threshold unavailable")
    return {"MAX_TSS": max_tss, "POD80_MIN_FAR": float(pod80["threshold"])}


def evaluate(y, p, t, roles, role, prevalence):
    mask = roles == role
    if int(mask.sum()) == 0 or len(np.unique(y[mask])) != 2:
        raise ValueError(f"role {role} lacks both classes")
    return {
        **v1.threshold_metrics(y[mask], p[mask], t),
        **v1.probability_metrics(y[mask], p[mask], prevalence),
        "matched_detection": {
            str(pod): v1.minimum_far_at_pod(y[mask], p[mask], pod)
            for pod in (0.6, 0.7, 0.8, 0.9)
        },
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
    }


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    fit = roles == "fit"
    cal = roles == "calibration"
    if len(np.unique(y[fit])) != 2 or len(np.unique(y[cal])) != 2:
        raise ValueError("fit/calibration role lacks both classes")
    prevalence = float(np.mean(y[fit]))

    # Specialist models are frozen to the same five-seed XGBoost recipe used by
    # the successful modality-separated development diagnostic.
    raw_solar = cs.fit_xgb_family(frame, base, y, fit)
    raw_xrs = cs.fit_xgb_family(frame, xrs, y, fit)
    raw_proton = cs.fit_xgb_family(frame, proton, y, fit)

    probability = {}
    metadata = {}
    probability["BASE_SOLAR"], metadata["BASE_SOLAR"] = calibrated_solar(raw_solar, y, roles)

    late, _late_t, late_meta = cs.fit_late_stack([raw_solar, raw_xrs, raw_proton], y, roles)
    probability["LATE_FUSION_SOLAR_XRS_PROTON"] = late
    metadata["LATE_FUSION_SOLAR_XRS_PROTON"] = late_meta

    xrs_rel = family_reliability(frame, xrs)
    proton_rel = family_reliability(frame, proton)
    fusion = ResidualLogitFusion()
    fusion.fit(
        raw_solar[cal], raw_xrs[cal], raw_proton[cal], y[cal],
        xrs_reliability=xrs_rel[cal], proton_reliability=proton_rel[cal],
    )
    residual = fusion.predict_proba(
        raw_solar, raw_xrs, raw_proton,
        xrs_reliability=xrs_rel, proton_reliability=proton_rel,
    )
    probability["IRIS_RESIDUAL_EXPERT_V1"] = residual
    metadata["IRIS_RESIDUAL_EXPERT_V1"] = {
        **fusion.diagnostics(),
        "xrs_reliability_mean": float(np.mean(xrs_rel)),
        "proton_reliability_mean": float(np.mean(proton_rel)),
        "xrs_zero_reliability_rows": int(np.sum(xrs_rel == 0)),
        "proton_zero_reliability_rows": int(np.sum(proton_rel == 0)),
    }

    model_thresholds = {name: thresholds(y, p, roles) for name, p in probability.items()}
    summary = {
        "status": "COMPLETED_RESIDUAL_EXPERT_ARCHITECTURE_DEVELOPMENT_DIAGNOSTIC",
        "target": v1.TARGET,
        "locked_test_accessed": False,
        "score_and_monitor_already_inspected": True,
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric_columns": dropped,
        "feature_family_sizes": {"solar": len(base), "xrs": len(xrs), "proton": len(proton)},
        "models": {},
        "paired_comparisons": {},
    }

    for name in MODELS:
        summary["models"][name] = {
            "metadata": metadata[name],
            "thresholds": model_thresholds[name],
            "policies": {},
        }
        for policy in POLICIES:
            t = model_thresholds[name][policy]
            summary["models"][name]["policies"][policy] = {
                "score": evaluate(y, probability[name], t, roles, "score", prevalence),
                "monitor": evaluate(y, probability[name], t, roles, "monitor", prevalence),
            }

    for policy in POLICIES:
        for role in ("score", "monitor"):
            role_mask = np.where(roles == role, role, "outside")
            for other in ("BASE_SOLAR", "LATE_FUSION_SOLAR_XRS_PROTON"):
                key = f"IRIS_RESIDUAL_EXPERT_V1_minus_{other}_{policy}_{role}"
                summary["paired_comparisons"][key] = cs.bootstrap_difference(
                    y,
                    probability["IRIS_RESIDUAL_EXPERT_V1"], model_thresholds["IRIS_RESIDUAL_EXPERT_V1"][policy],
                    probability[other], model_thresholds[other][policy],
                    units, role_mask, role,
                    seed=20260906, replicates=10000,
                )

    rows = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
        "xrs_reliability": xrs_rel,
        "proton_reliability": proton_rel,
    })
    for name in MODELS:
        rows[name] = probability[name]
    rows.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    summary["predictions_sha256"] = digest(output / "predictions.csv")
    save_json(output / "summary.json", summary)
    save_json(output / "receipt.json", {
        "status": "DEVELOPMENT_ONLY_RESIDUAL_EXPERT_RUN",
        "preregistration": "config/residual_expert_architecture_preregistration_2026-09-06.json",
        "architecture_source": "src/iris_sep/modeling/residual_logit_fusion.py",
        "locked_test_accessed": False,
        "score_and_monitor_prior_inspection_disclosed": True,
        "post_result_hyperparameter_changes": False,
        "unfavorable_models_or_seeds_dropped": False,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
