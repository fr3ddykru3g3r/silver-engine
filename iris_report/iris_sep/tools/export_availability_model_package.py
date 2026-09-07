"""Train and export the frozen IRIS-SEP availability-conditioned package.

Development packaging only. This tool does not access locked test data and does
not select new model hyperparameters. It serializes the already-preregistered
availability architecture and writes replay inputs for load-only verification.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from iris_report.iris_sep.src.iris_sep.model_package import PACKAGE_FORMAT, sha256_file
from iris_report.iris_sep.src.iris_sep.modeling.availability_fallback import STATE_EXPERTS
from iris_report.iris_sep.tools import run_availability_conditioned_fallback as af
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1

SEEDS = list(v1.SEEDS)
SOURCE_RUN = 34083153471
SOURCE_HEAD = "a43e059d04cdc112e773807219a3cc32b7142979"
SOURCE_ARTIFACT_SHA256 = "583cd505ae9e676dadb43b852a18a9fe26547f0cf57efc036ca208722d0297e6"


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(features: Path, events: Path, output: Path):
    output = Path(output)
    if output.exists():
        raise ValueError("output directory must be new")
    output.mkdir(parents=True)
    models_dir = output / "models"
    models_dir.mkdir()

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)
    monitor = roles == "monitor"
    monitor_rows_excluded = int(monitor.sum())
    if monitor_rows_excluded:
        keep = ~monitor
        frame = frame.loc[keep].reset_index(drop=True)
        y = np.asarray(y)[keep]
        roles = np.asarray(roles)[keep]
        units = np.asarray(units)[keep]
    if np.any(roles == "monitor"):
        raise ValueError("monitor exclusion failed")

    fitted = af._fit_all_states(frame, y, roles, units, base, xrs, proton)
    family_names = {"solar": list(base), "xrs": list(xrs), "proton": list(proton)}
    model_files = {}
    for family in ("solar", "xrs", "proton"):
        entries = []
        models = fitted["family_models"][family]
        if len(models) != len(SEEDS):
            raise ValueError("unexpected specialist count")
        for seed, model in zip(SEEDS, models):
            rel = Path("models") / f"{family}_seed_{seed}.json"
            path = output / rel
            model.save_model(path)
            entries.append({"seed": int(seed), "path": str(rel), "sha256": sha256_file(path)})
        model_files[family] = entries

    states = {}
    for state in STATE_EXPERTS:
        states[state] = {
            "experts": list(STATE_EXPERTS[state]),
            "calibration_intercept": float(fitted["calibration_intercept"][state]),
            "thresholds": {k: float(v) for k, v in fitted["thresholds"][state].items()},
            "stack": None if state == "NO_XRS_OR_PROTON" else {
                "intercept": float(fitted["stacks"][state].fit_.intercept),
                "weights": [float(v) for v in fitted["stacks"][state].fit_.weights],
                "l2_weight": float(fitted["stacks"][state].config.l2_weight),
            },
        }

    manifest = {
        "format": PACKAGE_FORMAT,
        "status": "DEVELOPMENT_ONLY_RELOADABLE_PACKAGE",
        "target": v1.TARGET,
        "architecture": "IRIS_AVAILABILITY_CONDITIONED_EVIDENCE_STACK_V1",
        "source_evidence": {
            "availability_fallback_run_id": SOURCE_RUN,
            "availability_fallback_head": SOURCE_HEAD,
            "availability_fallback_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        },
        "source_data": {
            "feature_table_sha256": af.digest(features),
            "event_catalogue_sha256": af.digest(events),
            "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
            "locked_test_accessed": False,
        },
        "seeds": SEEDS,
        "feature_families": family_names,
        "fit_prevalence": float(fitted["fit_prevalence"]),
        "evidence_limit": float(cf.EVIDENCE_LIMIT),
        "model_files": model_files,
        "states": states,
        "operator_permissions": {
            "FULL": "NORMAL_ONLY_IF_ADMISSION_PASSES",
            "NO_XRS": "DEGRADED",
            "NO_PROTON": "DEGRADED",
            "NO_XRS_OR_PROTON": "ABSTAIN"
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__
        },
        "training_receipt": {
            "positive_event_units": int(positive_units),
            "purged_units": purged,
            "oof_rows": int(fitted["oof_rows"]),
            "oof_positives": int(fitted["oof_positives"]),
            "inner_folds": fitted["inner_folds"],
            "runtime_retraining_supported": False,
            "runtime_recalibration_supported": False,
            "runtime_rethresholding_supported": False
        },
        "claim_boundary": "Development package for deterministic replay and research demonstration; not operational certification or fresh superiority evidence."
    }
    save_json(output / "manifest.json", manifest)

    replay = frame.loc[:, ["window_end"] + list(base) + list(xrs) + list(proton)].copy()
    replay.rename(columns={"window_end": "issue_time"}, inplace=True)
    replay.to_csv(output / "replay_input.csv", index=False)

    reference = pd.DataFrame({"issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")})
    for state in STATE_EXPERTS:
        reference[f"p_{state}"] = fitted["probability"][state]
    reference.to_csv(output / "export_reference_predictions.csv", index=False, float_format="%.17g")

    save_json(output / "package_receipt.json", {
        "status": "EXPORTED_RELOADABLE_AVAILABILITY_PACKAGE",
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "replay_input_sha256": sha256_file(output / "replay_input.csv"),
        "reference_predictions_sha256": sha256_file(output / "export_reference_predictions.csv"),
        "specialist_model_count": 15,
        "locked_test_accessed": False,
        "monitor_used": False
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
