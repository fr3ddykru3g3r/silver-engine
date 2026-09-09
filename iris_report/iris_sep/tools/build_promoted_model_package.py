"""Build the promoted IRIS-SEP model once, export it, reload it, and verify replay.

Development-only packaging step. The fit/calibration/threshold roles are frozen;
monitor rows are removed before any modeling and locked test is never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost

from iris_report.iris_sep.src.iris_sep.promoted_model_package import (
    export_promoted_package,
    load_promoted_package,
    sha256_file,
)
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1


REPLAY_TOLERANCE = 1e-12


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(features: Path, events: Path, output: Path) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise ValueError("output must be new and immutable")
    output.mkdir(parents=True)

    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)

    # The deployable development package must not be influenced by the already-
    # inspected monitor. Remove it before model fitting or replay verification.
    monitor = roles == "monitor"
    monitor_rows_excluded = int(monitor.sum())
    if monitor_rows_excluded:
        keep = ~monitor
        frame = frame.loc[keep].reset_index(drop=True)
        y = np.asarray(y)[keep]
        event_ids = np.asarray(event_ids)[keep]
        roles = np.asarray(roles)[keep]
        units = np.asarray(units)[keep]
    if np.any(roles == "monitor"):
        raise ValueError("monitor exclusion failed")
    if np.any(roles == "locked_test"):
        raise ValueError("locked test is forbidden")

    clean = transfer.build_clean_model(frame, y, roles, units, base, xrs, proton)
    stack = clean["stack"].diagnostics()
    feature_families = {"SOLAR": list(base), "XRS": list(xrs), "PROTON": list(proton)}
    family_models = {
        "SOLAR": clean["models"]["solar"],
        "XRS": clean["models"]["xrs"],
        "PROTON": clean["models"]["proton"],
    }
    source_bindings = {
        "feature_table_sha256": digest(features),
        "event_catalogue_sha256": digest(events),
        "architecture_config": "config/current_development_architecture_v1.json",
        "architecture": "IRIS_CROSSFIT_EVIDENCE_STACK_V1",
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    role_counts = {role: int(np.sum(roles == role)) for role in ("fit", "calibration", "threshold", "score")}
    training_receipt = {
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "fit_rows": role_counts["fit"],
        "fit_positives": int(y[roles == "fit"].sum()),
        "calibration_rows": role_counts["calibration"],
        "threshold_rows": role_counts["threshold"],
        "score_rows_for_replay_only": role_counts["score"],
        "score_labels_used_for_fitting": False,
        "positive_event_units": int(positive_units),
        "purged_units": purged,
        "dropped_non_numeric_columns": dropped,
    }
    package_dir = output / "model_package"
    manifest = export_promoted_package(
        output_dir=package_dir,
        family_models=family_models,
        feature_families=feature_families,
        fit_prevalence=clean["fit_prevalence"],
        stack_intercept=stack["intercept"],
        stack_weights=stack["weights"],
        evidence_limit=cf.EVIDENCE_LIMIT,
        calibration_intercept=clean["calibration_intercept"],
        thresholds=clean["thresholds"],
        source_bindings=source_bindings,
        dependency_versions=versions,
        training_receipt=training_receipt,
    )

    loaded = load_promoted_package(package_dir)
    replay = loaded.predict(frame)
    reference = np.asarray(clean["clean_probability"], dtype=np.float64)
    candidate = np.asarray(replay["probability"], dtype=np.float64)
    max_abs = float(np.max(np.abs(reference - candidate)))
    if max_abs > REPLAY_TOLERANCE:
        raise ValueError(f"load-only replay exceeds tolerance: {max_abs}")

    score = roles == "score"
    mismatches = {}
    for policy, threshold in clean["thresholds"].items():
        reference_decision = reference[score] >= float(threshold)
        loaded_decision = candidate[score] >= float(loaded.thresholds[policy])
        mismatches[policy] = int(np.sum(reference_decision != loaded_decision))
        if mismatches[policy]:
            raise ValueError(f"load-only {policy} decision mismatch: {mismatches[policy]}")

    reference_table = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": roles,
        "unit_id": units,
        "label": y,
        "in_memory_probability": reference,
        "load_only_probability": candidate,
    })
    reference_table.to_csv(output / "reference_predictions.csv", index=False, float_format="%.17g")

    receipt = {
        "status": "PROMOTED_MODEL_PACKAGE_BUILT_AND_LOAD_ONLY_REPLAY_VERIFIED",
        "scope": "DEVELOPMENT_MODEL_PACKAGE_NOT_OPERATIONALLY_CERTIFIED",
        "locked_test_accessed": False,
        "monitor_used": False,
        "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
        "runtime_training_allowed": False,
        "specialist_model_count": 15,
        "model_families": ["SOLAR", "XRS", "PROTON"],
        "manifest_sha256": sha256_file(package_dir / "manifest.json"),
        "package_receipt_sha256": sha256_file(package_dir / "package_receipt.json"),
        "reference_predictions_sha256": sha256_file(output / "reference_predictions.csv"),
        "load_only_max_abs_probability_difference": max_abs,
        "load_only_score_decision_mismatches": mismatches,
        "replay_tolerance": REPLAY_TOLERANCE,
        "thresholds": manifest["thresholds"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "dependency_versions": versions,
        "claim_boundary": "This proves deterministic package serialization/reload on the development interface. It does not establish fresh forecast skill or operational certification.",
    }
    save_json(output / "build_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.output), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
