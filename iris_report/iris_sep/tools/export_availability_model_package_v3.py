"""Export and self-replay the frozen IRIS-SEP distilled availability V3 package.

Development packaging only. The architecture and hyperparameters were already
preregistered and evaluated by the V3 replay. This tool does not access locked
test data, search hyperparameters or grant NORMAL trust. It serializes the V3
student/teacher state stacks plus the 15 family specialists and verifies that a
fresh load-only replay reproduces the in-memory V3 probabilities.
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

from iris_report.iris_sep.src.iris_sep.model_package import (
    PACKAGE_FORMAT_V2,
    STATE_EXPERTS,
    STATE_OPERATOR_PERMISSION,
    STATE_STACK_KIND_V3,
    V3_ARCHITECTURE,
    LoadedAvailabilityPackage,
    expected_state_feature_schema,
    expected_state_feature_schema_sha256,
    sha256_file,
)
from iris_report.iris_sep.tools import run_availability_distilled_fallback_v3 as v3
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_crossfit_evidence_stack_diagnostic as cf
from iris_report.iris_sep.tools import run_promoted_stack_missingness_transfer as transfer
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as public_benchmark


SEEDS = list(public_benchmark.SEEDS)
V3_REPLAY_RUN = 34138370215
V3_REPLAY_HEAD = "abe39c72d5ed6fb1070a0601b158117a487cb19f"
V3_REPLAY_ARTIFACT_SHA256 = "0d6c38cc34d1d55604bade0611f37d5635620ad6112d35a9f193ef51b39b2390"
REPLAY_TOLERANCE = 1e-10


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _model_entries(output: Path, family: str, models) -> list[dict[str, object]]:
    entries = []
    if len(models) != len(SEEDS):
        raise ValueError(f"unexpected {family} specialist count")
    for seed, model in zip(SEEDS, models):
        relative = Path("models") / f"{family}_seed_{seed}.json"
        destination = output / relative
        model.get_booster().save_model(str(destination))
        entries.append({
            "seed": int(seed),
            "path": relative.as_posix(),
            "sha256": sha256_file(destination),
        })
    return entries


def _stack_payload(model, *, kind: str) -> dict[str, object]:
    diagnostics = model.diagnostics()
    payload = {
        "intercept": float(diagnostics["intercept"]),
        "weights": [float(value) for value in diagnostics["weights"]],
        "l2_weight": float(diagnostics["l2_weight"]),
    }
    if kind == "DISTILLED_POSITIVE_EVIDENCE_STACK":
        payload.update({
            "teacher_weight": float(diagnostics["teacher_weight"]),
            "hard_label_weight": float(diagnostics["hard_label_weight"]),
            "hard_bce": float(diagnostics["hard_bce"]),
            "teacher_bce": float(diagnostics["teacher_bce"]),
        })
    return payload


def run(features: Path, events: Path, output: Path) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise ValueError("output directory must be new and immutable")
    output.mkdir(parents=True)
    (output / "models").mkdir()

    root = Path(__file__).resolve().parents[1]
    prereg = root / v3.PREREG
    if not prereg.is_file():
        raise ValueError("distilled V3 preregistration missing")

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

    fitted = v3._fit_distilled_states(frame, y, roles, units, base, xrs, proton)
    fit = roles == "fit"
    family_models = {
        "solar": transfer.fit_family_models(frame, base, y, fit),
        "xrs": transfer.fit_family_models(frame, xrs, y, fit),
        "proton": transfer.fit_family_models(frame, proton, y, fit),
    }
    feature_families = {
        "solar": list(base),
        "xrs": list(xrs),
        "proton": list(proton),
    }
    model_files = {
        family: _model_entries(output, family, family_models[family])
        for family in ("solar", "xrs", "proton")
    }

    states: dict[str, dict[str, object]] = {}
    for state in STATE_EXPERTS:
        kind = STATE_STACK_KIND_V3[state]
        if state == "FULL":
            stack = _stack_payload(fitted["teacher"], kind=kind)
        elif state in v3.ONE_FEED_STATES:
            stack = _stack_payload(fitted["students"][state], kind=kind)
        else:
            stack = None
        states[state] = {
            "experts": list(STATE_EXPERTS[state]),
            "stack_kind": kind,
            "stack": stack,
            "calibration_intercept": float(fitted["calibration_intercept"][state]),
            "thresholds": {
                key: float(value)
                for key, value in fitted["thresholds"][state].items()
            },
        }

    state_schemas = {
        state: {
            "schema": expected_state_feature_schema(feature_families, state),
            "sha256": expected_state_feature_schema_sha256(feature_families, state),
        }
        for state in STATE_EXPERTS
    }

    manifest = {
        "format": PACKAGE_FORMAT_V2,
        "status": "DEVELOPMENT_ONLY_RELOADABLE_DISTILLED_V3_PACKAGE",
        "target": public_benchmark.TARGET,
        "architecture": V3_ARCHITECTURE,
        "serialization": "XGBOOST_BOOSTER_JSON",
        "runtime_training_allowed": False,
        "source_evidence": {
            "v3_replay_run_id": V3_REPLAY_RUN,
            "v3_replay_head": V3_REPLAY_HEAD,
            "v3_replay_artifact_sha256": V3_REPLAY_ARTIFACT_SHA256,
        },
        "source_data": {
            "feature_table_sha256": v3.digest(features),
            "event_catalogue_sha256": v3.digest(events),
            "monitor_rows_excluded_before_modeling": monitor_rows_excluded,
            "locked_test_accessed": False,
        },
        "seeds": SEEDS,
        "feature_families": feature_families,
        "state_feature_schemas": state_schemas,
        "fit_prevalence": float(fitted["fit_prevalence"]),
        "evidence_limit": float(cf.EVIDENCE_LIMIT),
        "model_files": model_files,
        "states": states,
        "distillation": {
            "teacher_weight": float(v3.TEACHER_WEIGHT),
            "hard_label_weight": float(1.0 - v3.TEACHER_WEIGHT),
            "l2_weight": float(v3.L2_WEIGHT),
            "preregistration": v3.PREREG,
            "preregistration_sha256": v3.digest(prereg),
            "teacher_fit_uses_cross_fitted_evidence_only": True,
            "missing_expert_available_to_student_at_runtime": False,
        },
        "operator_permissions": dict(STATE_OPERATOR_PERMISSION),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "training_receipt": {
            "positive_event_units": int(positive_units),
            "purged_units": purged,
            "oof_rows": int(fitted["oof_rows"]),
            "oof_positives": int(fitted["oof_positives"]),
            "inner_folds": fitted["inner_folds"],
            "teacher_probability_range": [
                float(fitted["teacher_probability_min"]),
                float(fitted["teacher_probability_max"]),
            ],
            "runtime_retraining_supported": False,
            "runtime_recalibration_supported": False,
            "runtime_rethresholding_supported": False,
        },
        "claim_boundary": (
            "Development V3 package for deterministic replay and future untouched evaluation; "
            "not operational certification and does not grant NORMAL trust."
        ),
    }
    manifest_path = output / "manifest.json"
    save_json(manifest_path, manifest)

    # The V2 loader requires a receipt before it will load any models. Write the
    # trust anchor first, then self-replay the complete package below.
    receipt_path = output / "package_receipt.json"
    receipt = {
        "format": PACKAGE_FORMAT_V2,
        "status": "EXPORTED_DISTILLED_V3_AVAILABILITY_PACKAGE",
        "manifest_sha256": sha256_file(manifest_path),
        "specialist_model_count": 15,
        "serialization": "XGBOOST_BOOSTER_JSON",
        "runtime_training_allowed": False,
        "locked_test_accessed": False,
        "monitor_used": False,
    }
    save_json(receipt_path, receipt)

    loaded = LoadedAvailabilityPackage.load(output)
    replay_frame = frame.loc[:, list(base) + list(xrs) + list(proton)].copy()
    max_abs_by_state: dict[str, float] = {}
    for state in STATE_EXPERTS:
        replay_probability = loaded.predict(replay_frame, state=state)
        reference_probability = np.asarray(fitted["probability"][state], dtype=np.float64)
        maximum = float(np.max(np.abs(replay_probability - reference_probability)))
        max_abs_by_state[state] = maximum
        if maximum > REPLAY_TOLERANCE:
            raise ValueError(
                f"load-only V3 package replay mismatch for {state}: max abs diff {maximum}"
            )

    replay = frame.loc[:, ["window_end"] + list(base) + list(xrs) + list(proton)].copy()
    replay.rename(columns={"window_end": "issue_time"}, inplace=True)
    replay_path = output / "replay_input.csv"
    replay.to_csv(replay_path, index=False)

    reference = pd.DataFrame({
        "issue_time": frame["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    })
    for state in STATE_EXPERTS:
        reference[f"p_{state}"] = fitted["probability"][state]
    reference_path = output / "export_reference_predictions.csv"
    reference.to_csv(reference_path, index=False, float_format="%.17g")

    receipt.update({
        "replay_input_sha256": sha256_file(replay_path),
        "reference_predictions_sha256": sha256_file(reference_path),
        "self_replay_tolerance": REPLAY_TOLERANCE,
        "self_replay_max_abs_difference_by_state": max_abs_by_state,
        "self_replay_passed": True,
    })
    save_json(receipt_path, receipt)
    # Re-loading after receipt augmentation verifies the manifest trust anchor
    # remains stable and the updated receipt is still admissible.
    LoadedAvailabilityPackage.load(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
