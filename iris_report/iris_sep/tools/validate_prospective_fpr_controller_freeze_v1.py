"""Source-only validator for the prospective FPR-controller confirmation package.

This validator never loads protected outcomes, event identities, event counts, or
model scores. It only checks frozen contracts and an execution manifest. It must
fail closed until a custodian has truthfully completed every required field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "config" / "prospective_fpr_controller_confirmation_v1_preregistration_2026-09-12.json"
FREEZE = ROOT / "config" / "fpr_controller_v2_freeze_2026-09-12.json"
DEFAULT_MANIFEST = ROOT / "config" / "prospective_fpr_controller_execution_manifest_template_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contracts(prereg: dict[str, Any], freeze: dict[str, Any]) -> None:
    assert prereg["study_id"] == "IRIS_SEP_PROSPECTIVE_FPR_CONTROLLER_CONFIRMATION_V1"
    assert prereg["status"] == "FROZEN_NOT_EXECUTED_BLOCKED_ON_SOURCE_MODEL_AND_CUSTODIAN"
    assert prereg["protected_pool"]["development_side_outcome_access"] == "FORBIDDEN"
    assert prereg["protected_pool"]["development_side_event_count_access"] == "FORBIDDEN"
    assert prereg["protected_pool"]["development_side_episode_identity_access"] == "FORBIDDEN"
    assert prereg["controller_freeze"]["quantile"] == 0.80
    assert prereg["controller_freeze"]["history_window_days"] == 120
    assert prereg["controller_freeze"]["minimum_resolved_negative_history"] == 30
    assert prereg["controller_freeze"]["post_horizon_label_buffer_hours"] == 1
    assert prereg["information_floor"]["minimum_distinct_onset_episodes"] == 50
    assert prereg["information_floor"]["minimum_quiet_blocks"] == 500

    assert freeze["freeze_id"] == "IRIS_SEP_FPR_CONTROLLER_V2_FREEZE"
    assert freeze["status"] == "FROZEN_FOR_FUTURE_UNINSPECTED_EVALUATION_ONLY"
    assert freeze["frozen_policy"]["quantile"] == prereg["controller_freeze"]["quantile"]
    assert freeze["frozen_policy"]["history_window_days"] == prereg["controller_freeze"]["history_window_days"]
    assert freeze["frozen_policy"]["minimum_resolved_negative_history"] == prereg["controller_freeze"]["minimum_resolved_negative_history"]
    assert freeze["frozen_policy"]["post_horizon_label_buffer_hours"] == prereg["controller_freeze"]["post_horizon_label_buffer_hours"]
    assert freeze["no_more_2017_selection"]["controller_quantile_tuning"] == "FORBIDDEN_FOR_ANY_FRESH_CLAIM"


def missing_execution_fields(manifest: dict[str, Any]) -> list[str]:
    required = {
        "source_model.model_id": manifest["source_model"].get("model_id"),
        "source_model.model_artifact_sha256_or_deterministic_rule_sha256": manifest["source_model"].get("model_artifact_sha256_or_deterministic_rule_sha256"),
        "source_model.feature_schema_sha256": manifest["source_model"].get("feature_schema_sha256"),
        "source_model.frozen_fixed_threshold": manifest["source_model"].get("frozen_fixed_threshold"),
        "source_model.training_data_boundary": manifest["source_model"].get("training_data_boundary"),
        "source_model.software_environment_receipt": manifest["source_model"].get("software_environment_receipt"),
        "source_model.causal_feature_verification_receipt": manifest["source_model"].get("causal_feature_verification_receipt"),
        "protected_cohort.custodian_cohort_hash": manifest["protected_cohort"].get("custodian_cohort_hash"),
        "protected_cohort.cohort_end_selected_by_custodian": manifest["protected_cohort"].get("cohort_end_selected_by_custodian"),
        "protected_cohort.overlap_check_against_inspected_registry_v2": manifest["protected_cohort"].get("overlap_check_against_inspected_registry_v2"),
        "protected_cohort.overlap_check_against_inspected_registry_v3": manifest["protected_cohort"].get("overlap_check_against_inspected_registry_v3"),
        "custodian.name_or_identifier": manifest["custodian"].get("name_or_identifier"),
        "custodian.attestation_timestamp_utc": manifest["custodian"].get("attestation_timestamp_utc"),
        "custodian.independence_attestation": manifest["custodian"].get("independence_attestation"),
        "execution.execution_commit_sha": manifest["execution"].get("execution_commit_sha"),
        "execution.execution_environment_sha256": manifest["execution"].get("execution_environment_sha256"),
    }
    return [name for name, value in required.items() if value is None or value == ""]


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    assert manifest["study_id"] == "IRIS_SEP_PROSPECTIVE_FPR_CONTROLLER_CONFIRMATION_V1"
    exposure = manifest["protected_cohort"]
    if exposure.get("development_side_outcomes_exposed_before_freeze") is not False:
        raise RuntimeError("protected outcomes were exposed before freeze")
    if exposure.get("development_side_event_counts_exposed_before_freeze") is not False:
        raise RuntimeError("protected event counts were exposed before freeze")
    if exposure.get("development_side_event_identities_exposed_before_freeze") is not False:
        raise RuntimeError("protected event identities were exposed before freeze")
    if exposure.get("development_side_model_scores_exposed_before_conclusion_freeze") is not False:
        raise RuntimeError("protected model scores were exposed before conclusion freeze")

    missing = missing_execution_fields(manifest)
    execution_frozen = manifest.get("manifest_status") == "EXECUTION_FROZEN"
    ready = bool(execution_frozen and not missing)
    return {
        "study_id": manifest["study_id"],
        "execution_ready": ready,
        "manifest_status": manifest.get("manifest_status"),
        "missing_required_fields": missing,
        "protected_outcome_access_performed_by_validator": False,
        "status": "READY_FOR_CUSTODIAN_EXECUTION" if ready else "BLOCKED_FAIL_CLOSED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    prereg = load_json(PREREG)
    freeze = load_json(FREEZE)
    manifest = load_json(args.manifest)
    validate_contracts(prereg, freeze)
    result = validate_manifest(manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["execution_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
