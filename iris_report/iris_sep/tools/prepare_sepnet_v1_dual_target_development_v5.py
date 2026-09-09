"""Fail-closed V1 dual-target development cohort bound to frozen v3 roles.

This builder reads the pinned publisher training CSV plus the pinned v3
development mapping.  It never reads a publisher testing file or SEPVAL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from iris_report.iris_sep.tools import prepare_sepnet_v1_development as v3_builder
from iris_report.iris_sep.tools import prepare_sepnet_v1_dual_target_development as v4_builder
from iris_report.iris_sep.workstreams.luna_g_data_pipeline.iris_sep_pipeline import cohort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_V3_CSV = PROJECT_ROOT / "data_processed" / "sepnet_v1_development_v3.csv"
PINNED_V3_MANIFEST = PROJECT_ROOT / "receipts" / "sepnet_v1_development_v3_manifest.json"
EXPECTED_V3_CSV_SHA256 = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
EXPECTED_V3_MANIFEST_SHA256 = "18c10d4fc76a2ce5e03b9a271951003f274435aa00180fcb90e4f2947eedaebb"
MAPPING_COLUMNS = (
    "issue_id",
    "role",
    "unit_id",
    "window_begin",
    "window_end",
    v4_builder.OPERATIONAL_TARGET,
)
TARGET_SCHEMA = (
    {"name": v4_builder.GENERAL_TARGET, "role": "training_target", "dtype": "binary"},
    {"name": v4_builder.OPERATIONAL_TARGET, "role": "evaluation_calibration_threshold_target", "dtype": "binary"},
    {"name": v4_builder.MAX_FLUX_TARGET, "role": "continuous_audit_target", "dtype": "nonnegative_finite_float"},
)


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping_bytes(frame: pd.DataFrame) -> bytes:
    return frame.loc[:, list(MAPPING_COLUMNS)].to_csv(index=False, lineterminator="\n").encode("utf-8")


def _assert_v3_mapping(candidate: pd.DataFrame, frozen: pd.DataFrame) -> str:
    """Require exact row order and exact values for all frozen mapping fields."""
    missing = [column for column in MAPPING_COLUMNS if column not in candidate or column not in frozen]
    if missing:
        raise ValueError(f"frozen v3 mapping columns missing: {missing}")
    left = candidate.loc[:, list(MAPPING_COLUMNS)].reset_index(drop=True)
    right = frozen.loc[:, list(MAPPING_COLUMNS)].reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise ValueError("candidate does not have the exact ordered frozen v3 mapping schema and length")
    for column in MAPPING_COLUMNS:
        if left[column].tolist() != right[column].tolist():
            raise ValueError(f"candidate differs from frozen v3 mapping in {column}")
    left_hash = hashlib.sha256(_mapping_bytes(left)).hexdigest()
    right_hash = hashlib.sha256(_mapping_bytes(right)).hexdigest()
    if left_hash != right_hash:
        raise ValueError("candidate and frozen v3 mapping hashes differ")
    return right_hash


def _assert_strict_role_purge(frame: pd.DataFrame) -> None:
    """Reject an adjacent role beginning at or before the inclusive endpoint."""
    parsed = frame.copy()
    parsed["window_end"] = pd.to_datetime(parsed["window_end"], utc=True, errors="raise")
    previous_end = None
    for role in v4_builder.ROLES:
        role_rows = parsed.loc[parsed["role"] == role]
        if role_rows.empty:
            raise ValueError(f"frozen role {role} is empty")
        role_start = role_rows["window_end"].min()
        if previous_end is not None and role_start <= previous_end + pd.Timedelta(hours=24):
            raise ValueError(f"strict inclusive 24-hour purge violated before {role}")
        previous_end = role_rows["window_end"].max()


def _assert_valid_max_flux(frame: pd.DataFrame) -> None:
    max_flux = pd.to_numeric(frame[v4_builder.MAX_FLUX_TARGET], errors="raise")
    finite = max_flux.map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf"))
    if not finite.all():
        raise ValueError("future_SEP_MaxFlux must contain only finite values")
    if (max_flux < 0).any():
        raise ValueError("future_SEP_MaxFlux must be nonnegative")


def _assert_distinct_safe_destinations(output_csv: Path, manifest_path: Path) -> None:
    output_resolved = output_csv.expanduser().resolve(strict=False)
    manifest_resolved = manifest_path.expanduser().resolve(strict=False)
    if output_resolved == manifest_resolved:
        raise ValueError("output and manifest paths alias the same destination")
    protected = {
        v4_builder.PINNED_SOURCE.resolve(),
        PINNED_V3_CSV.resolve(),
        PINNED_V3_MANIFEST.resolve(),
    }
    if output_resolved in protected or manifest_resolved in protected:
        raise ValueError("output paths may not alias a pinned input artifact")
    if output_csv.exists() or manifest_path.exists():
        raise ValueError("v5 dual-target outputs are immutable; choose unused paths")


def _atomic_exclusive_write(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically, failing if the destination exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic and exclusive: EEXIST never overwrites
    finally:
        temporary.unlink(missing_ok=True)


def prepare(output_csv: Path, manifest_path: Path) -> dict[str, object]:
    _assert_distinct_safe_destinations(output_csv, manifest_path)
    if v3_builder.sha256_file(PINNED_V3_CSV) != EXPECTED_V3_CSV_SHA256:
        raise ValueError("pinned v3 development CSV hash mismatch")
    if v3_builder.sha256_file(PINNED_V3_MANIFEST) != EXPECTED_V3_MANIFEST_SHA256:
        raise ValueError("pinned v3 development manifest hash mismatch")

    with tempfile.TemporaryDirectory(prefix="iris-sep-v5-stage-") as temporary_directory:
        staging = Path(temporary_directory)
        staged_csv = staging / "candidate.csv"
        staged_manifest = staging / "candidate-manifest.json"
        v4_builder.prepare(staged_csv, staged_manifest)
        candidate = pd.read_csv(staged_csv, float_precision="round_trip")

    frozen = pd.read_csv(PINNED_V3_CSV, float_precision="round_trip")
    mapping_sha256 = _assert_v3_mapping(candidate, frozen)
    _assert_strict_role_purge(candidate)
    _assert_valid_max_flux(candidate)

    feature_columns = list(candidate.columns[8:])
    if len(feature_columns) != v4_builder.EXPECTED_FEATURE_COUNT:
        raise ValueError("candidate ordered causal feature schema is not exactly 98 columns")
    feature_schema_sha256 = _json_hash(feature_columns)
    target_schema_sha256 = _json_hash(TARGET_SCHEMA)
    output_bytes = candidate.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()

    dependency_paths = {
        "v5_tool": Path(__file__),
        "v4_dual_target_builder": Path(v4_builder.__file__),
        "v3_prepare_and_build_units": Path(v3_builder.__file__),
        "cohort_assignment_implementation": Path(cohort.__file__),
    }
    source_hashes = {name: v3_builder.sha256_file(path) for name, path in dependency_paths.items()}
    manifest = {
        "status": "DEVELOPMENT_ONLY_LEGACY_DUAL_TARGET_NOT_FINAL_BENCHMARK",
        "source_kind": "PINNED_LOCAL_PUBLISHER_TRAINING_CSV_PLUS_FROZEN_V3_DEVELOPMENT_MAPPING",
        "publisher_training_source_repo_relative": "data_external/sepnet_v1/rolling_combinded_training.csv",
        "publisher_training_source_sha256": v4_builder.EXPECTED_SOURCE_SHA256,
        "frozen_v3_csv_repo_relative": "data_processed/sepnet_v1_development_v3.csv",
        "frozen_v3_csv_sha256": EXPECTED_V3_CSV_SHA256,
        "frozen_v3_manifest_repo_relative": "receipts/sepnet_v1_development_v3_manifest.json",
        "frozen_v3_manifest_sha256": EXPECTED_V3_MANIFEST_SHA256,
        "frozen_v3_mapping_columns": list(MAPPING_COLUMNS),
        "frozen_v3_mapping_sha256": mapping_sha256,
        "output_sha256": output_sha256,
        "rows": int(len(candidate)),
        "ordered_feature_columns": feature_columns,
        "ordered_feature_schema_sha256": feature_schema_sha256,
        "ordered_target_schema": list(TARGET_SCHEMA),
        "ordered_target_schema_sha256": target_schema_sha256,
        "source_code_sha256": source_hashes,
        "training_target": v4_builder.GENERAL_TARGET,
        "evaluation_calibration_threshold_target": v4_builder.OPERATIONAL_TARGET,
        "continuous_audit_target": v4_builder.MAX_FLUX_TARGET,
        "roles_derived_only_from": v4_builder.OPERATIONAL_TARGET,
        "split_policy": "EXACT_ORDERED_MAPPING_EQUALITY_WITH_PINNED_DEVELOPMENT_V3",
        "purge_hours": 24,
        "purge_boundary": "NEXT_ROLE_START_STRICTLY_GREATER_THAN_PRIOR_ROLE_INCLUSIVE_24H_ENDPOINT",
        "locked_test_rows_present": False,
        "testing_or_sepval_artifact_accessed": False,
        "allowed_uses": ["dual_target_development", "baseline_reproduction", "development_only_model_selection"],
        "forbidden_claims": ["final_new_crossing_score", "SEPVAL_score", "operational_certification", "breakthrough"],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    wrote_output = False
    try:
        _atomic_exclusive_write(output_csv, output_bytes)
        wrote_output = True
        _atomic_exclusive_write(manifest_path, manifest_bytes)
    except Exception:
        if wrote_output and not manifest_path.exists():
            output_csv.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data_processed" / "sepnet_v1_development_v5_dual_target.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "receipts" / "sepnet_v1_development_v5_dual_target_manifest.json")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.output, arguments.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
