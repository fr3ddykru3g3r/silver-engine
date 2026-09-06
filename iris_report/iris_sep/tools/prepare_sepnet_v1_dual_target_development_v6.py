"""Auditable V1 dual-target cohort with pure validation and safe publication.

Production entrypoints are fixed to the pinned local publisher training source
and frozen v3 mapping.  Pure validators accept synthetic frames solely so the
negative contracts can be tested without touching testing data or SEPVAL.
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
from iris_report.iris_sep.tools import prepare_sepnet_v1_dual_target_development_v5 as v5_builder
from iris_report.iris_sep.workstreams.luna_g_data_pipeline.iris_sep_pipeline import cohort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURE_SCHEMA_SHA256 = "7bca82f223f1be0adbd8afc6e30aed238ed52b3bb2339a98fa9c9cbd944436b5"
MAPPING_COLUMNS = v5_builder.MAPPING_COLUMNS
TARGET_SCHEMA = v5_builder.TARGET_SCHEMA


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _read_source_with_hash(path: Path, expected_sha256: str) -> pd.DataFrame:
    """Controlled seam for hash-failure tests; production passes only PINNED_SOURCE."""
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("publisher training source hash mismatch")
    return pd.read_csv(path)


def _validate_source_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = raw.copy()
    required = {"window_begin", "window_end", *v4_builder.TARGET_COLUMNS}
    missing_targets = sorted(required - set(frame.columns))
    if missing_targets:
        raise ValueError(f"publisher training source missing required targets or windows: {missing_targets}")
    frame["window_begin"] = pd.to_datetime(frame["window_begin"], utc=True, errors="raise")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="raise")
    if frame.duplicated(["window_begin", "window_end"]).any():
        raise ValueError("duplicate canonical UTC windows")
    if not (frame["window_end"] - frame["window_begin"] == pd.Timedelta(hours=24)).all():
        raise ValueError("every predictor window must be exactly 24 hours")
    for target in (v4_builder.GENERAL_TARGET, v4_builder.OPERATIONAL_TARGET):
        if frame[target].isna().any() or not frame[target].isin([0, 1]).all():
            raise ValueError(f"{target} must be complete and binary")
    if (frame[v4_builder.OPERATIONAL_TARGET] > frame[v4_builder.GENERAL_TARGET]).any():
        raise ValueError("operational positives must be a subset of general SEP positives")
    v5_builder._assert_valid_max_flux(frame)
    frame = frame.sort_values(["window_end", "window_begin"]).reset_index(drop=True)
    cadence = frame["window_end"].diff().dropna().dt.total_seconds().div(3600)
    if (cadence <= 0).any() or ((cadence % 24) != 0).any():
        raise ValueError("publisher training windows must have positive daily cadence with allowed gaps")
    features = [
        column for column in frame.columns
        if column not in {"window_begin", "window_end"} and not column.lower().startswith("future_")
    ]
    if len(features) != v4_builder.EXPECTED_FEATURE_COUNT or _json_hash(features) != EXPECTED_FEATURE_SCHEMA_SHA256:
        raise ValueError("ordered causal predictor schema must exactly match the pinned 98-column schema")
    return frame, features


def _assert_role_classes(frame: pd.DataFrame) -> None:
    for role in v4_builder.ROLES:
        role_rows = frame.loc[frame["role"] == role]
        if role_rows.empty:
            raise ValueError(f"role {role} is empty")
        for target in (v4_builder.GENERAL_TARGET, v4_builder.OPERATIONAL_TARGET):
            if role_rows[target].nunique(dropna=False) != 2:
                raise ValueError(f"role {role} lacks both classes for {target}")


def _mapping_bytes(frame: pd.DataFrame) -> bytes:
    return frame.loc[:, list(MAPPING_COLUMNS)].to_csv(index=False, lineterminator="\n").encode()


def _assert_candidate(candidate: pd.DataFrame, frozen: pd.DataFrame) -> tuple[str, list[str]]:
    mapping_hash = v5_builder._assert_v3_mapping(candidate, frozen)
    v5_builder._assert_strict_role_purge(candidate)
    v5_builder._assert_valid_max_flux(candidate)
    _assert_role_classes(candidate)
    features = list(candidate.columns[8:])
    if len(features) != v4_builder.EXPECTED_FEATURE_COUNT or _json_hash(features) != EXPECTED_FEATURE_SCHEMA_SHA256:
        raise ValueError("candidate feature schema differs from pinned ordered schema")
    return mapping_hash, features


def _assert_destinations(output_csv: Path, manifest_path: Path) -> None:
    v5_builder._assert_distinct_safe_destinations(output_csv, manifest_path)


def _atomic_exclusive_publish(path: Path, payload: bytes) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        published = path.stat(follow_symlinks=False)
        return published.st_dev, published.st_ino, _sha256_bytes(payload)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_only_own_publication(path: Path, identity: tuple[int, int, str]) -> bool:
    """Best-effort rollback that refuses to unlink a replaced/raced artifact."""
    expected_device, expected_inode, expected_hash = identity
    try:
        before = path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (expected_device, expected_inode):
            return False
        if v3_builder.sha256_file(path) != expected_hash:
            return False
        after = path.stat(follow_symlinks=False)
        if (after.st_dev, after.st_ino) != (expected_device, expected_inode):
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def prepare(output_csv: Path, manifest_path: Path) -> dict[str, object]:
    _assert_destinations(output_csv, manifest_path)
    source = v4_builder.PINNED_SOURCE.resolve(strict=True)
    raw = _read_source_with_hash(source, v4_builder.EXPECTED_SOURCE_SHA256)
    _, source_features = _validate_source_frame(raw)
    if v3_builder.sha256_file(v5_builder.PINNED_V3_CSV) != v5_builder.EXPECTED_V3_CSV_SHA256:
        raise ValueError("pinned v3 development CSV hash mismatch")
    if v3_builder.sha256_file(v5_builder.PINNED_V3_MANIFEST) != v5_builder.EXPECTED_V3_MANIFEST_SHA256:
        raise ValueError("pinned v3 development manifest hash mismatch")
    with tempfile.TemporaryDirectory(prefix="iris-sep-v6-stage-") as directory:
        staged_csv = Path(directory) / "candidate.csv"
        staged_manifest = Path(directory) / "candidate.json"
        v4_builder.prepare(staged_csv, staged_manifest)
        candidate = pd.read_csv(staged_csv, float_precision="round_trip")
    frozen = pd.read_csv(v5_builder.PINNED_V3_CSV, float_precision="round_trip")
    mapping_hash, features = _assert_candidate(candidate, frozen)
    if features != source_features:
        raise ValueError("candidate and publisher source ordered predictor schemas differ")

    output_bytes = candidate.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    dependency_paths = {
        "v6_tool": Path(__file__),
        "v5_tool": Path(v5_builder.__file__),
        "v4_dual_target_builder": Path(v4_builder.__file__),
        "v3_prepare_and_build_units": Path(v3_builder.__file__),
        "cohort_assignment_implementation": Path(cohort.__file__),
    }
    manifest = {
        "status": "DEVELOPMENT_ONLY_LEGACY_DUAL_TARGET_NOT_FINAL_BENCHMARK",
        "source_kind": "PINNED_LOCAL_PUBLISHER_TRAINING_CSV_PLUS_FROZEN_V3_DEVELOPMENT_MAPPING",
        "publisher_training_source_repo_relative": "data_external/sepnet_v1/rolling_combinded_training.csv",
        "publisher_training_source_sha256": v4_builder.EXPECTED_SOURCE_SHA256,
        "frozen_v3_csv_repo_relative": "data_processed/sepnet_v1_development_v3.csv",
        "frozen_v3_csv_sha256": v5_builder.EXPECTED_V3_CSV_SHA256,
        "frozen_v3_manifest_repo_relative": "receipts/sepnet_v1_development_v3_manifest.json",
        "frozen_v3_manifest_sha256": v5_builder.EXPECTED_V3_MANIFEST_SHA256,
        "frozen_v3_mapping_columns": list(MAPPING_COLUMNS),
        "frozen_v3_mapping_sha256": mapping_hash,
        "output_sha256": _sha256_bytes(output_bytes),
        "rows": len(candidate),
        "ordered_feature_columns": features,
        "ordered_feature_schema_sha256": _json_hash(features),
        "ordered_target_schema": list(TARGET_SCHEMA),
        "ordered_target_schema_sha256": _json_hash(TARGET_SCHEMA),
        "source_code_sha256": {name: v3_builder.sha256_file(path) for name, path in dependency_paths.items()},
        "training_target": v4_builder.GENERAL_TARGET,
        "evaluation_calibration_threshold_target": v4_builder.OPERATIONAL_TARGET,
        "continuous_audit_target": v4_builder.MAX_FLUX_TARGET,
        "roles_derived_only_from": v4_builder.OPERATIONAL_TARGET,
        "split_policy": "EXACT_ORDERED_MAPPING_EQUALITY_WITH_PINNED_DEVELOPMENT_V3",
        "purge_boundary": "NEXT_ROLE_START_STRICTLY_GREATER_THAN_PRIOR_ROLE_INCLUSIVE_24H_ENDPOINT",
        "locked_test_rows_present": False,
        "testing_or_sepval_artifact_accessed": False,
        "allowed_uses": ["dual_target_development", "baseline_reproduction", "development_only_model_selection"],
        "forbidden_claims": [
            "final_new_crossing_score", "SEPVAL_score", "superiority", "breakthrough",
            "operational_certification", "production_readiness",
        ],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    publication = _atomic_exclusive_publish(output_csv, output_bytes)
    try:
        _atomic_exclusive_publish(manifest_path, manifest_bytes)
    except Exception:
        _remove_only_own_publication(output_csv, publication)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data_processed" / "sepnet_v1_development_v6_dual_target.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "receipts" / "sepnet_v1_development_v6_dual_target_manifest.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
