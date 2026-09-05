"""Build the immutable SEPNET V1 dual-target development cohort.

Only the pinned publisher *training* CSV is accepted.  The general SEP label is
available for model fitting, while the stricter operational label is retained
for evaluation, calibration, and threshold selection.  This adapter is not the
final new-crossing benchmark and must never be used for a SEPVAL claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from iris_report.iris_sep.tools.prepare_sepnet_v1_development import (
    EXPECTED_SOURCE_SHA256,
    ROLES,
    TARGET as OPERATIONAL_TARGET,
    build_units,
    sha256_file,
)
from iris_report.iris_sep.workstreams.luna_g_data_pipeline.iris_sep_pipeline.cohort import (
    assign_chronological_roles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE = PROJECT_ROOT / "data_external" / "sepnet_v1" / "rolling_combinded_training.csv"
GENERAL_TARGET = "future_SEP_label"
MAX_FLUX_TARGET = "future_SEP_MaxFlux"
TARGET_COLUMNS = (GENERAL_TARGET, OPERATIONAL_TARGET, MAX_FLUX_TARGET)
EXPECTED_FEATURE_COUNT = 98


def _issue_id(row: object) -> str:
    payload = f"{row.window_begin.isoformat()}|{row.window_end.isoformat()}".encode()
    return hashlib.sha256(payload).hexdigest()


def prepare(output_csv: Path, manifest_path: Path) -> dict[str, object]:
    """Create a dual-target cohort without accepting an arbitrary data source."""
    source = PINNED_SOURCE.resolve(strict=True)
    expected_source = PINNED_SOURCE.resolve()
    if source != expected_source or source.name != "rolling_combinded_training.csv":
        raise ValueError("only the pinned local SEPNET V1 publisher training CSV is permitted")
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source hash does not match the pinned publisher training file")
    if output_csv.exists() or manifest_path.exists():
        raise ValueError("dual-target development outputs are immutable; choose new paths")

    frame = pd.read_csv(source)
    required = {"window_begin", "window_end", *TARGET_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError("publisher training source is missing required dual-target columns")
    frame["window_begin"] = pd.to_datetime(frame["window_begin"], utc=True, errors="raise")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="raise")
    if frame.duplicated(["window_begin", "window_end"]).any():
        raise ValueError("duplicate canonical UTC windows")
    if not (frame["window_end"] - frame["window_begin"] == pd.Timedelta(hours=24)).all():
        raise ValueError("every predictor window must be exactly 24 hours")
    for target in (GENERAL_TARGET, OPERATIONAL_TARGET):
        if frame[target].isna().any() or not frame[target].isin([0, 1]).all():
            raise ValueError(f"{target} must be complete and binary")
    if (frame[OPERATIONAL_TARGET] > frame[GENERAL_TARGET]).any():
        raise ValueError("operational positives must be a subset of general SEP positives")
    if not pd.api.types.is_numeric_dtype(frame[MAX_FLUX_TARGET]):
        raise ValueError("future_SEP_MaxFlux must be numeric")
    if frame[MAX_FLUX_TARGET].isna().any() or (frame[MAX_FLUX_TARGET] < 0).any():
        raise ValueError("future_SEP_MaxFlux must be complete and nonnegative")

    frame = frame.sort_values(["window_end", "window_begin"]).reset_index(drop=True)
    cadence_hours = frame["window_end"].diff().dropna().dt.total_seconds().div(3600)
    if (cadence_hours <= 0).any() or ((cadence_hours % 24) != 0).any():
        raise ValueError("publisher training windows must have positive daily cadence with allowed gaps")

    future_columns = [column for column in frame.columns if column.lower().startswith("future_")]
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"window_begin", "window_end"}
        and not column.lower().startswith("future_")
    ]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"expected exactly {EXPECTED_FEATURE_COUNT} causal predictors")
    if any(column.lower().startswith("future_") for column in feature_columns):
        raise ValueError("future outcome leaked into causal predictors")

    # Deliberately use the operational target for the same episode units and
    # quotas as development_v3.  The general target never influences roles.
    units = build_units(frame)
    positive_units = sum(unit.label for unit in units)
    cutoffs = (int(positive_units * 0.70), int(positive_units * 0.80), int(positive_units * 0.90))
    provisional_counts = {role: 0 for role in ROLES}
    positives_seen = 0
    for unit in units:
        if positives_seen < cutoffs[0]:
            role = "train"
        elif positives_seen < cutoffs[1]:
            role = "validation_monitor"
        elif positives_seen < cutoffs[2]:
            role = "validation_calibration"
        else:
            role = "validation_threshold"
        provisional_counts[role] += 1
        positives_seen += unit.label
    assigned = assign_chronological_roles(units, provisional_counts, purge_hours=24)

    row_roles: dict[int, tuple[str, str]] = {}
    for role in ROLES:
        for unit in assigned[role]:
            for raw_index in unit.issue_ids:
                row_roles[int(raw_index)] = (role, unit.unit_id)
    selected = frame.loc[sorted(row_roles)].copy()
    selected.insert(0, "issue_id", [_issue_id(row) for row in selected.itertuples()])
    selected.insert(1, "role", [row_roles[int(index)][0] for index in selected.index])
    selected.insert(2, "unit_id", [row_roles[int(index)][1] for index in selected.index])
    selected = selected[
        ["issue_id", "role", "unit_id", "window_begin", "window_end", *TARGET_COLUMNS, *feature_columns]
    ]
    for role in ROLES:
        role_frame = selected.loc[selected["role"] == role]
        if role_frame.empty:
            raise ValueError(f"role {role} is empty")
        for target in (GENERAL_TARGET, OPERATIONAL_TARGET):
            if role_frame[target].nunique() != 2:
                raise ValueError(f"role {role} lacks both classes for {target}")
    selected["window_begin"] = selected["window_begin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    selected["window_end"] = selected["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False, float_format="%.17g")

    role_counts = {}
    for role in ROLES:
        role_frame = selected.loc[selected["role"] == role]
        role_counts[role] = {
            "rows": int(len(role_frame)),
            "units": int(role_frame["unit_id"].nunique()),
            "general_positive_windows": int(role_frame[GENERAL_TARGET].sum()),
            "operational_positive_windows": int(role_frame[OPERATIONAL_TARGET].sum()),
        }
    label_contingency = {
        f"general_{general}_operational_{operational}": int(count)
        for (general, operational), count in selected.groupby(
            [GENERAL_TARGET, OPERATIONAL_TARGET], dropna=False
        ).size().items()
    }
    manifest = {
        "status": "DEVELOPMENT_ONLY_LEGACY_DUAL_TARGET_NOT_FINAL_BENCHMARK",
        "source_kind": "PINNED_LOCAL_PUBLISHER_TRAINING_CSV_ONLY",
        "source_path_repo_relative": "data_external/sepnet_v1/rolling_combinded_training.csv",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256_file(output_csv),
        "feature_columns": feature_columns,
        "feature_column_count": len(feature_columns),
        "training_target": GENERAL_TARGET,
        "evaluation_target": OPERATIONAL_TARGET,
        "continuous_audit_target": MAX_FLUX_TARGET,
        "target_semantics": {
            GENERAL_TARGET: "PUBLISHER_LEGACY_GENERAL_FUTURE_SEP_WINDOW_LABEL",
            OPERATIONAL_TARGET: "PUBLISHER_LEGACY_FUTURE_OPERATIONAL_WINDOW_LABEL_NOT_AUDITED_NEW_CROSSING",
            MAX_FLUX_TARGET: "PUBLISHER_LEGACY_FUTURE_WINDOW_MAXIMUM_FLUX",
        },
        "roles_derived_only_from": OPERATIONAL_TARGET,
        "excluded_other_future_columns": sorted(set(future_columns) - set(TARGET_COLUMNS)),
        "quiet_block_days": 7,
        "purge_hours": 24,
        "purge_boundary": "STRICTLY_GREATER_THAN_INCLUSIVE_24H_TARGET_ENDPOINT",
        "cadence_contract": "POSITIVE_MULTIPLES_OF_24_HOURS_WITH_GAPS_ALLOWED",
        "split_policy": "IDENTICAL_TO_DEVELOPMENT_V3_CHRONOLOGICAL_70_10_10_10_OPERATIONAL_POSITIVE_EPISODE_QUOTAS",
        "positive_episode_units_before_purge": positive_units,
        "purged_units": len(assigned["purged"]),
        "role_counts": role_counts,
        "label_contingency_selected": label_contingency,
        "locked_test_rows_present": False,
        "test_or_sepval_source_accessed": False,
        "allowed_uses": ["dual_target_development", "baseline_reproduction", "development_only_model_selection"],
        "forbidden_claims": ["final_new_crossing_score", "SEPVAL_score", "operational_certification", "breakthrough"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "sepnet_v1_development_v4_dual_target.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "receipts" / "sepnet_v1_development_v4_dual_target_manifest.json",
    )
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.output, arguments.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
