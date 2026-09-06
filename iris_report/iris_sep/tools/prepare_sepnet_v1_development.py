"""Prepare the publisher-separated SEPNET V1 training split for development.

This is a legacy-target baseline/development adapter. It must never be confused
with the final new-crossing SEPVAL evaluation cohort.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd

from iris_report.iris_sep.workstreams.luna_g_data_pipeline.iris_sep_pipeline.cohort import (
    assign_chronological_roles,
)
from iris_report.iris_sep.workstreams.luna_g_data_pipeline.iris_sep_pipeline.schemas import CohortUnit


EXPECTED_SOURCE_SHA256 = "59e9e659798798047728cf85a59f2e182dbbff87c5becdae068b27a5b9ed2454"
TARGET = "future_Operational_SEP_label"
ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_id(kind: str, start: pd.Timestamp) -> str:
    token = hashlib.sha256(start.isoformat().encode("ascii")).hexdigest()[:16]
    return f"{kind}-{token}"


def build_units(frame: pd.DataFrame, quiet_block_days: int = 7) -> list[CohortUnit]:
    units: list[CohortUnit] = []
    current_indices: list[int] = []
    current_label: int | None = None
    for index, row in frame.iterrows():
        label = int(row[TARGET])
        starts_new = False
        if current_indices:
            previous_time = frame.loc[current_indices[-1], "window_end"]
            gap = row["window_end"] - previous_time
            starts_new = (
                label != current_label
                or gap > pd.Timedelta(hours=36)
                or (label == 0 and len(current_indices) >= quiet_block_days)
            )
        if starts_new:
            first = frame.loc[current_indices[0], "window_end"]
            last = frame.loc[current_indices[-1], "window_end"]
            kind = "episode" if current_label == 1 else "quiet_block"
            units.append(CohortUnit(_unit_id(kind, first), kind, tuple(str(i) for i in current_indices), first.to_pydatetime(), last.to_pydatetime(), int(current_label)))
            current_indices = []
        current_indices.append(int(index)); current_label = label
    if current_indices:
        first = frame.loc[current_indices[0], "window_end"]
        last = frame.loc[current_indices[-1], "window_end"]
        kind = "episode" if current_label == 1 else "quiet_block"
        units.append(CohortUnit(_unit_id(kind, first), kind, tuple(str(i) for i in current_indices), first.to_pydatetime(), last.to_pydatetime(), int(current_label)))
    return units


def prepare(source: Path, output_csv: Path, manifest_path: Path) -> dict[str, object]:
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source hash does not match the pinned publisher training file")
    if output_csv.exists() or manifest_path.exists():
        raise ValueError("development outputs are immutable; choose new paths")
    frame = pd.read_csv(source)
    required = {"window_begin", "window_end", TARGET}
    if not required.issubset(frame.columns):
        raise ValueError("source is missing required columns")
    frame["window_begin"] = pd.to_datetime(frame["window_begin"], utc=True, errors="raise")
    frame["window_end"] = pd.to_datetime(frame["window_end"], utc=True, errors="raise")
    if frame.duplicated(["window_begin", "window_end"]).any():
        raise ValueError("duplicate canonical UTC windows")
    if not (frame["window_end"] - frame["window_begin"] == pd.Timedelta(hours=24)).all():
        raise ValueError("every predictor window must be exactly 24 hours")
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("target must be binary")
    frame = frame.sort_values(["window_end", "window_begin"]).reset_index(drop=True)
    cadence_hours = frame["window_end"].diff().dropna().dt.total_seconds().div(3600)
    if (cadence_hours <= 0).any() or ((cadence_hours % 24) != 0).any():
        raise ValueError("publisher training windows must follow a positive daily cadence with allowed gaps")
    future_columns = [name for name in frame.columns if name.lower().startswith("future_")]
    feature_columns = [
        name for name in frame.columns
        if name not in {"window_begin", "window_end", TARGET}
        and not name.lower().startswith("future_")
    ]
    if any(name.lower().startswith("future_") for name in feature_columns):
        raise ValueError("future outcome leaked into features")
    units = build_units(frame)
    positive_units = sum(unit.label for unit in units)
    positive_cutoffs = (
        int(positive_units * 0.70),
        int(positive_units * 0.80),
        int(positive_units * 0.90),
    )
    provisional_counts = {role: 0 for role in ROLES}
    positives_seen = 0
    for unit in units:
        if positives_seen < positive_cutoffs[0]: role = "train"
        elif positives_seen < positive_cutoffs[1]: role = "validation_monitor"
        elif positives_seen < positive_cutoffs[2]: role = "validation_calibration"
        else: role = "validation_threshold"
        provisional_counts[role] += 1
        positives_seen += unit.label
    counts = provisional_counts
    assigned = assign_chronological_roles(units, counts, purge_hours=24)
    row_roles: dict[int, tuple[str, str]] = {}
    for role in ROLES:
        for unit in assigned[role]:
            for raw_index in unit.issue_ids:
                row_roles[int(raw_index)] = (role, unit.unit_id)
    selected = frame.loc[sorted(row_roles)].copy()
    selected.insert(0, "issue_id", [hashlib.sha256(f"{row.window_begin.isoformat()}|{row.window_end.isoformat()}".encode()).hexdigest() for row in selected.itertuples()])
    selected.insert(1, "role", [row_roles[int(index)][0] for index in selected.index])
    selected.insert(2, "unit_id", [row_roles[int(index)][1] for index in selected.index])
    selected = selected[["issue_id", "role", "unit_id", "window_begin", "window_end", TARGET, *feature_columns]]
    for role in ROLES:
        labels = selected.loc[selected["role"] == role, TARGET]
        if len(labels) == 0 or labels.nunique() != 2:
            raise ValueError(f"role {role} lacks both classes")
    selected["window_begin"] = selected["window_begin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    selected["window_end"] = selected["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    role_counts = {
        role: {
            "rows": int((selected["role"] == role).sum()),
            "positive_windows": int(selected.loc[selected["role"] == role, TARGET].sum()),
            "units": int(selected.loc[selected["role"] == role, "unit_id"].nunique()),
        }
        for role in ROLES
    }
    manifest = {
        "status": "DEVELOPMENT_ONLY_NOT_FINAL_BENCHMARK",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "output_sha256": sha256_file(output_csv),
        "target": TARGET,
        "target_semantics": "PUBLISHER_LEGACY_FUTURE_OPERATIONAL_WINDOW_LABEL_NOT_AUDITED_NEW_CROSSING",
        "feature_columns": len(feature_columns),
        "excluded_future_columns": sorted(future_columns),
        "quiet_block_days": 7,
        "purge_hours": 24,
        "purge_boundary": "STRICTLY_GREATER_THAN_INCLUSIVE_24H_TARGET_ENDPOINT",
        "cadence_contract": "POSITIVE_MULTIPLES_OF_24_HOURS_WITH_GAPS_ALLOWED",
        "split_policy": "CHRONOLOGICAL_70_10_10_10_POSITIVE_EPISODE_QUOTAS",
        "positive_episode_units_before_purge": positive_units,
        "role_counts": role_counts,
        "purged_units": len(assigned["purged"]),
        "locked_test_rows_present": False,
        "allowed_uses": ["baseline_reproduction", "pipeline_smoke_test", "development_only_model_selection"],
        "forbidden_claims": ["final_new_crossing_score", "SEPVAL_score", "operational_certification", "breakthrough"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
