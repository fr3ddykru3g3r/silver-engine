"""Build a safe train-only NEW-crossing package for the frozen missingness benchmark.

This adapter uses only the hash-pinned public SEP-PRISM feature table and CLEAR
operational event catalogue already accepted by the development diagnostics.
It reuses the hardened NEW-crossing target and episode-disjoint chronological
role construction, excludes the already-inspected 2023-2025 monitor, and never
includes locked-test data.

Pre-existing unavailable cells are conservatively marked unavailable to the
synthetic outage experiment.  The benchmark may hide only genuinely observed,
finite score-role cells; this adapter makes no claim that every historical
unavailable cell was physically structural rather than transient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs
from iris_report.iris_sep.tools import run_missingness_forecast_benchmark as missing
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1


FORMAT = missing.FORMAT
TARGET = missing.TARGET
SCOPE = "TRAIN_ONLY_NEW_CROSSING_MISSINGNESS"
RETAINED_ROLES = tuple(missing.ROLES)
UPSTREAM_SEP_PRISM_COMMIT = "e138dcd72c1952a00e11e1a0b025337f9e7c93fb"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def issue_id(timestamp) -> str:
    return hashlib.sha256(timestamp.isoformat().encode("ascii")).hexdigest()


def build(features: Path, events: Path, output: Path) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise ValueError("output directory must be new and immutable")
    output.mkdir(parents=True)

    # prepare_frame fails closed on both pinned public source hashes and derives
    # the exact NEW-crossing target while removing already-active issue times.
    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(
        frame, y, event_ids, None
    )

    keep = np.isin(roles, RETAINED_ROLES)
    if not keep.any():
        raise ValueError("no retained train-only rows")
    if np.any(roles[keep] == "monitor") or np.any(roles[keep] == "outside"):
        raise ValueError("monitor/outside row leaked into train-only package")

    retained_frame = frame.loc[keep].reset_index(drop=True)
    retained_y = np.asarray(y, dtype=np.int8)[keep]
    retained_roles = np.asarray(roles, dtype="U16")[keep]
    retained_units = np.asarray(units, dtype="U64")[keep]

    # Families come from the hardened feature splitter: future_* and CLEAR
    # OSEP/GSEP catalogue labels are excluded from learned predictors.
    feature_names = list(base) + list(xrs) + list(proton)
    if len(feature_names) != len(set(feature_names)) or not feature_names:
        raise ValueError("causal feature names must be unique and non-empty")

    raw_values = retained_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    observed = np.isfinite(raw_values)
    unavailable = ~observed
    values = raw_values.copy()
    # Missing entries are never interpreted as observations.  A finite sentinel
    # makes the package portable; the masks remain the authoritative provenance.
    values[unavailable] = 0.0

    if not observed.any() or np.any(observed & unavailable):
        raise ValueError("invalid observed/unavailable masks")

    issue_times = retained_frame["window_end"]
    issue_ids = np.asarray([issue_id(t) for t in issue_times], dtype="U64")
    if len(np.unique(issue_ids)) != len(issue_ids):
        raise ValueError("issue identifiers are not unique")
    issue_seconds = issue_times.astype("int64").to_numpy(dtype=np.int64) // 10**9
    if np.any(np.diff(issue_seconds) <= 0):
        raise ValueError("retained issue times are not strictly increasing")

    role_support = {}
    for role in RETAINED_ROLES:
        mask = retained_roles == role
        if not mask.any() or len(np.unique(retained_y[mask])) != 2:
            raise ValueError(f"{role} lacks both classes")
        role_support[role] = {
            "rows": int(mask.sum()),
            "positives": int(retained_y[mask].sum()),
            "units": int(len(np.unique(retained_units[mask]))),
            "from": issue_times[mask].min().isoformat(),
            "to": issue_times[mask].max().isoformat(),
        }

    source_manifest = {
        "format": "IRIS_SEP_PUBLIC_NEW_CROSSING_MISSINGNESS_SOURCE_V1",
        "target": TARGET,
        "scope": SCOPE,
        "upstream_sep_prism_commit": UPSTREAM_SEP_PRISM_COMMIT,
        "feature_table_sha256": digest(features),
        "expected_feature_table_sha256": v1.EXPECTED_FEATURE_SHA256,
        "event_catalogue_sha256": digest(events),
        "expected_event_catalogue_sha256": v1.EXPECTED_EVENT_SHA256,
        "target_builder": "run_public_new_crossing_benchmark.derive_target",
        "role_builder": "run_public_new_crossing_benchmark_v2.build_units via run_context_stability_diagnostic.build_scope_roles",
        "feature_splitter": "run_public_new_crossing_benchmark_v2.feature_sets",
        "monitor_start_excluded": cs.MONITOR_START.isoformat(),
        "monitor_rows_included": False,
        "locked_test_included": False,
        "positive_event_units_pre_monitor": int(positive_units),
        "purged_units": list(purged),
        "role_support": role_support,
        "feature_count": int(len(feature_names)),
        "feature_schema_sha256": json_digest(feature_names),
        "dropped_non_numeric_columns": list(dropped),
        "preexisting_unavailable_cells": int(unavailable.sum()),
        "observed_cells": int(observed.sum()),
        "preexisting_unavailable_semantics": (
            "EXCLUDED_FROM_ARTIFICIAL_HOLDOUT_AND_RECONSTRUCTION; "
            "NO_CLAIM_THAT_EVERY_HISTORICAL_GAP_HAS_PHYSICAL_STRUCTURAL_CAUSE"
        ),
        "score_block_prior_inspection_disclosed": True,
        "allowed_use": "DEVELOPMENT_ONLY_SYNTHETIC_FORECAST_TIME_OUTAGE_ROBUSTNESS",
        "forbidden_claims": [
            "final_locked_new_crossing_result",
            "operational_certification",
            "superiority",
            "breakthrough",
        ],
    }
    source_manifest_path = output / "source_manifest.json"
    save_json(source_manifest_path, source_manifest)

    package_path = output / "package.npz"
    np.savez_compressed(
        package_path,
        values=values,
        observed_mask=observed,
        structural_unavailable_mask=unavailable,
        labels=retained_y,
        roles=retained_roles,
        issue_ids=issue_ids,
        unit_ids=retained_units,
        issue_time_unix_seconds=issue_seconds.astype(np.float64),
    )

    metadata = {
        "format": FORMAT,
        "target": TARGET,
        "scope": SCOPE,
        "locked_test_included": False,
        "chronological_roles_verified": True,
        "episode_disjoint_roles_verified": True,
        "purge_hours": int(v1.PURGE_HOURS),
        "feature_names": feature_names,
        "feature_schema_sha256": json_digest(feature_names),
        "source_manifest_sha256": digest(source_manifest_path),
        "source_feature_table_sha256": digest(features),
        "source_event_catalogue_sha256": digest(events),
        "monitor_rows_included": False,
        "score_block_prior_inspection_disclosed": True,
        "preexisting_unavailable_mask_semantics": source_manifest[
            "preexisting_unavailable_semantics"
        ],
        "unobserved_value_sentinel": 0.0,
    }
    metadata_path = output / "metadata.json"
    save_json(metadata_path, metadata)

    # Re-open through the exact frozen consumer contract before publishing.
    missing.load_package(package_path, metadata_path)

    receipt = {
        "status": "COMPLETED_PUBLIC_TRAIN_ONLY_NEW_CROSSING_MISSINGNESS_PACKAGE",
        "target": TARGET,
        "rows": int(len(retained_y)),
        "positives": int(retained_y.sum()),
        "feature_count": int(len(feature_names)),
        "observed_cells": int(observed.sum()),
        "preexisting_unavailable_cells": int(unavailable.sum()),
        "package_sha256": digest(package_path),
        "metadata_sha256": digest(metadata_path),
        "source_manifest_sha256": digest(source_manifest_path),
        "consumer_contract_validation_passed": True,
        "locked_test_accessed": False,
        "monitor_included": False,
        "development_only": True,
    }
    save_json(output / "package_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.features, args.events, args.output), indent=2))


if __name__ == "__main__":
    main()
