"""Build a train-only NEW-crossing package for the legacy missingness benchmark.

This adapter uses only the hash-pinned public SEP-PRISM feature table and CLEAR
operational event catalogue already accepted by the development diagnostics.
It reuses the hardened NEW-crossing target and episode-disjoint chronological
role construction, excludes the already-inspected 2023-2025 monitor, and never
includes locked-test data.

IMPORTANT PROVENANCE BOUNDARY
-----------------------------
The released aggregate table does not preserve enough row/cell lineage to infer
that a finite value is a native sensor observation, or that a missing value is
physically structural. The frozen consumer format unfortunately names its two
availability fields ``observed_mask`` and ``structural_unavailable_mask``.
For backward compatibility this adapter retains those field names, but their
scientific semantics here are only:

- ``observed_mask`` = finite value available at the released aggregate interface;
- ``structural_unavailable_mask`` = pre-existing unavailable aggregate cell.

Neither field establishes native-vs-reconstructed source provenance. Provenance
is ``UNKNOWN`` unless separately demonstrated under
``config/source_provenance_contract_v1.json``. Consequently this package may be
used for aggregate-interface perturbation stress tests, not a native-sensor
reconstruction-truth claim.
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
PROVENANCE_CONTRACT = "config/source_provenance_contract_v1.json"


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

    project_root = Path(__file__).resolve().parents[1]
    provenance_path = project_root / PROVENANCE_CONTRACT
    if not provenance_path.exists():
        raise ValueError("source provenance contract missing")

    # prepare_frame fails closed on both pinned public source hashes and derives
    # the exact NEW-crossing target while removing already-active issue times.
    frame, y, event_ids, base, xrs, proton, dropped = cs.prepare_frame(features, events)
    roles, units, purged, positive_units = cs.build_scope_roles(frame, y, event_ids, None)

    keep = np.isin(roles, RETAINED_ROLES)
    if not keep.any():
        raise ValueError("no retained train-only rows")
    if np.any(roles[keep] == "monitor") or np.any(roles[keep] == "outside"):
        raise ValueError("monitor/outside row leaked into train-only package")

    retained_frame = frame.loc[keep].reset_index(drop=True)
    retained_y = np.asarray(y, dtype=np.int8)[keep]
    retained_roles = np.asarray(roles, dtype="U16")[keep]
    retained_units = np.asarray(units, dtype="U64")[keep]

    feature_names = list(base) + list(xrs) + list(proton)
    if len(feature_names) != len(set(feature_names)) or not feature_names:
        raise ValueError("feature names must be unique and non-empty")

    raw_values = retained_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    finite_available = np.isfinite(raw_values)
    preexisting_unavailable = ~finite_available
    values = raw_values.copy()
    values[preexisting_unavailable] = 0.0

    if not finite_available.any() or np.any(finite_available & preexisting_unavailable):
        raise ValueError("invalid finite/unavailable masks")

    issue_times = retained_frame["window_end"]
    issue_ids = np.asarray([issue_id(t) for t in issue_times], dtype="U64")
    if len(np.unique(issue_ids)) != len(issue_ids):
        raise ValueError("issue identifiers are not unique")
    issue_seconds = v1._unix_seconds(issue_times)
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
        "source_provenance_contract": PROVENANCE_CONTRACT,
        "source_provenance_contract_sha256": digest(provenance_path),
        "aggregate_cell_provenance": "UNKNOWN_UNLESS_SEPARATELY_DEMONSTRATED",
        "finite_cell_claimed_native_observation": False,
        "missing_cell_claimed_physically_structural": False,
        "target_builder": "run_public_new_crossing_benchmark.derive_target",
        "role_builder": "run_context_stability_diagnostic.build_scope_roles",
        "feature_splitter": "run_public_new_crossing_benchmark.feature_sets",
        "monitor_start_excluded": cs.MONITOR_START.isoformat(),
        "monitor_rows_included": False,
        "locked_test_included": False,
        "positive_event_units_pre_monitor": int(positive_units),
        "purged_units": list(purged),
        "role_support": role_support,
        "feature_count": int(len(feature_names)),
        "feature_schema_sha256": json_digest(feature_names),
        "dropped_non_numeric_columns": list(dropped),
        "preexisting_unavailable_cells": int(preexisting_unavailable.sum()),
        "finite_interface_cells": int(finite_available.sum()),
        "legacy_mask_field_semantics": {
            "observed_mask": "FINITE_AT_RELEASED_AGGREGATE_INTERFACE_NOT_NATIVE_PROVENANCE",
            "structural_unavailable_mask": "PREEXISTING_UNAVAILABLE_AT_AGGREGATE_INTERFACE_NOT_PHYSICAL_STRUCTURAL_PROOF"
        },
        "score_block_prior_inspection_disclosed": True,
        "allowed_use": "DEVELOPMENT_ONLY_AGGREGATE_INTERFACE_PERTURBATION_STRESS",
        "forbidden_claims": [
            "native_sensor_reconstruction_truth",
            "strict_prospective_input_causality",
            "final_locked_new_crossing_result",
            "operational_certification",
            "superiority",
            "breakthrough"
        ]
    }
    source_manifest_path = output / "source_manifest.json"
    save_json(source_manifest_path, source_manifest)

    package_path = output / "package.npz"
    np.savez_compressed(
        package_path,
        values=values,
        observed_mask=finite_available,
        structural_unavailable_mask=preexisting_unavailable,
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
        "source_provenance_contract_sha256": digest(provenance_path),
        "monitor_rows_included": False,
        "score_block_prior_inspection_disclosed": True,
        "aggregate_cell_provenance": "UNKNOWN_UNLESS_SEPARATELY_DEMONSTRATED",
        "legacy_observed_mask_semantics": "FINITE_AT_RELEASED_AGGREGATE_INTERFACE_NOT_NATIVE_PROVENANCE",
        "legacy_structural_mask_semantics": "PREEXISTING_UNAVAILABLE_NOT_PHYSICAL_STRUCTURAL_PROOF",
        "unobserved_value_sentinel": 0.0
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
        "finite_interface_cells": int(finite_available.sum()),
        "preexisting_unavailable_cells": int(preexisting_unavailable.sum()),
        "package_sha256": digest(package_path),
        "metadata_sha256": digest(metadata_path),
        "source_manifest_sha256": digest(source_manifest_path),
        "source_provenance_contract_sha256": digest(provenance_path),
        "consumer_contract_validation_passed": True,
        "finite_cell_claimed_native_observation": False,
        "locked_test_accessed": False,
        "monitor_included": False,
        "development_only": True
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
