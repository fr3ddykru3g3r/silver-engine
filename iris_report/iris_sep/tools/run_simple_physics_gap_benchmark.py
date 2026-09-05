"""Run the reduced rotate-and-spread model on deliberately hidden train-only magnetic maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from iris_report.iris_sep.src.iris_sep.physics_gap_experiment import (
    benchmark_hidden_maps,
)
from iris_report.iris_sep.src.iris_sep.simple_physics import RotateSpreadConfig


FORMAT = "IRIS_SEP_TRAIN_ONLY_MAGNETIC_MAP_PACKAGE_V1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def load_package(package: Path, metadata: Path):
    meta = json.loads(Path(metadata).read_text())
    if (
        meta.get("format") != FORMAT
        or meta.get("scope") != "TRAIN_ONLY_TRANSIENT_MAGNETIC_MAP_GAPS"
        or meta.get("locked_test_included") is not False
    ):
        raise ValueError(
            "magnetic-map package scope/format/locked boundary mismatch"
        )
    if SHA256.fullmatch(str(meta.get("source_manifest_sha256", ""))) is None:
        raise ValueError("source_manifest_sha256 required")
    geometry = meta.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("explicit map geometry/transport configuration required")
    required_geometry = [
        "longitude_degrees_per_pixel",
        "rotation_degrees_per_day",
        "diffusion_pixels2_per_day",
        "max_substep_hours",
        "validated_horizon_hours",
    ]
    if any(key not in geometry for key in required_geometry):
        raise ValueError("incomplete geometry/transport configuration")
    config = RotateSpreadConfig(
        **{key: geometry[key] for key in required_geometry}
    )

    with np.load(package, allow_pickle=False) as archive:
        required = [
            "maps",
            "map_observed",
            "structural_unavailable",
            "roles",
            "issue_ids",
            "issue_time_unix_seconds",
        ]
        if any(key not in archive for key in required):
            raise ValueError("missing magnetic-map package arrays")
        data = {key: archive[key] for key in required}

    maps = np.asarray(data["maps"], dtype=np.float64)
    observed = np.asarray(data["map_observed"], dtype=bool)
    structural = np.asarray(data["structural_unavailable"], dtype=bool)
    roles = np.asarray(data["roles"], dtype=str)
    issue_ids = np.asarray(data["issue_ids"], dtype=str)
    times = np.asarray(data["issue_time_unix_seconds"], dtype=np.float64)
    row_count = maps.shape[0] if maps.ndim == 3 else 0
    if row_count == 0 or any(
        array.shape != (row_count,)
        for array in [observed, structural, roles, issue_ids, times]
    ):
        raise ValueError("magnetic-map package arrays do not align")
    if len(np.unique(issue_ids)) != row_count or np.any(issue_ids == ""):
        raise ValueError("issue_ids must be unique and nonempty")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("map issue times must be strictly increasing")
    if np.any(observed & structural):
        raise ValueError("structurally unavailable map cannot be observed")
    if not np.isfinite(maps[observed]).all():
        raise ValueError("observed maps must be finite")
    if "score" not in set(roles.tolist()) or any(
        "locked" in role.lower() or "test" in role.lower()
        for role in roles
    ):
        raise ValueError(
            "package must contain train-only score role and no locked/test role"
        )
    return (
        meta,
        maps,
        observed,
        structural,
        roles,
        issue_ids,
        times,
        config,
    )


def run(
    package: Path,
    metadata: Path,
    output: Path,
    missing_fraction: float,
    seed: int,
):
    output = Path(output)
    if output.exists():
        raise ValueError("immutable output directory already exists")
    if not math.isfinite(missing_fraction) or not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be in (0,1)")

    (
        meta,
        maps,
        observed,
        structural,
        roles,
        issue_ids,
        times,
        config,
    ) = load_package(package, metadata)
    eligible = observed & ~structural & (roles == "score")
    locations = np.flatnonzero(eligible)
    if len(locations) < 1:
        raise ValueError("no eligible score-role magnetic maps")
    count = max(
        1,
        min(
            len(locations),
            int(round(len(locations) * missing_fraction)),
        ),
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(locations, size=count, replace=False)
    holdout = np.zeros(len(observed), dtype=bool)
    holdout[chosen] = True

    output.mkdir(parents=True)
    preregistration = {
        "scope": "TRAIN_ONLY_HIDDEN_MAGNETIC_MAP_DIAGNOSTIC",
        "method": "ROTATE_SPREAD_2D_V1",
        "comparator": "LAST_REAL_MAP_PERSISTENCE",
        "missing_fraction": float(missing_fraction),
        "seed": int(seed),
        "package_sha256": digest(package),
        "metadata_sha256": digest(metadata),
        "source_manifest_sha256": meta["source_manifest_sha256"],
        "geometry": meta["geometry"],
        "locked_test_accessed": False,
        "downstream_sep_scored": False,
    }
    save_json(output / "preregistration.json", preregistration)
    np.savez(
        output / "holdout.npz",
        holdout_rows=holdout,
        issue_ids=issue_ids,
    )
    result = benchmark_hidden_maps(
        maps=maps,
        map_observed=observed,
        structural_unavailable=structural,
        issue_time_unix_seconds=times,
        roles=roles,
        holdout_rows=holdout,
        config=config,
    )
    receipt = {
        "status": "COMPLETED_TRAIN_ONLY_HIDDEN_MAP_DIAGNOSTIC",
        "result": result,
        "preregistration_sha256": digest(output / "preregistration.json"),
        "holdout_sha256": digest(output / "holdout.npz"),
        "locked_test_accessed": False,
        "downstream_sep_scored": False,
        "physics_advantage_established": False,
        "claim_boundary": (
            "This runner can execute when a verified train-only magnetic-map "
            "package exists. Lower pixel error alone is not enough to admit "
            "physics to the SEP forecaster; downstream forecast preservation "
            "must later be demonstrated."
        ),
    }
    save_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.package,
                args.metadata,
                args.output,
                args.missing_fraction,
                args.seed,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
