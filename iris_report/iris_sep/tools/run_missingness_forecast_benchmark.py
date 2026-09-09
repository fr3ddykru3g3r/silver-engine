"""Run the simple IRIS-SEP forecast-time missing-data experiment on a safe train-only package."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from iris_report.iris_sep.src.iris_sep.fixed_forecast_benchmark import (
    fit_fixed_reference_forecaster,
    predict_with_frozen_reference,
    score_recovery_arm,
)
from iris_report.iris_sep.src.iris_sep.missingness_experiment import (
    deterministic_transient_random_holdout,
    recover_causal_forward_fill,
    recover_train_median,
)
from iris_report.iris_sep.src.iris_sep.missingness_recovery import (
    reconstruction_metrics,
)


FORMAT = "IRIS_SEP_TRAIN_ONLY_MISSINGNESS_PACKAGE_V1"
TARGET = "new_sep_10mev_10pfu_within_24h"
ROLES = ["fit", "calibration", "threshold", "score"]
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def load_package(npz_path: Path, metadata_path: Path):
    """Load and fail closed on anything other than a train-only NEW-crossing fold."""
    metadata = json.loads(Path(metadata_path).read_text())
    if (
        metadata.get("format") != FORMAT
        or metadata.get("target") != TARGET
        or metadata.get("scope") != "TRAIN_ONLY_NEW_CROSSING_MISSINGNESS"
    ):
        raise ValueError("package metadata scope/target/format mismatch")
    if metadata.get("locked_test_included") is not False:
        raise ValueError("locked test must be excluded")
    if (
        metadata.get("chronological_roles_verified") is not True
        or metadata.get("episode_disjoint_roles_verified") is not True
    ):
        raise ValueError(
            "chronological and episode-disjoint verification must be declared"
        )
    if (
        not isinstance(metadata.get("purge_hours"), (int, float))
        or metadata["purge_hours"] < 24
    ):
        raise ValueError("purge_hours must be at least 24")
    if SHA256.fullmatch(str(metadata.get("source_manifest_sha256", ""))) is None:
        raise ValueError("source_manifest_sha256 required")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = [
            "values",
            "observed_mask",
            "structural_unavailable_mask",
            "labels",
            "roles",
            "issue_ids",
            "unit_ids",
            "issue_time_unix_seconds",
        ]
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"missing package arrays: {missing}")
        data = {key: archive[key] for key in required}

    values = np.asarray(data["values"], dtype=np.float64)
    observed = np.asarray(data["observed_mask"], dtype=bool)
    structural = np.asarray(data["structural_unavailable_mask"], dtype=bool)
    labels = np.asarray(data["labels"])
    roles = np.asarray(data["roles"], dtype=str)
    issue_ids = np.asarray(data["issue_ids"], dtype=str)
    unit_ids = np.asarray(data["unit_ids"], dtype=str)
    times = np.asarray(data["issue_time_unix_seconds"], dtype=np.float64)
    row_count = values.shape[0] if values.ndim == 2 else 0

    if (
        row_count == 0
        or observed.shape != values.shape
        or structural.shape != values.shape
    ):
        raise ValueError(
            "feature arrays must be matching non-empty 2-D arrays"
        )
    if any(
        array.shape != (row_count,)
        for array in [labels, roles, issue_ids, unit_ids, times]
    ):
        raise ValueError("row metadata arrays must align")
    if (
        len(np.unique(issue_ids)) != row_count
        or np.any(issue_ids == "")
        or np.any(unit_ids == "")
    ):
        raise ValueError("issue_ids must be unique and identifiers nonempty")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("issue times must be finite and strictly increasing")
    if set(roles.tolist()) != set(ROLES):
        raise ValueError(
            "package roles must be exactly fit/calibration/threshold/score"
        )
    order = {role: index for index, role in enumerate(ROLES)}
    encoded_roles = np.array([order[role] for role in roles])
    if np.any(np.diff(encoded_roles) < 0):
        raise ValueError("roles must be chronological noninterleaved blocks")

    for unit in np.unique(unit_ids):
        if len(np.unique(roles[unit_ids == unit])) != 1:
            raise ValueError("unit/episode crosses role boundary")

    purge_hours = float(metadata["purge_hours"])
    for left, right in zip(ROLES, ROLES[1:]):
        gap_hours = (
            times[roles == right].min() - times[roles == left].max()
        ) / 3600.0
        if not gap_hours > purge_hours:
            raise ValueError(f"{left}->{right} does not satisfy strict purge")

    feature_names = metadata.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or len(feature_names) != values.shape[1]
        or len(set(feature_names)) != len(feature_names)
        or any(not isinstance(name, str) or not name for name in feature_names)
    ):
        raise ValueError("feature_names must uniquely match feature columns")

    return (
        metadata,
        values,
        observed,
        structural,
        labels.astype(np.int8),
        roles,
        issue_ids,
        unit_ids,
        times,
    )


def _hidden_metrics(truth, recovered, holdout):
    if not holdout.any():
        return None
    return reconstruction_metrics(truth, recovered, holdout)


def run(
    package: Path,
    metadata: Path,
    output: Path,
    missing_fraction: float,
    seed: int,
):
    """Run one immutable random transient-gap experiment on score-role rows only."""
    output = Path(output)
    if output.exists():
        raise ValueError("immutable output directory already exists")
    if not math.isfinite(missing_fraction) or not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be in (0,1)")

    (
        package_metadata,
        values,
        observed,
        structural,
        labels,
        roles,
        issue_ids,
        unit_ids,
        _times,
    ) = load_package(package, metadata)

    forecaster = fit_fixed_reference_forecaster(
        values=values,
        observed_mask=observed,
        structural_unavailable_mask=structural,
        labels=labels,
        roles=roles,
        seed=seed,
    )
    supported = forecaster.supported_features
    eligible_observed = observed.copy()
    eligible_observed[:, ~supported] = False
    score_rows = roles == "score"
    holdout = deterministic_transient_random_holdout(
        eligible_observed,
        structural,
        missing_fraction=missing_fraction,
        seed=seed,
        row_eligibility=score_rows,
    )
    experimental_observed = observed & ~holdout

    preregistration = {
        "scope": "TRAIN_ONLY_NEW_CROSSING_MISSINGNESS_RANDOM_HOLDOUT",
        "target": TARGET,
        "package_sha256": digest(package),
        "metadata_sha256": digest(metadata),
        "source_manifest_sha256": package_metadata["source_manifest_sha256"],
        "missing_fraction": float(missing_fraction),
        "seed": int(seed),
        "holdout_role": "score",
        "frozen_forecaster": (
            "L2_LOGISTIC_BALANCED_C1_WITH_REFERENCE_CALIBRATION_THRESHOLD"
        ),
        "recovery_arms": [
            "MASK_AWARE_NO_FILL",
            "TRAIN_FIT_MEDIAN",
            "CAUSAL_FORWARD_FILL",
        ],
        "physics_arm_included": False,
        "physics_reason": (
            "Requires separately validated map geometry and map-to-feature "
            "reconstruction; do not fabricate tabular physics."
        ),
        "locked_test_accessed": False,
        "selection_on_score_forbidden": True,
    }
    output.mkdir(parents=True)
    save_json(output / "preregistration.json", preregistration)
    np.savez(
        output / "holdout.npz",
        holdout_mask=holdout,
        score_rows=score_rows,
        supported_features=supported,
    )

    reference = predict_with_frozen_reference(
        forecaster,
        values=values,
        observed_mask=observed,
        reconstructed_mask=np.zeros_like(observed),
    )
    no_fill = predict_with_frozen_reference(
        forecaster,
        values=values,
        observed_mask=experimental_observed,
        reconstructed_mask=np.zeros_like(observed),
    )

    fit_rows = roles == "fit"
    supported_columns = np.flatnonzero(supported)
    median_sub = recover_train_median(
        values[:, supported_columns],
        observed[:, supported_columns],
        structural[:, supported_columns],
        holdout[:, supported_columns],
        fit_rows=fit_rows,
    )
    median_values = values.copy()
    median_values[:, supported_columns] = median_sub.values
    median_reconstructed = np.zeros_like(observed)
    median_reconstructed[:, supported_columns] = median_sub.reconstructed_mask
    median = predict_with_frozen_reference(
        forecaster,
        values=median_values,
        observed_mask=experimental_observed,
        reconstructed_mask=median_reconstructed,
    )

    forward_sub = recover_causal_forward_fill(
        values[:, supported_columns],
        observed[:, supported_columns],
        structural[:, supported_columns],
        holdout[:, supported_columns],
    )
    forward_values = values.copy()
    forward_values[:, supported_columns] = forward_sub.values
    forward_reconstructed = np.zeros_like(observed)
    forward_reconstructed[:, supported_columns] = forward_sub.reconstructed_mask
    forward = predict_with_frozen_reference(
        forecaster,
        values=forward_values,
        observed_mask=experimental_observed,
        reconstructed_mask=forward_reconstructed,
    )

    arms = {
        "MASK_AWARE_NO_FILL": {
            "forecast": score_recovery_arm(
                forecaster,
                labels=labels,
                roles=roles,
                reference_probabilities=reference,
                candidate_probabilities=no_fill,
            ),
            "reconstruction": None,
        },
        "TRAIN_FIT_MEDIAN": {
            "forecast": score_recovery_arm(
                forecaster,
                labels=labels,
                roles=roles,
                reference_probabilities=reference,
                candidate_probabilities=median,
            ),
            "reconstruction": _hidden_metrics(
                values[:, supported_columns],
                median_sub.values,
                holdout[:, supported_columns],
            ),
        },
        "CAUSAL_FORWARD_FILL": {
            "forecast": score_recovery_arm(
                forecaster,
                labels=labels,
                roles=roles,
                reference_probabilities=reference,
                candidate_probabilities=forward,
            ),
            "reconstruction": _hidden_metrics(
                values[:, supported_columns],
                forward_sub.values,
                holdout[:, supported_columns]
                & forward_sub.reconstructed_mask,
            ),
        },
    }

    np.savez(
        output / "predictions.npz",
        reference=reference,
        no_fill=no_fill,
        train_median=median,
        causal_forward_fill=forward,
        issue_ids=issue_ids,
        unit_ids=unit_ids,
    )
    receipt = {
        "status": "COMPLETED_TRAIN_ONLY_MISSINGNESS_DIAGNOSTIC",
        "target": TARGET,
        "rows": int(len(labels)),
        "score_rows": int(score_rows.sum()),
        "held_out_cells": int(holdout.sum()),
        "supported_features": int(supported.sum()),
        "feature_count": int(values.shape[1]),
        "threshold": float(forecaster.threshold),
        "arms": arms,
        "preregistration_sha256": digest(output / "preregistration.json"),
        "holdout_sha256": digest(output / "holdout.npz"),
        "predictions_sha256": digest(output / "predictions.npz"),
        "locked_test_accessed": False,
        "superiority_established": False,
        "final_new_crossing_result": False,
        "claim_boundary": (
            "Train-only robustness diagnostic. Physics is not scored until real "
            "map geometry and causal map-to-feature reconstruction are available."
        ),
    }
    save_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-fraction", type=float, default=0.10)
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
