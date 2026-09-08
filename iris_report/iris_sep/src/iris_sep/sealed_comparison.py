"""Prospective comparator sealing for untouched IRIS-SEP evaluation.

Comparators must be frozen and bound at the same forecast issue time, before the
24-hour target outcome is known.  The built-in comparator is the promoted
package's fit-role prevalence climatology.  External comparators are accepted
only when an immutable artifact digest and pre-outcome probability are supplied.

This module contains no training, calibration, threshold-selection or model
selection path.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .promoted_model_package import ARCHITECTURE, TARGET
from .sealed_evaluation import validate_forecast_seal


FORMAT = "IRIS_SEP_SEALED_COMPARISON_V1"
EVALUATION_FORMAT = "IRIS_SEP_SEALED_COMPARATOR_EVALUATION_V1"
MAX_SEAL_DELAY = timedelta(minutes=5)


class SealedComparisonError(ValueError):
    """Raised when a comparator is not demonstrably frozen before outcome."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealedComparisonError("comparator payload is not canonical JSON") from exc


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SealedComparisonError(f"{name} must be lowercase SHA-256")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SealedComparisonError(f"{name} must be finite")
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise SealedComparisonError(f"{name} outside [0,1]")
    return out


def _time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SealedComparisonError(f"{name} is invalid") from exc
    else:
        raise SealedComparisonError(f"{name} must be datetime/ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SealedComparisonError(f"{name} must be timezone-aware")
    return parsed


def build_fit_prevalence_climatology(
    *,
    package_manifest: Mapping[str, Any],
    package_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the only package-native no-retraining comparator.

    The reference probability is the prevalence fixed on the package's fit role;
    it is not re-estimated on the prospective evaluation cohort.
    """
    if not isinstance(package_manifest, Mapping):
        raise SealedComparisonError("package_manifest must be a mapping")
    if package_manifest.get("architecture") != ARCHITECTURE or package_manifest.get("target") != TARGET:
        raise SealedComparisonError("promoted package identity mismatch")
    if package_manifest.get("runtime_training_allowed") is not False:
        raise SealedComparisonError("package permits runtime training")
    prevalence = _probability(package_manifest.get("fit_prevalence"), "fit_prevalence")
    if not 0.0 < prevalence < 1.0:
        raise SealedComparisonError("fit_prevalence must be in (0,1)")
    return {
        "comparator_id": "FIT_ROLE_PREVALENCE_CLIMATOLOGY",
        "comparator_kind": "FROZEN_CONSTANT_PROBABILITY",
        "target": TARGET,
        "probability": prevalence,
        "artifact_sha256": _digest(package_manifest_sha256, "package_manifest_sha256"),
        "threshold": None,
        "trained_or_tuned_on_evaluation_cohort": False,
        "reference_note": "Frozen promoted-package fit-role prevalence; not evaluation-cohort prevalence.",
    }


def build_external_frozen_comparator(
    *,
    comparator_id: str,
    comparator_version: str,
    artifact_sha256: str,
    probability: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Describe one externally produced forecast frozen before the outcome."""
    if not isinstance(comparator_id, str) or not comparator_id.strip():
        raise SealedComparisonError("comparator_id is required")
    if not isinstance(comparator_version, str) or not comparator_version.strip():
        raise SealedComparisonError("comparator_version is required")
    return {
        "comparator_id": comparator_id.strip(),
        "comparator_kind": "EXTERNAL_IMMUTABLE_FORECAST",
        "comparator_version": comparator_version.strip(),
        "target": TARGET,
        "probability": _probability(probability, "external comparator probability"),
        "artifact_sha256": _digest(artifact_sha256, "artifact_sha256"),
        "threshold": None if threshold is None else _probability(threshold, "external comparator threshold"),
        "trained_or_tuned_on_evaluation_cohort": False,
    }


def seal_comparison(
    *,
    forecast_seal: Mapping[str, Any],
    comparators: Sequence[Mapping[str, Any]],
    sealed_at: datetime,
) -> dict[str, Any]:
    """Bind same-issue comparators to an already-valid immutable forecast seal."""
    forecast = validate_forecast_seal(forecast_seal)
    issue = _time(forecast["issued_at"], "forecast issued_at")
    sealed = _time(sealed_at, "comparison sealed_at")
    if sealed < issue or sealed - issue > MAX_SEAL_DELAY:
        raise SealedComparisonError("comparators must be sealed within five minutes of forecast issuance")
    if isinstance(comparators, (str, bytes)) or not isinstance(comparators, Sequence) or not comparators:
        raise SealedComparisonError("at least one comparator is required")

    frozen = []
    seen = set()
    for raw in comparators:
        if not isinstance(raw, Mapping):
            raise SealedComparisonError("comparator must be a mapping")
        row = dict(raw)
        comparator_id = row.get("comparator_id")
        if not isinstance(comparator_id, str) or not comparator_id or comparator_id in seen:
            raise SealedComparisonError("comparator IDs must be unique non-empty strings")
        seen.add(comparator_id)
        if row.get("target") != TARGET:
            raise SealedComparisonError("comparator target mismatch")
        if row.get("trained_or_tuned_on_evaluation_cohort") is not False:
            raise SealedComparisonError("comparator was trained or tuned on evaluation cohort")
        _probability(row.get("probability"), f"{comparator_id} probability")
        _digest(row.get("artifact_sha256"), f"{comparator_id} artifact_sha256")
        threshold = row.get("threshold")
        if threshold is not None:
            _probability(threshold, f"{comparator_id} threshold")
        frozen.append(row)

    payload = {
        "format": FORMAT,
        "target": TARGET,
        "issued_at": issue.isoformat(),
        "sealed_at": sealed.isoformat(),
        "forecast_seal_sha256": forecast["forecast_seal_sha256"],
        "comparators": frozen,
        "training_allowed": False,
        "recalibration_allowed": False,
        "rethresholding_allowed": False,
        "post_outcome_comparator_selection_allowed": False,
    }
    payload["comparison_seal_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def validate_sealed_comparison(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or receipt.get("format") != FORMAT or receipt.get("target") != TARGET:
        raise SealedComparisonError("unsupported sealed comparison")
    unsigned = dict(receipt)
    claimed = unsigned.pop("comparison_seal_sha256", None)
    if _digest(claimed, "comparison_seal_sha256") != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise SealedComparisonError("comparison seal digest mismatch")
    issue = _time(receipt.get("issued_at"), "issued_at")
    sealed = _time(receipt.get("sealed_at"), "sealed_at")
    if sealed < issue or sealed - issue > MAX_SEAL_DELAY:
        raise SealedComparisonError("comparison was not sealed at issue time")
    if any(receipt.get(key) is not False for key in (
        "training_allowed", "recalibration_allowed", "rethresholding_allowed", "post_outcome_comparator_selection_allowed"
    )):
        raise SealedComparisonError("comparison receipt permits post-freeze adaptation")
    comparators = receipt.get("comparators")
    if not isinstance(comparators, list) or not comparators:
        raise SealedComparisonError("sealed comparator list missing")
    seen = set()
    for row in comparators:
        if not isinstance(row, Mapping):
            raise SealedComparisonError("invalid sealed comparator")
        cid = row.get("comparator_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise SealedComparisonError("invalid comparator ID")
        seen.add(cid)
        if row.get("target") != TARGET or row.get("trained_or_tuned_on_evaluation_cohort") is not False:
            raise SealedComparisonError("comparator freeze contract violated")
        _probability(row.get("probability"), f"{cid} probability")
        _digest(row.get("artifact_sha256"), f"{cid} artifact_sha256")
        if row.get("threshold") is not None:
            _probability(row.get("threshold"), f"{cid} threshold")
    return dict(receipt)


def evaluate_sealed_comparators(
    *,
    comparison_receipts: Sequence[Mapping[str, Any]],
    label_receipt: Mapping[str, Any],
    full_state_probabilities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Score predeclared comparators on exactly the eligible sealed cohort.

    ``full_state_probabilities`` is an optional mapping from forecast seal SHA to
    the FULL IRIS probability.  When supplied, paired Brier deltas are reported.
    No parameter is estimated from the outcomes.
    """
    labels = label_receipt.get("rows") if isinstance(label_receipt, Mapping) else None
    if not isinstance(labels, list):
        raise SealedComparisonError("label receipt rows missing")
    label_by_forecast = {
        str(row.get("forecast_seal_sha256")): int(row["label"])
        for row in labels
        if isinstance(row, Mapping) and row.get("eligible_new_crossing_issue") is True and row.get("label") in (0, 1)
    }
    rows_by_id: dict[str, list[tuple[int, float, str]]] = {}
    for raw in comparison_receipts:
        receipt = validate_sealed_comparison(raw)
        forecast_sha = str(receipt["forecast_seal_sha256"])
        if forecast_sha not in label_by_forecast:
            continue
        y = label_by_forecast[forecast_sha]
        for comparator in receipt["comparators"]:
            rows_by_id.setdefault(str(comparator["comparator_id"]), []).append((y, float(comparator["probability"]), forecast_sha))

    results: dict[str, Any] = {}
    for comparator_id, rows in sorted(rows_by_id.items()):
        y = np.asarray([r[0] for r in rows], dtype=float)
        p = np.asarray([r[1] for r in rows], dtype=float)
        comparator_brier = float(np.mean((p - y) ** 2))
        result: dict[str, Any] = {"rows": len(rows), "positives": int(np.sum(y)), "brier": comparator_brier}
        if full_state_probabilities is not None:
            try:
                iris_p = np.asarray([_probability(full_state_probabilities[r[2]], "FULL probability") for r in rows], dtype=float)
            except KeyError as exc:
                raise SealedComparisonError("FULL probability missing for one sealed forecast") from exc
            iris_brier = float(np.mean((iris_p - y) ** 2))
            result["iris_full_brier"] = iris_brier
            result["iris_minus_comparator_brier"] = iris_brier - comparator_brier
            result["iris_brier_skill_vs_comparator"] = (
                1.0 - iris_brier / comparator_brier if comparator_brier > 0 else math.nan
            )
        results[comparator_id] = result
    return {
        "format": EVALUATION_FORMAT,
        "target": TARGET,
        "comparators": results,
        "training_allowed": False,
        "recalibration_allowed": False,
        "rethresholding_allowed": False,
    }
