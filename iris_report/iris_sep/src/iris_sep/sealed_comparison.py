"""Prospective comparator sealing for untouched IRIS-SEP evaluation.

Comparators must be frozen and bound at the same forecast issue time, before the
24-hour target outcome is known. The built-in comparator is the promoted
package's fit-role prevalence climatology. External comparators are accepted
only when an immutable artifact digest and pre-outcome probability are supplied.

Comparator scoring uses the probabilities embedded in the sealed IRIS forecasts;
callers cannot substitute a separate probability map after outcomes are known.
The entire forecast/comparator/label cohort must align one-to-one before scoring.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .promoted_model_package import ARCHITECTURE, TARGET
from .sealed_evaluation import (
    threshold_metrics,
    validate_forecast_seal,
    validate_label_receipt,
)


FORMAT = "IRIS_SEP_SEALED_COMPARISON_V1"
EVALUATION_FORMAT = "IRIS_SEP_SEALED_COMPARATOR_EVALUATION_V2"
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
    """Build the package-native no-retraining comparator."""
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
    if any(
        receipt.get(key) is not False
        for key in (
            "training_allowed",
            "recalibration_allowed",
            "rethresholding_allowed",
            "post_outcome_comparator_selection_allowed",
        )
    ):
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


def _comparator_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    cid = str(row["comparator_id"])
    kind = str(row.get("comparator_kind", ""))
    version = str(row.get("comparator_version", "PACKAGE_NATIVE"))
    return cid, kind, version


def evaluate_sealed_comparators(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    comparison_receipts: Sequence[Mapping[str, Any]],
    label_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Score comparators and sealed FULL IRIS probabilities on one exact cohort.

    There is intentionally no caller-supplied IRIS probability map. FULL
    probabilities are read directly from the validated forecast seals to prevent
    post-outcome substitution. Every forecast must have exactly one comparator
    receipt and one V2 outcome row, including unresolved/ineligible cases.
    """
    labels = validate_label_receipt(label_receipt)
    if isinstance(forecast_seals, (str, bytes)) or not isinstance(forecast_seals, Sequence) or not forecast_seals:
        raise SealedComparisonError("forecast_seals must be a non-empty sequence")
    if isinstance(comparison_receipts, (str, bytes)) or not isinstance(comparison_receipts, Sequence) or not comparison_receipts:
        raise SealedComparisonError("comparison_receipts must be a non-empty sequence")

    forecasts_by_hash: dict[str, dict[str, Any]] = {}
    issue_to_hash: dict[str, str] = {}
    package_hashes: set[str] = set()
    architecture_ids: set[str] = set()
    for raw in forecast_seals:
        forecast = validate_forecast_seal(raw)
        sha = str(forecast["forecast_seal_sha256"])
        issue = str(forecast["issued_at"])
        if sha in forecasts_by_hash or issue in issue_to_hash:
            raise SealedComparisonError("duplicate forecast identity in comparison cohort")
        forecasts_by_hash[sha] = forecast
        issue_to_hash[issue] = sha
        package_hashes.add(str(forecast["package_manifest_sha256"]))
        architecture_ids.add(str(forecast["architecture_id"]))
    if len(package_hashes) != 1 or len(architecture_ids) != 1:
        raise SealedComparisonError("forecast model package/architecture changed within comparison cohort")

    comparisons_by_forecast: dict[str, dict[str, Any]] = {}
    expected_identity_set: set[tuple[str, str, str]] | None = None
    for raw in comparison_receipts:
        receipt = validate_sealed_comparison(raw)
        forecast_sha = str(receipt["forecast_seal_sha256"])
        if forecast_sha not in forecasts_by_hash:
            raise SealedComparisonError("comparison receipt references forecast outside supplied cohort")
        if forecast_sha in comparisons_by_forecast:
            raise SealedComparisonError("duplicate comparison receipt for one forecast")
        forecast = forecasts_by_hash[forecast_sha]
        if str(receipt["issued_at"]) != str(forecast["issued_at"]):
            raise SealedComparisonError("comparison issue time does not match forecast seal")
        identity_set = {_comparator_identity(row) for row in receipt["comparators"]}
        if expected_identity_set is None:
            expected_identity_set = identity_set
        elif identity_set != expected_identity_set:
            raise SealedComparisonError("comparator set/version changed within common cohort")
        comparisons_by_forecast[forecast_sha] = receipt
    if set(comparisons_by_forecast) != set(forecasts_by_hash):
        raise SealedComparisonError("every forecast must have exactly one sealed comparison receipt")

    label_rows = labels["rows"]
    label_by_forecast = {str(row["forecast_seal_sha256"]): row for row in label_rows}
    if set(label_by_forecast) != set(forecasts_by_hash):
        raise SealedComparisonError("outcome labels do not exactly match comparison forecast cohort")

    rows_by_id: dict[str, list[tuple[int, float, float, float | None]]] = {}
    unresolved = 0
    ineligible = 0
    for forecast_sha, forecast in forecasts_by_hash.items():
        label = label_by_forecast[forecast_sha]
        if str(label["issued_at"]) != str(forecast["issued_at"]):
            raise SealedComparisonError("label issue time does not match forecast seal")
        if not label["outcome_resolved"]:
            unresolved += 1
            continue
        if label["eligible_new_crossing_issue"] is not True:
            ineligible += 1
            continue
        if label["label"] not in (0, 1):
            raise SealedComparisonError("eligible resolved row is missing binary label")
        y = int(label["label"])
        iris_p = _probability(forecast["probabilities"]["FULL"], "sealed FULL IRIS probability")
        receipt = comparisons_by_forecast[forecast_sha]
        for comparator in receipt["comparators"]:
            cid = str(comparator["comparator_id"])
            cp = _probability(comparator["probability"], f"{cid} probability")
            threshold = comparator.get("threshold")
            rows_by_id.setdefault(cid, []).append(
                (y, cp, iris_p, None if threshold is None else float(threshold))
            )

    results: dict[str, Any] = {}
    for comparator_id, rows in sorted(rows_by_id.items()):
        y = np.asarray([r[0] for r in rows], dtype=float)
        p = np.asarray([r[1] for r in rows], dtype=float)
        iris_p = np.asarray([r[2] for r in rows], dtype=float)
        comparator_brier = float(np.mean((p - y) ** 2))
        iris_brier = float(np.mean((iris_p - y) ** 2))
        thresholds = {r[3] for r in rows}
        result: dict[str, Any] = {
            "rows": len(rows),
            "positives": int(np.sum(y)),
            "comparator_brier": comparator_brier,
            "iris_full_brier": iris_brier,
            "iris_minus_comparator_brier": iris_brier - comparator_brier,
            "iris_brier_skill_vs_comparator": (
                1.0 - iris_brier / comparator_brier if comparator_brier > 0 else math.nan
            ),
        }
        if len(thresholds) != 1:
            raise SealedComparisonError("comparator threshold changed within common cohort")
        threshold = next(iter(thresholds))
        if threshold is not None:
            result["comparator_numerical_threshold_metrics"] = threshold_metrics(
                y.astype(int), p, float(threshold)
            )
        results[comparator_id] = result

    return {
        "format": EVALUATION_FORMAT,
        "target": TARGET,
        "forecast_count": len(forecasts_by_hash),
        "resolved_eligible_rows": sum(len(rows) for rows in rows_by_id.values()) // max(len(rows_by_id), 1),
        "unresolved_rows": unresolved,
        "resolved_ineligible_rows": ineligible,
        "package_manifest_sha256": next(iter(package_hashes)),
        "architecture_id": next(iter(architecture_ids)),
        "common_cohort_verified": True,
        "iris_probabilities_sourced_from_forecast_seals": True,
        "comparators": results,
        "training_allowed": False,
        "recalibration_allowed": False,
        "rethresholding_allowed": False,
        "independence_verified": False,
    }
