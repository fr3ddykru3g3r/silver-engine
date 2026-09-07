"""Fail-closed forecast-time provenance gate for prospective IRIS-SEP inputs.

The released historical aggregate table remains useful for retrospective research,
but a finite value is not evidence that the value was natively observed or
available at forecast issue time.  This module enforces the stricter contract for
prospective/replay-as-of-time claims.

A prospective probability is permitted only when every model input feature has
one explicit lineage record in an allowed causal state and all relevant source
and transform times are no later than the information cutoffs.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "IRIS_SEP_FORECAST_INPUT_PROVENANCE_GATE_V1"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "config" / "source_provenance_contract_v1.json"
CAUSAL_STATES = frozenset({"NATIVE_OBSERVED", "ALTERNATE_SOURCE_OBSERVED", "RECONSTRUCTED_CAUSAL"})


class ForecastInputProvenanceError(ValueError):
    """Raised when a prospective forecast input manifest is not causally admissible."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastInputProvenanceError("provenance payload is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ForecastInputProvenanceError(f"{name} must be a timezone-aware datetime")
    return value


def _parse_time(value: Any, name: str, *, required: bool = True) -> datetime | None:
    if value is None:
        if required:
            raise ForecastInputProvenanceError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise ForecastInputProvenanceError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastInputProvenanceError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForecastInputProvenanceError(f"{name} must be timezone-aware")
    return parsed


def load_provenance_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the checked-in source-provenance contract used by the gate."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastInputProvenanceError("source provenance contract could not be loaded") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("contract_id"), str):
        raise ForecastInputProvenanceError("source provenance contract is malformed")
    return payload


def _normalize_required_features(required_features: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(required_features, Mapping) or not required_features:
        raise ForecastInputProvenanceError("required_features must be a non-empty family-to-features mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for family, features in required_features.items():
        family_name = str(family)
        if not family_name:
            raise ForecastInputProvenanceError("required feature family cannot be empty")
        if isinstance(features, (str, bytes)):
            raise ForecastInputProvenanceError(f"required feature list for {family_name} must be a sequence")
        values = tuple(str(value) for value in features)
        if not values or any(not value for value in values):
            raise ForecastInputProvenanceError(f"required feature list for {family_name} cannot be empty")
        if len(set(values)) != len(values):
            raise ForecastInputProvenanceError(f"duplicate required feature in family {family_name}")
        overlap = seen.intersection(values)
        if overlap:
            raise ForecastInputProvenanceError(f"feature names must be globally unique; duplicates={sorted(overlap)}")
        seen.update(values)
        normalized[family_name] = tuple(sorted(values))
    return dict(sorted(normalized.items()))


def evaluate_forecast_input_provenance(
    *,
    issued_at: datetime,
    records: Sequence[Mapping[str, Any]],
    required_features: Mapping[str, Sequence[str]],
    development_information_cutoff: datetime | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether an exact prospective feature vector is causally admissible.

    Each record must describe exactly one required feature and include:
      feature_name, family, state, source_revision, observed_at_utc,
      published_at_utc.

    RECONSTRUCTED_CAUSAL additionally requires transform_id and
    transform_fit_through_utc, and must explicitly declare
    uses_future_values=False and retrospective_fit=False.
    """
    issue = _require_aware(issued_at, "issued_at")
    dev_cutoff = None
    if development_information_cutoff is not None:
        dev_cutoff = _require_aware(development_information_cutoff, "development_information_cutoff")
        if dev_cutoff > issue:
            raise ForecastInputProvenanceError("development_information_cutoff cannot be after issued_at")

    required = _normalize_required_features(required_features)
    contract_payload = dict(contract) if contract is not None else load_provenance_contract()
    contract_id = contract_payload.get("contract_id")
    allowed_states = set(contract_payload.get("allowed_states", ()))
    contract_families = contract_payload.get("families", {})
    if not isinstance(contract_id, str) or not allowed_states or not isinstance(contract_families, Mapping):
        raise ForecastInputProvenanceError("source provenance contract is malformed")
    unknown_required = sorted(set(required) - set(str(k) for k in contract_families))
    if unknown_required:
        raise ForecastInputProvenanceError(f"required feature families absent from contract: {unknown_required}")

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ForecastInputProvenanceError("records must be a sequence of lineage mappings")

    expected = {(family, feature) for family, features in required.items() for feature in features}
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    structural_errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            structural_errors.append(f"record[{index}]:NOT_A_MAPPING")
            continue
        family = str(record.get("family", ""))
        feature = str(record.get("feature_name", ""))
        key = (family, feature)
        if not family or not feature:
            structural_errors.append(f"record[{index}]:MISSING_FAMILY_OR_FEATURE")
            continue
        if key not in expected:
            structural_errors.append(f"{family}/{feature}:UNEXPECTED_FEATURE")
            continue
        if key in by_key:
            structural_errors.append(f"{family}/{feature}:DUPLICATE_LINEAGE")
            continue
        by_key[key] = record

    missing_keys = sorted(expected - set(by_key))
    structural_errors.extend(f"{family}/{feature}:MISSING_LINEAGE" for family, feature in missing_keys)

    feature_results: list[dict[str, Any]] = []
    reasons = list(structural_errors)
    for family, feature in sorted(expected):
        record = by_key.get((family, feature))
        if record is None:
            feature_results.append({
                "family": family,
                "feature_name": feature,
                "state": "MISSING_LINEAGE",
                "permitted": False,
                "reasons": ["MISSING_LINEAGE"],
            })
            continue

        state = str(record.get("state", ""))
        local: list[str] = []
        if state not in allowed_states:
            local.append("STATE_NOT_ALLOWED_BY_CONTRACT")
        elif state not in CAUSAL_STATES:
            local.append(f"NONCAUSAL_OR_UNUSABLE_STATE:{state}")

        source_revision = record.get("source_revision")
        if not isinstance(source_revision, str) or not source_revision.strip():
            local.append("SOURCE_REVISION_REQUIRED")

        observed = None
        published = None
        try:
            observed = _parse_time(record.get("observed_at_utc"), "observed_at_utc")
        except ForecastInputProvenanceError:
            local.append("OBSERVATION_TIME_REQUIRED_OR_INVALID")
        try:
            published = _parse_time(record.get("published_at_utc"), "published_at_utc")
        except ForecastInputProvenanceError:
            local.append("PUBLICATION_TIME_REQUIRED_OR_INVALID")

        if observed is not None and observed > issue:
            local.append("OBSERVATION_AFTER_ISSUE_TIME")
        if published is not None and published > issue:
            local.append("PUBLICATION_AFTER_ISSUE_TIME")
        if observed is not None and published is not None and published < observed:
            local.append("PUBLICATION_PRECEDES_OBSERVATION")

        uses_future = record.get("uses_future_values")
        retrospective_fit = record.get("retrospective_fit")
        if uses_future is not False:
            local.append("FUTURE_VALUE_USE_NOT_EXPLICITLY_FALSE")
        if retrospective_fit is not False:
            local.append("RETROSPECTIVE_FIT_NOT_EXPLICITLY_FALSE")

        if state == "RECONSTRUCTED_CAUSAL":
            transform_id = record.get("transform_id")
            if not isinstance(transform_id, str) or not transform_id.strip():
                local.append("CAUSAL_RECONSTRUCTION_TRANSFORM_ID_REQUIRED")
            if dev_cutoff is None:
                local.append("DEVELOPMENT_INFORMATION_CUTOFF_REQUIRED_FOR_RECONSTRUCTION")
            fit_through = None
            try:
                fit_through = _parse_time(
                    record.get("transform_fit_through_utc"),
                    "transform_fit_through_utc",
                )
            except ForecastInputProvenanceError:
                local.append("TRANSFORM_FIT_THROUGH_REQUIRED_OR_INVALID")
            if fit_through is not None and dev_cutoff is not None and fit_through > dev_cutoff:
                local.append("TRANSFORM_FIT_EXTENDS_BEYOND_DEVELOPMENT_CUTOFF")
            if fit_through is not None and fit_through > issue:
                local.append("TRANSFORM_FIT_EXTENDS_BEYOND_ISSUE_TIME")
        else:
            if record.get("transform_fit_through_utc") is not None:
                local.append("OBSERVED_STATE_MUST_NOT_CARRY_TRANSFORM_FIT_TIME")

        permitted = not local
        if not permitted:
            reasons.extend(f"{family}/{feature}:{reason}" for reason in local)
        feature_results.append({
            "family": family,
            "feature_name": feature,
            "state": state,
            "permitted": permitted,
            "reasons": local,
            "source_revision": source_revision if isinstance(source_revision, str) else None,
        })

    family_results: dict[str, dict[str, Any]] = {}
    for family, features in required.items():
        rows = [row for row in feature_results if row["family"] == family]
        family_results[family] = {
            "required_feature_count": len(features),
            "permitted_feature_count": sum(bool(row["permitted"]) for row in rows),
            "permitted": bool(rows) and all(bool(row["permitted"]) for row in rows),
        }

    normalized_manifest = {
        "contract_id": contract_id,
        "issued_at": issue.isoformat(),
        "development_information_cutoff": dev_cutoff.isoformat() if dev_cutoff is not None else None,
        "required_features": {family: list(features) for family, features in required.items()},
        "records": [dict(record) for _, record in sorted(by_key.items())],
    }
    manifest_sha = _sha256(_canonical_json(normalized_manifest))
    permitted = not reasons and all(value["permitted"] for value in family_results.values())
    return {
        "format": FORMAT,
        "contract_id": contract_id,
        "status": "VALID" if permitted else "BLOCKED",
        "forecast_probability_permitted": permitted,
        "issued_at": issue.isoformat(),
        "development_information_cutoff": dev_cutoff.isoformat() if dev_cutoff is not None else None,
        "manifest_sha256": manifest_sha,
        "required_feature_count": len(expected),
        "lineage_record_count": len(by_key),
        "family_results": family_results,
        "feature_results": feature_results,
        "reasons": sorted(set(reasons)),
    }


def require_forecast_input_provenance(**kwargs: Any) -> dict[str, Any]:
    """Return a VALID gate receipt or fail closed with the exact rejection reasons."""
    result = evaluate_forecast_input_provenance(**kwargs)
    if not result["forecast_probability_permitted"]:
        reason_text = "; ".join(result["reasons"][:12]) or "unspecified provenance failure"
        raise ForecastInputProvenanceError(f"prospective forecast input blocked: {reason_text}")
    return result
