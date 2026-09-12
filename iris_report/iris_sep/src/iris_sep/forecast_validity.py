"""Fail-closed validity classification for pre-issue SEP forecast records.

This module consumes predictor/provenance facts only. It does not read, derive or
score SEP outcomes. Recovery limits remain external source-specific contracts;
this code deliberately does not turn synthetic masking percentages into an
operational threshold.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping

VALIDITY_STATES = ("VALID", "DEGRADED", "ABSTAIN")

ABSTAIN_REASON_CODES = frozenset({
    "STALE_INPUT",
    "REQUIRED_FEED_ABSENT",
    "AMBIGUOUS_PROTON_CHANNEL",
    "STRUCTURAL_UNAVAILABILITY",
    "EXCESSIVE_TRANSIENT_LOSS",
    "CAUSAL_AVAILABILITY_RECEIPT_FAILED",
    "LEDGER_INTEGRITY_FAILURE",
    "FUTURE_OBSERVATION",
    "PROVENANCE_UNCERTAIN",
})
DEGRADED_REASON_CODES = frozenset({"TRANSIENT_FORWARD_FILL_APPLIED"})
ALLOWED_REASON_CODES = ABSTAIN_REASON_CODES | DEGRADED_REASON_CODES
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ForecastValidityError(ValueError):
    """Raised when a validity record itself is malformed."""


def _parse_utc(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastValidityError(f"invalid timestamp: {value}") from exc
    if dt.tzinfo is None:
        raise ForecastValidityError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_sha256(value: str, field: str) -> str:
    value = str(value).lower()
    if not _SHA256.fullmatch(value):
        raise ForecastValidityError(f"{field} must be a 64-character SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_record(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(body))).hexdigest()


def _nonnegative_number_map(value: Any, field: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ForecastValidityError(f"{field} must be a mapping")
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ForecastValidityError(f"invalid {field} value for {key}") from exc
        if number < 0:
            raise ForecastValidityError(f"negative {field} value for {key}")
        out[str(key)] = number
    return out


def derive_reason_codes(assessment: Mapping[str, Any]) -> list[str]:
    """Map pre-issue evidence facts to frozen reason codes.

    Source-specific freshness/recovery thresholds may be supplied by a separately
    frozen execution contract. This evaluator applies them mechanically; it does
    not choose them. Unknown flags are rejected. Observation timestamps are
    compared with issue time so future values cannot be silently used.
    """
    issue = _parse_utc(str(assessment["issue_time"]))
    flags = dict(assessment.get("flags", {}))
    allowed_flags = {
        "stale_input": "STALE_INPUT",
        "required_feed_absent": "REQUIRED_FEED_ABSENT",
        "ambiguous_proton_channel": "AMBIGUOUS_PROTON_CHANNEL",
        "structural_unavailability": "STRUCTURAL_UNAVAILABILITY",
        "excessive_transient_loss": "EXCESSIVE_TRANSIENT_LOSS",
        "causal_availability_receipt_failed": "CAUSAL_AVAILABILITY_RECEIPT_FAILED",
        "ledger_integrity_failed": "LEDGER_INTEGRITY_FAILURE",
        "provenance_uncertain": "PROVENANCE_UNCERTAIN",
        "transient_forward_fill_applied": "TRANSIENT_FORWARD_FILL_APPLIED",
    }
    unknown = sorted(set(flags) - set(allowed_flags))
    if unknown:
        raise ForecastValidityError(f"unknown assessment flags: {unknown}")

    reasons = {code for name, code in allowed_flags.items() if bool(flags.get(name, False))}

    observations = assessment.get("observation_times", {}) or {}
    if not isinstance(observations, Mapping):
        raise ForecastValidityError("observation_times must be a mapping")
    for source_id, timestamp in observations.items():
        observed = _parse_utc(str(timestamp))
        if observed > issue:
            reasons.add("FUTURE_OBSERVATION")

    source_hashes = assessment.get("source_hashes", {}) or {}
    if not isinstance(source_hashes, Mapping):
        raise ForecastValidityError("source_hashes must be a mapping")
    required_sources = assessment.get("required_sources", []) or []
    if not isinstance(required_sources, list) or not all(isinstance(x, str) for x in required_sources):
        raise ForecastValidityError("required_sources must be a list of strings")
    if any(source not in source_hashes for source in required_sources):
        reasons.add("REQUIRED_FEED_ABSENT")

    ages = _nonnegative_number_map(assessment.get("input_age_seconds", {}), "input_age_seconds")
    max_ages = _nonnegative_number_map(assessment.get("input_max_age_seconds", {}), "input_max_age_seconds")
    for source_id, limit in max_ages.items():
        if source_id not in ages:
            reasons.add("REQUIRED_FEED_ABSENT")
        elif ages[source_id] > limit:
            reasons.add("STALE_INPUT")

    fill_gaps = _nonnegative_number_map(
        assessment.get("forward_fill_gap_seconds", {}), "forward_fill_gap_seconds"
    )
    fill_limits = _nonnegative_number_map(
        assessment.get("forward_fill_max_gap_seconds", {}), "forward_fill_max_gap_seconds"
    )
    for source_id, gap in fill_gaps.items():
        if source_id not in fill_limits:
            reasons.add("CAUSAL_AVAILABILITY_RECEIPT_FAILED")
        elif gap > fill_limits[source_id]:
            reasons.add("EXCESSIVE_TRANSIENT_LOSS")
        else:
            reasons.add("TRANSIENT_FORWARD_FILL_APPLIED")

    explicit = assessment.get("reason_codes", []) or []
    if not isinstance(explicit, list):
        raise ForecastValidityError("reason_codes must be a list")
    unknown_reasons = sorted(set(explicit) - ALLOWED_REASON_CODES)
    if unknown_reasons:
        raise ForecastValidityError(f"unknown reason codes: {unknown_reasons}")
    reasons.update(explicit)
    return sorted(reasons)


def classify(reason_codes: list[str]) -> str:
    codes = set(reason_codes)
    unknown = sorted(codes - ALLOWED_REASON_CODES)
    if unknown:
        raise ForecastValidityError(f"unknown reason codes: {unknown}")
    if codes & ABSTAIN_REASON_CODES:
        return "ABSTAIN"
    if codes & DEGRADED_REASON_CODES:
        return "DEGRADED"
    return "VALID"


def build_validity_record(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Build one auditable forecast-validity record from pre-issue facts only."""
    issue = _parse_utc(str(assessment["issue_time"]))
    reasons = derive_reason_codes(assessment)

    genesis = bool(assessment.get("ledger_is_genesis", False))
    previous = assessment.get("ledger_previous_hash")
    if previous is None:
        if not genesis:
            reasons = sorted(set(reasons) | {"LEDGER_INTEGRITY_FAILURE"})
    else:
        try:
            previous = _require_sha256(str(previous), "ledger_previous_hash")
        except ForecastValidityError:
            reasons = sorted(set(reasons) | {"LEDGER_INTEGRITY_FAILURE"})
            previous = None
    if genesis and assessment.get("ledger_previous_hash") is not None:
        reasons = sorted(set(reasons) | {"LEDGER_INTEGRITY_FAILURE"})

    state = classify(reasons)
    candidate_alert = assessment.get("alert")
    if candidate_alert not in (True, False, 0, 1, None):
        raise ForecastValidityError("alert must be boolean or null")
    if state != "ABSTAIN" and candidate_alert is None:
        raise ForecastValidityError("VALID/DEGRADED records require a candidate alert")
    alert = None if state == "ABSTAIN" else bool(candidate_alert)

    ages = _nonnegative_number_map(assessment.get("input_age_seconds", {}), "input_age_seconds")
    max_ages = _nonnegative_number_map(assessment.get("input_max_age_seconds", {}), "input_max_age_seconds")
    fill_gaps = _nonnegative_number_map(
        assessment.get("forward_fill_gap_seconds", {}), "forward_fill_gap_seconds"
    )
    fill_limits = _nonnegative_number_map(
        assessment.get("forward_fill_max_gap_seconds", {}), "forward_fill_max_gap_seconds"
    )

    hashes = assessment.get("source_hashes", {}) or {}
    if not isinstance(hashes, Mapping):
        raise ForecastValidityError("source_hashes must be a mapping")
    clean_hashes = {
        str(key): _require_sha256(str(value), f"source_hashes[{key}]")
        for key, value in hashes.items()
    }

    body = {
        "format": "IRIS_SEP_FORECAST_VALIDITY_RECORD_V1",
        "issue_time": _iso_z(issue),
        "alert": alert,
        "validity_state": state,
        "reason_codes": reasons,
        "input_age_seconds": dict(sorted(ages.items())),
        "input_max_age_seconds": dict(sorted(max_ages.items())),
        "forward_fill_gap_seconds": dict(sorted(fill_gaps.items())),
        "forward_fill_max_gap_seconds": dict(sorted(fill_limits.items())),
        "source_hashes": dict(sorted(clean_hashes.items())),
        "model_hash": _require_sha256(str(assessment["model_hash"]), "model_hash"),
        "threshold_hash": _require_sha256(str(assessment["threshold_hash"]), "threshold_hash"),
        "ledger_previous_hash": previous,
        "protected_outcomes_accessed": False,
    }
    body["validity_record_hash"] = _hash_record(body)
    return body
