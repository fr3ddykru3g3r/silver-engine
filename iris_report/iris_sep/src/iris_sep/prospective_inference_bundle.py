"""Prospective, provenance-gated wrapper around the immutable IRIS-SEP replay bundle.

The existing inference bundle intentionally remains an offline retrospective
research artifact.  This wrapper is stricter: every element of the transformed
feature vector must have an explicit forecast-time lineage record that passes
``forecast_input_provenance`` before a prospective bundle can be created.

Replay re-runs the provenance gate from the bound raw lineage.  It never trusts
a stored ``VALID`` flag by itself.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from iris_report.iris_sep.src.iris_sep.forecast_input_provenance import (
    ForecastInputProvenanceError,
    evaluate_forecast_input_provenance,
    require_forecast_input_provenance,
)
from iris_report.iris_sep.src.iris_sep.inference_bundle import (
    InferenceBundleError,
    build_inference_bundle,
    replay_inference_bundle,
)


FORMAT = "IRIS_SEP_PROSPECTIVE_INFERENCE_BUNDLE_V1"
SCOPE = "PROSPECTIVE_RESEARCH_CAUSAL_INPUTS_VERIFIED_NOT_OPERATIONALLY_CERTIFIED"


class ProspectiveInferenceBundleError(ValueError):
    """Raised when a prospective bundle or its causal input binding fails."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProspectiveInferenceBundleError("prospective bundle is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ProspectiveInferenceBundleError("timezone-aware datetime required")
    return value.isoformat()


def _parse_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ProspectiveInferenceBundleError(f"{name} must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveInferenceBundleError(f"invalid {name}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveInferenceBundleError(f"{name} must be timezone-aware")
    return parsed


def _required_feature_count(required_features: Mapping[str, Sequence[str]]) -> int:
    if not isinstance(required_features, Mapping):
        raise ProspectiveInferenceBundleError("required_features must be a mapping")
    count = 0
    for values in required_features.values():
        if isinstance(values, (str, bytes)):
            raise ProspectiveInferenceBundleError("required feature lists must be sequences")
        count += len(tuple(values))
    return count


def build_prospective_inference_bundle(
    *,
    input_provenance_records: Sequence[Mapping[str, Any]],
    required_features: Mapping[str, Sequence[str]],
    development_information_cutoff: datetime | None,
    **inference_kwargs: Any,
) -> tuple[bytes, str]:
    """Build a prospective artifact only after exact-vector lineage passes.

    ``inference_kwargs`` are the same named arguments accepted by
    :func:`build_inference_bundle`.  The transformed feature vector must be 1-D
    and have exactly one required provenance record per element.
    """
    issued_at = inference_kwargs.get("issued_at")
    if not isinstance(issued_at, datetime):
        raise ProspectiveInferenceBundleError("issued_at is required in inference_kwargs")
    features = np.asarray(inference_kwargs.get("transformed_features"), dtype=float)
    if features.ndim != 1 or features.size == 0:
        raise ProspectiveInferenceBundleError("prospective transformed_features must be one non-empty 1-D vector")
    if _required_feature_count(required_features) != int(features.size):
        raise ProspectiveInferenceBundleError(
            "required provenance feature count must equal transformed feature-vector length"
        )

    try:
        gate = require_forecast_input_provenance(
            issued_at=issued_at,
            records=input_provenance_records,
            required_features=required_features,
            development_information_cutoff=development_information_cutoff,
        )
    except ForecastInputProvenanceError as exc:
        raise ProspectiveInferenceBundleError(str(exc)) from exc

    try:
        inner_bytes, inner_sha = build_inference_bundle(**inference_kwargs)
    except InferenceBundleError as exc:
        raise ProspectiveInferenceBundleError(str(exc)) from exc

    payload = {
        "format": FORMAT,
        "scope": SCOPE,
        "inner_inference_bundle": {
            "sha256": inner_sha,
            "bytes_b64": base64.b64encode(inner_bytes).decode("ascii"),
        },
        "provenance": {
            "issued_at": _iso(issued_at),
            "development_information_cutoff": (
                _iso(development_information_cutoff)
                if development_information_cutoff is not None
                else None
            ),
            "required_features": {
                str(family): [str(feature) for feature in features]
                for family, features in required_features.items()
            },
            "records": [dict(record) for record in input_provenance_records],
            "gate_receipt": gate,
        },
    }
    payload_bytes = _canonical_json(payload)
    envelope = {"payload": payload, "payload_sha256": _sha256(payload_bytes)}
    bundle_bytes = _canonical_json(envelope)
    return bundle_bytes, _sha256(bundle_bytes)


def replay_prospective_inference_bundle(
    *,
    bundle_bytes: bytes,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Verify outer binding, recompute causal lineage, then replay the inner bundle."""
    if not isinstance(bundle_bytes, bytes) or _sha256(bundle_bytes) != expected_bundle_sha256:
        raise ProspectiveInferenceBundleError("prospective bundle trust-anchor mismatch")
    try:
        envelope = json.loads(bundle_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProspectiveInferenceBundleError("prospective bundle is not valid JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
        raise ProspectiveInferenceBundleError("unexpected prospective bundle envelope")
    payload = envelope["payload"]
    if _sha256(_canonical_json(payload)) != envelope["payload_sha256"]:
        raise ProspectiveInferenceBundleError("prospective payload digest mismatch")
    if not isinstance(payload, Mapping) or payload.get("format") != FORMAT or payload.get("scope") != SCOPE:
        raise ProspectiveInferenceBundleError("unsupported prospective bundle format or scope")

    inner = payload.get("inner_inference_bundle")
    provenance = payload.get("provenance")
    if not isinstance(inner, Mapping) or not isinstance(provenance, Mapping):
        raise ProspectiveInferenceBundleError("prospective bundle bindings missing")
    try:
        inner_bytes = base64.b64decode(inner["bytes_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProspectiveInferenceBundleError("invalid inner inference bundle bytes") from exc
    inner_sha = str(inner.get("sha256", ""))
    if _sha256(inner_bytes) != inner_sha:
        raise ProspectiveInferenceBundleError("inner inference bundle digest mismatch")

    issue = _parse_time(provenance.get("issued_at"), "provenance issued_at")
    cutoff_raw = provenance.get("development_information_cutoff")
    cutoff = _parse_time(cutoff_raw, "development_information_cutoff") if cutoff_raw is not None else None
    required = provenance.get("required_features")
    records = provenance.get("records")
    if not isinstance(required, Mapping) or not isinstance(records, list):
        raise ProspectiveInferenceBundleError("raw provenance manifest missing")

    try:
        recomputed_gate = evaluate_forecast_input_provenance(
            issued_at=issue,
            records=records,
            required_features={str(k): tuple(str(x) for x in v) for k, v in required.items()},
            development_information_cutoff=cutoff,
        )
    except ForecastInputProvenanceError as exc:
        raise ProspectiveInferenceBundleError(str(exc)) from exc
    if not recomputed_gate["forecast_probability_permitted"]:
        reason_text = "; ".join(recomputed_gate["reasons"][:12])
        raise ProspectiveInferenceBundleError(f"prospective provenance replay blocked: {reason_text}")
    if recomputed_gate != provenance.get("gate_receipt"):
        raise ProspectiveInferenceBundleError("stored provenance gate receipt does not match recomputation")

    # The inner request issue time is independently bound by the immutable bundle.
    try:
        inner_envelope = json.loads(inner_bytes)
        inner_issue = _parse_time(
            inner_envelope["payload"]["request"]["issued_at"],
            "inner request issued_at",
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProspectiveInferenceBundleError("inner request issue time missing") from exc
    if inner_issue != issue:
        raise ProspectiveInferenceBundleError("provenance issue time does not match inner inference request")

    try:
        result = replay_inference_bundle(
            bundle_bytes=inner_bytes,
            expected_bundle_sha256=inner_sha,
        )
    except InferenceBundleError as exc:
        raise ProspectiveInferenceBundleError(str(exc)) from exc
    result["prospective_inference_bundle_sha256"] = expected_bundle_sha256
    result["prospective_inference_bundle_scope"] = SCOPE
    result["forecast_input_provenance"] = recomputed_gate
    result["forecast_probability_permitted_by_provenance"] = True
    return result
