"""Immutable, receipt-bound admission V2 inference bundle.

The bundle is an offline research artifact. It binds the admission policy,
source revisions, transformed feature values, raw model logits, calibration,
operator thresholds, request metadata, and the evidence receipt into one
canonical JSON envelope. A trusted caller must retain the bundle SHA-256
outside the bundle and supply it during replay.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from iris_report.iris_sep.src.iris_sep.pilot_admission_v2 import AdmissionPolicyV2, replay_forecast_v2
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import OperatorRuntimePolicy


FORMAT = "IRIS_SEP_INFERENCE_BUNDLE_V1"
SCOPE = "OFFLINE_RESEARCH_REPLAY_NOT_OPERATIONALLY_CERTIFIED"


class InferenceBundleError(ValueError):
    """Raised when immutable inference evidence is malformed or fails binding."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InferenceBundleError("bundle payload is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InferenceBundleError("timezone-aware datetime required")
    return value.isoformat()


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise InferenceBundleError("timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InferenceBundleError("invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InferenceBundleError("timestamp must be timezone-aware")
    return parsed


def _encode_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise InferenceBundleError("bound arrays must be non-empty")
    if not np.isfinite(array).all():
        raise InferenceBundleError("builder refuses nonfinite bound arrays")
    array = np.ascontiguousarray(array.astype("<f8", copy=False))
    raw = array.tobytes(order="C")
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": _sha256(raw),
        "bytes_b64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_array(payload: Any) -> np.ndarray:
    if not isinstance(payload, Mapping) or payload.get("dtype") != "<f8":
        raise InferenceBundleError("unsupported array encoding")
    shape = payload.get("shape")
    if not isinstance(shape, list) or any(not isinstance(v, int) or v < 0 for v in shape):
        raise InferenceBundleError("invalid array shape")
    try:
        raw = base64.b64decode(payload["bytes_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise InferenceBundleError("invalid array bytes") from exc
    if _sha256(raw) != payload.get("sha256"):
        raise InferenceBundleError("array digest mismatch")
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(raw) != expected * 8:
        raise InferenceBundleError("array byte length does not match shape")
    return np.frombuffer(raw, dtype="<f8").reshape(shape).copy()


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _admission_payload(policy: AdmissionPolicyV2) -> dict[str, Any]:
    return {
        "supported_from": _iso(policy.supported_from),
        "supported_through": _iso(policy.supported_through),
        "allowed_source_revisions": {k: list(v) for k, v in sorted(policy.allowed_source_revisions.items())},
        "maximum_abs_standardized_feature": float(policy.maximum_abs_standardized_feature),
    }


def _runtime_payload(policy: OperatorRuntimePolicy) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "calibration_id": policy.calibration_id,
        "schema_sha256": policy.schema_sha256,
        "operating_thresholds": {k: float(v) for k, v in sorted(policy.operating_thresholds.items())},
        "maximum_age_minutes": {k: float(v) for k, v in sorted(policy.maximum_age_minutes.items())},
        "critical_modalities": list(policy.critical_modalities),
    }


def static_inference_binding_sha256(*, admission_policy: AdmissionPolicyV2, runtime_policy: OperatorRuntimePolicy, calibration_intercept: float, model_version: str, input_schema_sha256: str) -> str:
    """Hash the static inference contract that a trusted evidence receipt must bind."""
    if not isinstance(calibration_intercept, (int, float)) or isinstance(calibration_intercept, bool) or not math.isfinite(calibration_intercept):
        raise InferenceBundleError("finite calibration intercept required")
    payload = {
        "format": FORMAT,
        "admission_policy": _admission_payload(admission_policy),
        "runtime_policy": _runtime_payload(runtime_policy),
        "calibration": {"calibration_id": runtime_policy.calibration_id, "method": "LOGIT_INTERCEPT_ONLY", "intercept": float(calibration_intercept)},
        "threshold": {"policy_id": runtime_policy.policy_id, "operating_thresholds": {k: float(v) for k, v in sorted(runtime_policy.operating_thresholds.items())}},
        "model_version": model_version,
        "input_schema_sha256": input_schema_sha256,
    }
    return _sha256(_canonical_json(payload))


def build_inference_bundle(
    *,
    admission_policy: AdmissionPolicyV2,
    runtime_policy: OperatorRuntimePolicy,
    source_revisions: Mapping[str, str],
    transformed_features: Any,
    model_outputs: Any,
    calibration_intercept: float,
    evidence_bytes: bytes,
    expected_evidence_sha256: str,
    issued_at: datetime,
    input_schema_sha256: str,
    data_freshness: Mapping[str, Mapping[str, Any]],
    missing_modalities: list[str] | tuple[str, ...],
    uncertainty: Mapping[str, Any],
    model_version: str,
) -> tuple[bytes, str]:
    """Create canonical bundle bytes and the external SHA-256 trust anchor."""
    if not isinstance(evidence_bytes, bytes) or _sha256(evidence_bytes) != expected_evidence_sha256:
        raise InferenceBundleError("evidence receipt digest mismatch")
    binding_sha = static_inference_binding_sha256(admission_policy=admission_policy, runtime_policy=runtime_policy, calibration_intercept=calibration_intercept, model_version=model_version, input_schema_sha256=input_schema_sha256)
    try:
        evidence_object = json.loads(evidence_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceBundleError("evidence receipt is not valid JSON") from exc
    if not isinstance(evidence_object, Mapping) or evidence_object.get("inference_binding_sha256") != binding_sha:
        raise InferenceBundleError("evidence receipt does not bind the static inference contract")
    features = _encode_array(transformed_features)
    outputs = _encode_array(model_outputs)
    output_array = _decode_array(outputs)
    aggregated_logit = float(np.median(output_array))
    calibrated_probability = _sigmoid(aggregated_logit + float(calibration_intercept))
    source_revisions = {str(k): str(v) for k, v in source_revisions.items()}
    request = {
        "issued_at": _iso(issued_at),
        "input_schema_sha256": input_schema_sha256,
        "data_freshness": {str(k): dict(v) for k, v in data_freshness.items()},
        "missing_modalities": sorted(set(str(v) for v in missing_modalities)),
        "uncertainty": dict(uncertainty),
        "model_version": model_version,
    }
    payload = {
        "format": FORMAT,
        "scope": SCOPE,
        "admission_policy": _admission_payload(admission_policy),
        "runtime_policy": _runtime_payload(runtime_policy),
        "source_revisions": dict(sorted(source_revisions.items())),
        "arrays": {"transformed_features": features, "model_outputs": outputs},
        "calibration": {
            "calibration_id": runtime_policy.calibration_id,
            "method": "LOGIT_INTERCEPT_ONLY",
            "intercept": float(calibration_intercept),
        },
        "derived": {
            "seed_probability_aggregation": "MEDIAN_LOGIT",
            "aggregated_logit": aggregated_logit,
            "calibrated_probability": calibrated_probability,
        },
        "threshold": {
            "policy_id": runtime_policy.policy_id,
            "operating_thresholds": {k: float(v) for k, v in sorted(runtime_policy.operating_thresholds.items())},
        },
        "evidence_receipt": {
            "sha256": expected_evidence_sha256,
            "inference_binding_sha256": binding_sha,
            "bytes_b64": base64.b64encode(evidence_bytes).decode("ascii"),
        },
        "request": request,
    }
    payload_bytes = _canonical_json(payload)
    envelope = {"payload": payload, "payload_sha256": _sha256(payload_bytes)}
    bundle_bytes = _canonical_json(envelope)
    return bundle_bytes, _sha256(bundle_bytes)


def replay_inference_bundle(*, bundle_bytes: bytes, expected_bundle_sha256: str) -> dict[str, Any]:
    """Verify every binding and replay admission V2 without caller-supplied arrays/policy."""
    if not isinstance(bundle_bytes, bytes) or _sha256(bundle_bytes) != expected_bundle_sha256:
        raise InferenceBundleError("bundle trust-anchor mismatch")
    try:
        envelope = json.loads(bundle_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceBundleError("bundle is not valid JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
        raise InferenceBundleError("unexpected bundle envelope")
    payload = envelope["payload"]
    if _sha256(_canonical_json(payload)) != envelope["payload_sha256"]:
        raise InferenceBundleError("payload digest mismatch")
    if payload.get("format") != FORMAT or payload.get("scope") != SCOPE:
        raise InferenceBundleError("unsupported bundle format or scope")

    admission_raw = payload.get("admission_policy")
    runtime_raw = payload.get("runtime_policy")
    if not isinstance(admission_raw, Mapping) or not isinstance(runtime_raw, Mapping):
        raise InferenceBundleError("policy payload missing")
    admission = AdmissionPolicyV2(
        supported_from=_parse_time(admission_raw.get("supported_from")),
        supported_through=_parse_time(admission_raw.get("supported_through")),
        allowed_source_revisions={str(k): tuple(str(x) for x in v) for k, v in admission_raw.get("allowed_source_revisions", {}).items()},
        maximum_abs_standardized_feature=float(admission_raw.get("maximum_abs_standardized_feature")),
    )
    runtime = OperatorRuntimePolicy(
        policy_id=str(runtime_raw.get("policy_id", "")),
        calibration_id=str(runtime_raw.get("calibration_id", "")),
        schema_sha256=str(runtime_raw.get("schema_sha256", "")),
        operating_thresholds={str(k): float(v) for k, v in runtime_raw.get("operating_thresholds", {}).items()},
        maximum_age_minutes={str(k): float(v) for k, v in runtime_raw.get("maximum_age_minutes", {}).items()},
        critical_modalities=tuple(str(v) for v in runtime_raw.get("critical_modalities", ())),
    )

    arrays = payload.get("arrays")
    if not isinstance(arrays, Mapping):
        raise InferenceBundleError("bound arrays missing")
    features = _decode_array(arrays.get("transformed_features"))
    outputs = _decode_array(arrays.get("model_outputs"))
    if not np.isfinite(features).all() or not np.isfinite(outputs).all():
        raise InferenceBundleError("nonfinite values in immutable bundle")

    calibration = payload.get("calibration")
    threshold = payload.get("threshold")
    derived = payload.get("derived")
    if not isinstance(calibration, Mapping) or calibration.get("method") != "LOGIT_INTERCEPT_ONLY":
        raise InferenceBundleError("unsupported calibration binding")
    if calibration.get("calibration_id") != runtime.calibration_id:
        raise InferenceBundleError("calibration identifier mismatch")
    intercept = float(calibration.get("intercept"))
    if not math.isfinite(intercept):
        raise InferenceBundleError("nonfinite calibration parameter")
    if not isinstance(threshold, Mapping) or threshold.get("policy_id") != runtime.policy_id or threshold.get("operating_thresholds") != {k: float(v) for k, v in sorted(runtime.operating_thresholds.items())}:
        raise InferenceBundleError("threshold binding mismatch")
    aggregated = float(np.median(outputs))
    calibrated = _sigmoid(aggregated + intercept)
    if not isinstance(derived, Mapping) or derived.get("seed_probability_aggregation") != "MEDIAN_LOGIT":
        raise InferenceBundleError("aggregation binding mismatch")
    if not math.isclose(float(derived.get("aggregated_logit")), aggregated, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(float(derived.get("calibrated_probability")), calibrated, rel_tol=0.0, abs_tol=1e-15):
        raise InferenceBundleError("derived probability binding mismatch")

    evidence = payload.get("evidence_receipt")
    if not isinstance(evidence, Mapping):
        raise InferenceBundleError("evidence receipt binding missing")
    try:
        evidence_bytes = base64.b64decode(evidence["bytes_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise InferenceBundleError("invalid evidence receipt bytes") from exc
    evidence_sha = str(evidence.get("sha256", ""))
    if _sha256(evidence_bytes) != evidence_sha:
        raise InferenceBundleError("evidence receipt digest mismatch")
    try:
        evidence_object = json.loads(evidence_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceBundleError("evidence receipt is not valid JSON") from exc

    request = payload.get("request")
    if not isinstance(request, Mapping):
        raise InferenceBundleError("request binding missing")
    missing = list(request.get("missing_modalities", ()))
    freshness = request.get("data_freshness")
    if not isinstance(freshness, Mapping):
        raise InferenceBundleError("freshness binding missing")
    bound_revisions = payload.get("source_revisions")
    if not isinstance(bound_revisions, Mapping):
        raise InferenceBundleError("source revision binding missing")
    actual_revisions: dict[str, str] = {}
    for modality in runtime.maximum_age_minutes:
        if modality in missing:
            continue
        record = freshness.get(modality)
        if not isinstance(record, Mapping) or not isinstance(record.get("source_revision"), str):
            raise InferenceBundleError("source revision absent from freshness record")
        actual_revisions[modality] = record["source_revision"]
    if dict(sorted(actual_revisions.items())) != dict(sorted((str(k), str(v)) for k, v in bound_revisions.items())):
        raise InferenceBundleError("source revision snapshot mismatch")
    binding_sha = static_inference_binding_sha256(admission_policy=admission, runtime_policy=runtime, calibration_intercept=intercept, model_version=str(request.get("model_version", "")), input_schema_sha256=str(request.get("input_schema_sha256", "")))
    if evidence.get("inference_binding_sha256") != binding_sha or not isinstance(evidence_object, Mapping) or evidence_object.get("inference_binding_sha256") != binding_sha:
        raise InferenceBundleError("evidence receipt static inference binding mismatch")

    result = replay_forecast_v2(
        admission_policy=admission,
        transformed_features=features,
        model_outputs=outputs,
        evidence_bytes=evidence_bytes,
        expected_evidence_sha256=evidence_sha,
        issued_at=_parse_time(request.get("issued_at")),
        calibrated_probability=calibrated,
        runtime_policy=runtime,
        input_schema_sha256=str(request.get("input_schema_sha256", "")),
        data_freshness={str(k): dict(v) for k, v in freshness.items()},
        missing_modalities=missing,
        uncertainty=dict(request.get("uncertainty", {})),
        model_version=str(request.get("model_version", "")),
    )
    result["inference_bundle_sha256"] = expected_bundle_sha256
    result["inference_bundle_payload_sha256"] = envelope["payload_sha256"]
    result["inference_bundle_scope"] = SCOPE
    return result
