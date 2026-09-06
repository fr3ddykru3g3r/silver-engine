"""Receipt-bound offline research replay. Fixtures are never benchmark evidence."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import build_operator_forecast


def replay_forecast(*, evidence_bytes: bytes | None, expected_evidence_sha256: str,
                    **request):
    """Verify evidence bindings and forecast-time availability before serialization.

    Caller must supply the independently trusted receipt digest. This verifies
    integrity, not publisher authenticity or scientific approval.
    """
    reasons = list(request.pop('abstention_reasons', ()))
    policy = request['runtime_policy']
    evidence = {}
    try:
        if evidence_bytes is None or hashlib.sha256(evidence_bytes).hexdigest() != expected_evidence_sha256:
            raise ValueError('missing or mutated evidence')
        evidence = json.loads(evidence_bytes)
        bindings = {'model_version': request['model_version'], 'schema_sha256': policy.schema_sha256,
                    'policy_id': policy.policy_id, 'calibration_id': policy.calibration_id,
                    'operating_thresholds': dict(policy.operating_thresholds),
                    'maximum_age_minutes': dict(policy.maximum_age_minutes),
                    'critical_modalities': list(policy.critical_modalities)}
        if not isinstance(evidence, dict):
            raise ValueError('receipt must be an object')
        if evidence.get('bindings') != bindings:
            reasons.append('MODEL_OR_POLICY_VERSION_MISMATCH')
        if evidence.get('scope') != 'SYNTHETIC_FIXTURE_ONLY' or evidence.get('locked_test_accessed') is not False:
            raise ValueError('only synthetic replay evidence accepted by this tool')
    except (ValueError, TypeError):
        reasons.append('EVIDENCE_RECEIPT_FAILURE')
    if any(not math.isfinite(v) for v in policy.maximum_age_minutes.values()):
        reasons.append('MODEL_OR_POLICY_VERSION_MISMATCH')
    issue = request['issued_at']
    freshness = {}
    for modality in policy.maximum_age_minutes:
        if modality in request['missing_modalities']:
            continue
        record = request['data_freshness'].get(modality, {})
        try:
            obs = datetime.fromisoformat(record['observed_at_utc'].replace('Z', '+00:00'))
            pub = datetime.fromisoformat(record['published_at_utc'].replace('Z', '+00:00'))
            if obs.tzinfo is None or pub.tzinfo is None or not obs <= pub <= issue:
                raise ValueError('not forecast-time available')
            age = (issue - obs).total_seconds() / 60
            freshness[modality] = dict(record, age_minutes=age)
        except (KeyError, ValueError, TypeError):
            reasons.append('INPUT_AVAILABILITY_FAILURE')
    uncertainty = request['uncertainty']
    required = ('between_seed_spread', 'calibration_uncertainty', 'input_quality')
    if any(not isinstance(uncertainty.get(k), (int, float)) or isinstance(uncertainty.get(k), bool)
           or not math.isfinite(uncertainty[k]) or not 0 <= uncertainty[k] <= 1 for k in required):
        reasons.append('UNCERTAINTY_FAILURE')
    request['data_freshness'] = freshness
    # Low-level builder requires a digest even for an abstention. Never invent one.
    digest = hashlib.sha256(evidence_bytes).hexdigest() if evidence_bytes is not None else '0' * 64
    result = build_operator_forecast(**request, evidence_receipt_sha256=digest,
                                     abstention_reasons=sorted(set(reasons)))
    result['evidence_receipt_sha256'] = digest if evidence_bytes is not None else None
    result['replay_scope'] = 'SYNTHETIC_FIXTURE_ONLY_NOT_FORECAST_EVIDENCE'
    return result
