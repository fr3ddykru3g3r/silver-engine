"""Fail-closed, advisory-only operator forecast contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
STATES = ("NORMAL", "MONITOR", "PREPARE", "PROTECT")


class OperatorContractError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorRuntimePolicy:
    policy_id: str
    calibration_id: str
    schema_sha256: str
    operating_thresholds: Mapping[str, float]
    maximum_age_minutes: Mapping[str, float]
    critical_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        modalities = set(self.maximum_age_minutes)
        if not self.policy_id or not self.calibration_id or _SHA256.fullmatch(self.schema_sha256) is None:
            raise OperatorContractError("runtime policy identifiers and schema SHA-256 are required")
        if modalities != {"magnetic", "eruption", "particle_context"}:
            raise OperatorContractError("runtime policy must define all three frozen modalities")
        if not set(self.critical_modalities) or not set(self.critical_modalities).issubset(modalities):
            raise OperatorContractError("critical modalities must be a non-empty frozen subset")
        if any(not isinstance(age, (int, float)) or age <= 0 for age in self.maximum_age_minutes.values()):
            raise OperatorContractError("maximum ages must be positive minutes")
        _state(0.0, self.operating_thresholds)


def _utc_z(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OperatorContractError("issued_at must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state(probability: float, thresholds: Mapping[str, float]) -> str:
    required = ("MONITOR", "PREPARE", "PROTECT")
    if tuple(sorted(thresholds, key=lambda name: required.index(name) if name in required else 99)) != required:
        if set(thresholds) != set(required):
            raise OperatorContractError("policy thresholds must define MONITOR, PREPARE, and PROTECT")
    values = [float(thresholds[name]) for name in required]
    if not 0 <= values[0] < values[1] < values[2] <= 1:
        raise OperatorContractError("operator thresholds must be strictly increasing in [0,1]")
    if probability >= values[2]: return "PROTECT"
    if probability >= values[1]: return "PREPARE"
    if probability >= values[0]: return "MONITOR"
    return "NORMAL"


def build_operator_forecast(
    *,
    issued_at: datetime,
    calibrated_probability: float | None,
    runtime_policy: OperatorRuntimePolicy,
    input_schema_sha256: str,
    data_freshness: Mapping[str, Mapping[str, Any]],
    missing_modalities: Sequence[str],
    uncertainty: Mapping[str, Any],
    model_version: str,
    evidence_receipt_sha256: str,
    abstention_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one advisory forecast; never emit a spacecraft command."""

    if not model_version:
        raise OperatorContractError("model version is required")
    if _SHA256.fullmatch(evidence_receipt_sha256) is None:
        raise OperatorContractError("evidence receipt must be a lowercase SHA-256")
    if input_schema_sha256 != runtime_policy.schema_sha256:
        abstention_reasons = tuple(abstention_reasons) + ("SCHEMA_FAILURE",)
    allowed_modalities = set(runtime_policy.maximum_age_minutes)
    missing = sorted(set(missing_modalities))
    if not set(missing).issubset(allowed_modalities) or not set(data_freshness).issubset(allowed_modalities):
        raise OperatorContractError("unknown modality in runtime inputs")
    critical_missing = sorted(set(missing).intersection(runtime_policy.critical_modalities))
    reasons = sorted(set(abstention_reasons).union({"CRITICAL_INPUT_MISSING"} if critical_missing else set()))
    for modality in sorted(allowed_modalities - set(missing)):
        record = data_freshness.get(modality)
        age = record.get("age_minutes") if isinstance(record, Mapping) else None
        if not isinstance(age, (int, float)) or age < 0 or age > runtime_policy.maximum_age_minutes[modality]:
            reasons.append("INPUT_TOO_STALE")
            break
    if calibrated_probability is not None and not 0 <= float(calibrated_probability) <= 1:
        raise OperatorContractError("calibrated probability must be in [0,1] or null")
    if reasons or calibrated_probability is None:
        if not reasons:
            reasons = ["EVIDENCE_RECEIPT_FAILURE"]
        status = "ABSTAIN"
        probability = None
        all_clear = None
        operator_state = None
    else:
        probability = float(calibrated_probability)
        all_clear = 1.0 - probability
        operator_state = _state(probability, runtime_policy.operating_thresholds)
        status = "DEGRADED" if missing else "VALID"
    return {
        "issued_at_utc": _utc_z(issued_at),
        "horizon_hours": 24,
        "forecast_status": status,
        "p_new_sep_10mev_10pfu_within_24h": probability,
        "all_clear_probability": all_clear,
        "calibration_id": runtime_policy.calibration_id,
        "operating_policy_id": runtime_policy.policy_id,
        "input_schema_sha256": input_schema_sha256,
        "data_freshness": dict(data_freshness),
        "missing_modalities": missing,
        "uncertainty": dict(uncertainty),
        "operator_state": operator_state,
        "abstention_reasons": reasons,
        "model_version": model_version,
        "evidence_receipt_sha256": evidence_receipt_sha256,
        "spacecraft_control": False,
        "product_class": "RESEARCH_DECISION_SUPPORT_NOT_OPERATIONALLY_CERTIFIED",
        "secondary_outputs": {
            "p_sep_100mev_1pfu": None,
            "peak_flux_quantiles_pfu": None,
            "onset_time_quantiles_utc": None,
            "validation_status": "DISABLED_PENDING_INDEPENDENT_VALIDATION"
        }
    }
