"""Sealed prospective evaluation primitives for IRIS-SEP.

Forecasts are sealed before their 24-hour outcomes are known. Outcome labels are
created only when primary >10 MeV proton observations provide fresh support at
the forecast issue and complete five-minute-or-better coverage through the exact
24-hour horizon endpoint. Incomplete or immature windows remain unresolved.

This module contains no training, calibration or threshold-selection path. It
also deliberately separates numerical threshold crossings from permission-
filtered alerts and never upgrades sample-size sufficiency into an independence
claim without an external witness/custodian mechanism.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FORMAT = "IRIS_SEP_SEALED_FORECAST_V2"
LABEL_FORMAT = "IRIS_SEP_SEALED_NEW_CROSSING_LABELS_V2"
EVALUATION_FORMAT = "IRIS_SEP_SEALED_COHORT_EVALUATION_V2"
TARGET = "NEW_GT10MEV_GE10PFU_CROSSING_WITHIN_24H"
HORIZON = timedelta(hours=24)
MAX_SEAL_DELAY = timedelta(minutes=5)
MAX_OUTCOME_GAP = timedelta(minutes=5)
STATES = ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")
POLICIES = ("MAX_TSS", "POD80_MIN_FAR")


class SealedEvaluationError(ValueError):
    """Raised when prospective sealing or outcome evaluation is invalid."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SealedEvaluationError("sealed evaluation payload is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SealedEvaluationError(f"{name} must be lowercase SHA-256")
    return value


def _time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SealedEvaluationError(f"{name} is invalid") from exc
    else:
        raise SealedEvaluationError(f"{name} must be datetime/ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SealedEvaluationError(f"{name} must be timezone-aware")
    return parsed


def _probability(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SealedEvaluationError(f"{name} must be finite")
    out = float(value)
    if not 0 <= out <= 1:
        raise SealedEvaluationError(f"{name} outside [0,1]")
    return out


def _validate_state_payloads(
    *,
    probabilities: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, Any]],
    operator_permissions: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, str]]:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(STATES):
        raise SealedEvaluationError("all four availability-state probabilities are required")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(STATES):
        raise SealedEvaluationError("threshold state coverage mismatch")
    if not isinstance(operator_permissions, Mapping) or set(operator_permissions) != set(STATES):
        raise SealedEvaluationError("operator-permission state coverage mismatch")

    probability_payload = {
        state: _probability(probabilities[state], f"probability {state}")
        for state in STATES
    }
    threshold_payload: dict[str, dict[str, float]] = {}
    for state in STATES:
        policies = thresholds[state]
        if not isinstance(policies, Mapping) or set(policies) != set(POLICIES):
            raise SealedEvaluationError("both frozen threshold policies required for every state")
        threshold_payload[state] = {
            policy: _probability(policies[policy], f"threshold {state}/{policy}")
            for policy in POLICIES
        }

    permission_payload: dict[str, str] = {}
    for state in STATES:
        value = operator_permissions[state]
        if not isinstance(value, str) or not value.strip():
            raise SealedEvaluationError(f"operator permission for {state} must be a non-empty string")
        permission_payload[state] = value.strip()
    return probability_payload, threshold_payload, permission_payload


def build_forecast_seal(
    *,
    issued_at: datetime,
    sealed_at: datetime,
    package_manifest_sha256: str,
    feature_row_sha256: str,
    source_authentication_sha256: str,
    causal_feature_derivation_sha256: str,
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
    operator_permissions: Mapping[str, str],
    architecture_id: str,
) -> dict[str, Any]:
    """Create an immutable forecast record before the outcome window matures."""
    issue = _time(issued_at, "issued_at")
    sealed = _time(sealed_at, "sealed_at")
    if sealed < issue:
        raise SealedEvaluationError("sealed_at cannot precede issued_at")
    if sealed - issue > MAX_SEAL_DELAY:
        raise SealedEvaluationError("forecast must be sealed within five minutes of issuance")
    if not isinstance(architecture_id, str) or not architecture_id.strip():
        raise SealedEvaluationError("architecture_id must be a non-empty string")

    probability_payload, threshold_payload, permission_payload = _validate_state_payloads(
        probabilities=probabilities,
        thresholds=thresholds,
        operator_permissions=operator_permissions,
    )
    payload = {
        "format": FORMAT,
        "target": TARGET,
        "issued_at": issue.isoformat(),
        "sealed_at": sealed.isoformat(),
        "outcome_window_end": (issue + HORIZON).isoformat(),
        "architecture_id": architecture_id.strip(),
        "package_manifest_sha256": _sha(package_manifest_sha256, "package_manifest_sha256"),
        "feature_row_sha256": _sha(feature_row_sha256, "feature_row_sha256"),
        "source_authentication_sha256": _sha(source_authentication_sha256, "source_authentication_sha256"),
        "causal_feature_derivation_sha256": _sha(
            causal_feature_derivation_sha256,
            "causal_feature_derivation_sha256",
        ),
        "probabilities": probability_payload,
        "thresholds": threshold_payload,
        "operator_permissions": permission_payload,
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
    }
    payload["forecast_seal_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_forecast_seal(seal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate both the digest and all forecast semantics.

    A self-consistent hash is not sufficient: invalid probabilities, thresholds,
    horizons or permissions are rejected even if a caller recomputes the hash.
    """
    if not isinstance(seal, Mapping) or seal.get("format") != FORMAT or seal.get("target") != TARGET:
        raise SealedEvaluationError("unsupported forecast seal")
    claimed = seal.get("forecast_seal_sha256")
    unsigned = dict(seal)
    unsigned.pop("forecast_seal_sha256", None)
    if _sha(claimed, "forecast_seal_sha256") != sha256_bytes(_canonical_json(unsigned)):
        raise SealedEvaluationError("forecast seal digest mismatch")

    issue = _time(seal.get("issued_at"), "issued_at")
    sealed = _time(seal.get("sealed_at"), "sealed_at")
    if sealed < issue or sealed - issue > MAX_SEAL_DELAY:
        raise SealedEvaluationError("forecast was not sealed at issue time")
    horizon = _time(seal.get("outcome_window_end"), "outcome_window_end")
    if horizon != issue + HORIZON:
        raise SealedEvaluationError("forecast outcome horizon does not equal issue plus 24 hours")
    architecture = seal.get("architecture_id")
    if not isinstance(architecture, str) or not architecture.strip():
        raise SealedEvaluationError("forecast architecture_id is invalid")

    _sha(seal.get("package_manifest_sha256"), "package_manifest_sha256")
    _sha(seal.get("feature_row_sha256"), "feature_row_sha256")
    _sha(seal.get("source_authentication_sha256"), "source_authentication_sha256")
    _sha(seal.get("causal_feature_derivation_sha256"), "causal_feature_derivation_sha256")
    _validate_state_payloads(
        probabilities=seal.get("probabilities", {}),
        thresholds=seal.get("thresholds", {}),
        operator_permissions=seal.get("operator_permissions", {}),
    )
    if any(
        seal.get(key) is not False
        for key in (
            "runtime_training_allowed",
            "runtime_recalibration_allowed",
            "runtime_rethresholding_allowed",
        )
    ):
        raise SealedEvaluationError("forecast seal permits runtime model changes")
    return dict(seal)


def _normalize_outcome_series(
    proton_times: Sequence[Any],
    proton_flux: Sequence[Any],
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = pd.to_datetime(list(proton_times), utc=True, errors="coerce")
    flux = pd.to_numeric(pd.Series(list(proton_flux)), errors="coerce").to_numpy(dtype=float)
    if len(times) != len(flux) or len(times) < 2 or bool(pd.isna(times).any()):
        raise SealedEvaluationError("proton outcome series is invalid")
    if not np.isfinite(flux).all() or np.any(flux < 0):
        raise SealedEvaluationError("proton outcome series must be finite and nonnegative")

    order = np.argsort(pd.DatetimeIndex(times).view("i8"), kind="mergesort")
    times = pd.DatetimeIndex(times[order])
    flux = flux[order]
    if bool(times.duplicated().any()):
        raise SealedEvaluationError("duplicate proton outcome timestamps are not permitted")
    return times, flux


def _coverage_for_issue(
    *,
    times: pd.DatetimeIndex,
    flux: np.ndarray,
    issue: pd.Timestamp,
) -> dict[str, Any]:
    end = issue + pd.Timedelta(hours=24)
    prior = np.flatnonzero(times <= issue)
    if len(prior) == 0:
        return {"resolved": False, "reason": "ISSUE_SUPPORT_MISSING"}
    issue_idx = int(prior[-1])
    issue_sample = times[issue_idx]
    issue_lag = issue - issue_sample
    if issue_lag < pd.Timedelta(0) or issue_lag > pd.Timedelta(minutes=5):
        return {
            "resolved": False,
            "reason": "ISSUE_SUPPORT_STALE",
            "issue_support_lag_seconds": float(issue_lag.total_seconds()),
        }
    if times[-1] < end:
        return {
            "resolved": False,
            "reason": "OUTCOME_WINDOW_IMMATURE",
            "issue_support_lag_seconds": float(issue_lag.total_seconds()),
        }

    endpoint_matches = np.flatnonzero(times == end)
    if len(endpoint_matches) != 1:
        return {
            "resolved": False,
            "reason": "OUTCOME_HORIZON_ENDPOINT_MISSING",
            "issue_support_lag_seconds": float(issue_lag.total_seconds()),
        }
    end_idx = int(endpoint_matches[0])
    if end_idx <= issue_idx:
        return {"resolved": False, "reason": "OUTCOME_WINDOW_INVALID"}

    covered_times = times[issue_idx : end_idx + 1]
    gaps = [
        float((b - a).total_seconds())
        for a, b in zip(covered_times[:-1], covered_times[1:])
    ]
    maximum = max(gaps) if gaps else math.inf
    if not math.isfinite(maximum) or maximum > MAX_OUTCOME_GAP.total_seconds():
        return {
            "resolved": False,
            "reason": "OUTCOME_GAP_EXCEEDS_5_MINUTES",
            "issue_support_lag_seconds": float(issue_lag.total_seconds()),
            "max_gap_seconds": maximum,
        }
    return {
        "resolved": True,
        "reason": "COMPLETE_PRIMARY_SERIES_SUPPORT",
        "issue_index": issue_idx,
        "end_index": end_idx,
        "issue_support_lag_seconds": float(issue_lag.total_seconds()),
        "max_gap_seconds": maximum,
        "sample_count": int(end_idx - issue_idx + 1),
    }


def derive_new_crossing_labels(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    proton_times: Sequence[Any],
    proton_flux: Sequence[Any],
    threshold_pfu: float = 10.0,
) -> dict[str, Any]:
    """Derive sampled NEW-crossing outcomes from complete primary proton data.

    A row is resolved only when:
    - the issue has a primary sample no more than five minutes old;
    - the 24-hour horizon has matured;
    - an observation exists at the exact horizon endpoint; and
    - every gap from issue support through the endpoint is <= five minutes.

    Incomplete windows remain unresolved even if an apparent crossing occurs
    before a later gap. This prevents optimistic positive labels from incomplete
    outcome data. Equivalence between this sampled crossing rule and any external
    catalogue event definition remains a separate validation requirement.
    """
    if not math.isfinite(float(threshold_pfu)) or threshold_pfu <= 0:
        raise SealedEvaluationError("threshold_pfu must be positive")
    times, flux = _normalize_outcome_series(proton_times, proton_flux)

    seen_hashes: set[str] = set()
    seen_issues: set[str] = set()
    rows = []
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        seal_sha = str(seal["forecast_seal_sha256"])
        issue = pd.Timestamp(_time(seal["issued_at"], "issued_at"))
        issue_key = issue.isoformat()
        if seal_sha in seen_hashes:
            raise SealedEvaluationError("duplicate forecast seal in outcome request")
        if issue_key in seen_issues:
            raise SealedEvaluationError("duplicate forecast issue time in outcome request")
        seen_hashes.add(seal_sha)
        seen_issues.add(issue_key)

        coverage = _coverage_for_issue(times=times, flux=flux, issue=issue)
        row: dict[str, Any] = {
            "forecast_seal_sha256": seal_sha,
            "issued_at": issue_key,
            "outcome_window_end": (issue + pd.Timedelta(hours=24)).isoformat(),
            "outcome_resolved": bool(coverage["resolved"]),
            "resolution_reason": str(coverage["reason"]),
            "eligible_new_crossing_issue": None,
            "label": None,
            "current_flux_pfu": None,
            "first_crossing_utc": None,
            "issue_support_lag_seconds": coverage.get("issue_support_lag_seconds"),
            "max_gap_seconds": coverage.get("max_gap_seconds"),
            "outcome_sample_count": coverage.get("sample_count", 0),
        }
        if not coverage["resolved"]:
            rows.append(row)
            continue

        issue_idx = int(coverage["issue_index"])
        end_idx = int(coverage["end_index"])
        current_flux = float(flux[issue_idx])
        eligible = current_flux < float(threshold_pfu)
        row["current_flux_pfu"] = current_flux
        row["eligible_new_crossing_issue"] = bool(eligible)
        if not eligible:
            rows.append(row)
            continue

        label = 0
        previous = current_flux
        for idx in range(issue_idx + 1, end_idx + 1):
            value = float(flux[idx])
            if previous < threshold_pfu <= value:
                label = 1
                row["first_crossing_utc"] = times[idx].isoformat()
                break
            previous = value
        row["label"] = label
        rows.append(row)

    payload = {
        "format": LABEL_FORMAT,
        "target": TARGET,
        "threshold_pfu": float(threshold_pfu),
        "forecast_count": len(rows),
        "resolved_count": sum(int(row["outcome_resolved"]) for row in rows),
        "unresolved_count": sum(int(not row["outcome_resolved"]) for row in rows),
        "eligible_resolved_count": sum(
            int(row["outcome_resolved"] and row["eligible_new_crossing_issue"] is True)
            for row in rows
        ),
        "positive_count": sum(int(row["label"] == 1) for row in rows),
        "maximum_permitted_gap_seconds": float(MAX_OUTCOME_GAP.total_seconds()),
        "exact_horizon_endpoint_required": True,
        "duplicate_timestamps_permitted": False,
        "sampled_crossing_equivalence_to_catalogue_definition_established": False,
        "rows": rows,
    }
    payload["label_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_label_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate V2 label receipt semantics before any model/comparator scoring."""
    if not isinstance(receipt, Mapping) or receipt.get("format") != LABEL_FORMAT or receipt.get("target") != TARGET:
        raise SealedEvaluationError("unsupported outcome-label receipt; V2 required")
    unsigned = dict(receipt)
    claimed = unsigned.pop("label_receipt_sha256", None)
    if _sha(claimed, "label_receipt_sha256") != sha256_bytes(_canonical_json(unsigned)):
        raise SealedEvaluationError("outcome-label receipt digest mismatch")
    if float(receipt.get("maximum_permitted_gap_seconds", -1)) != MAX_OUTCOME_GAP.total_seconds():
        raise SealedEvaluationError("outcome-label gap contract mismatch")
    if receipt.get("exact_horizon_endpoint_required") is not True:
        raise SealedEvaluationError("outcome-label receipt does not require exact horizon support")
    if receipt.get("duplicate_timestamps_permitted") is not False:
        raise SealedEvaluationError("outcome-label receipt permits duplicate timestamps")

    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise SealedEvaluationError("outcome-label rows missing")
    seen_hashes: set[str] = set()
    seen_issues: set[str] = set()
    resolved = unresolved = eligible = positives = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise SealedEvaluationError("invalid outcome-label row")
        seal_sha = _sha(row.get("forecast_seal_sha256"), "forecast_seal_sha256")
        issue = _time(row.get("issued_at"), "label issued_at").isoformat()
        end = _time(row.get("outcome_window_end"), "label outcome_window_end")
        if end != _time(issue, "label issued_at") + HORIZON:
            raise SealedEvaluationError("label outcome horizon mismatch")
        if seal_sha in seen_hashes or issue in seen_issues:
            raise SealedEvaluationError("duplicate forecast identity in outcome-label receipt")
        seen_hashes.add(seal_sha)
        seen_issues.add(issue)

        is_resolved = row.get("outcome_resolved")
        if not isinstance(is_resolved, bool):
            raise SealedEvaluationError("outcome_resolved must be boolean")
        reason = row.get("resolution_reason")
        if not isinstance(reason, str) or not reason:
            raise SealedEvaluationError("resolution_reason missing")
        if is_resolved:
            resolved += 1
            current = row.get("current_flux_pfu")
            if not isinstance(current, (int, float)) or isinstance(current, bool) or not math.isfinite(float(current)) or float(current) < 0:
                raise SealedEvaluationError("resolved outcome missing valid current flux")
            eligible_flag = row.get("eligible_new_crossing_issue")
            if not isinstance(eligible_flag, bool):
                raise SealedEvaluationError("resolved outcome eligibility must be boolean")
            if eligible_flag:
                eligible += 1
                if row.get("label") not in (0, 1):
                    raise SealedEvaluationError("eligible resolved outcome requires binary label")
                positives += int(row["label"] == 1)
            elif row.get("label") is not None:
                raise SealedEvaluationError("ineligible issue must not carry a binary label")
        else:
            unresolved += 1
            if row.get("eligible_new_crossing_issue") is not None or row.get("label") is not None:
                raise SealedEvaluationError("unresolved outcome must not carry eligibility or label")

    expected = {
        "forecast_count": len(rows),
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "eligible_resolved_count": eligible,
        "positive_count": positives,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SealedEvaluationError(f"outcome-label receipt {key} mismatch")
    return dict(receipt)


def _binary_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(prediction, dtype=bool)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred) or len(y) == 0:
        raise SealedEvaluationError("metric inputs must be aligned non-empty vectors")
    if not set(np.unique(y)).issubset({0, 1}):
        raise SealedEvaluationError("labels must be binary")
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else math.nan
    pofd = fp / (fp + tn) if fp + tn else math.nan
    tss = pod - pofd if math.isfinite(pod) and math.isfinite(pofd) else math.nan
    far = fp / (tp + fp) if tp + fp else math.nan
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "pod": pod, "far": far, "tss": tss}


def threshold_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise SealedEvaluationError("metric inputs must be aligned finite non-empty vectors")
    return _binary_metrics(y, p >= float(threshold))


def evaluate_sealed_cohort(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    label_receipt: Mapping[str, Any],
    policy: str = "MAX_TSS",
    review_fraction: float = 0.05,
    minimum_positive_support: int = 20,
) -> dict[str, Any]:
    """Score immutable forecasts without tuning any parameter on outcomes.

    Sample size sufficiency is reported separately from independence. This
    function cannot verify an external pre-outcome witness, custodian-controlled
    blindness, or provider-side source attestation, so it never emits an
    "independent evaluation established" claim by itself.
    """
    if policy not in POLICIES:
        raise SealedEvaluationError("unknown frozen policy")
    if not 0 < review_fraction <= 1:
        raise SealedEvaluationError("review_fraction must be in (0,1]")
    if not isinstance(minimum_positive_support, int) or minimum_positive_support < 1:
        raise SealedEvaluationError("minimum_positive_support must be positive")
    labels = validate_label_receipt(label_receipt)

    if isinstance(forecast_seals, (str, bytes)) or not isinstance(forecast_seals, Sequence) or not forecast_seals:
        raise SealedEvaluationError("forecast_seals must be a non-empty sequence")
    validated: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_issues: set[str] = set()
    package_hashes: set[str] = set()
    architecture_ids: set[str] = set()
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        seal_sha = str(seal["forecast_seal_sha256"])
        issue = str(seal["issued_at"])
        if seal_sha in seen_hashes:
            raise SealedEvaluationError("duplicate forecast seal in evaluation cohort")
        if issue in seen_issues:
            raise SealedEvaluationError("duplicate forecast issue time in evaluation cohort")
        seen_hashes.add(seal_sha)
        seen_issues.add(issue)
        package_hashes.add(str(seal["package_manifest_sha256"]))
        architecture_ids.add(str(seal["architecture_id"]))
        validated.append(seal)
    if len(package_hashes) != 1:
        raise SealedEvaluationError("model package changed within evaluation cohort")
    if len(architecture_ids) != 1:
        raise SealedEvaluationError("architecture changed within evaluation cohort")

    label_rows = labels["rows"]
    label_by_hash = {str(row["forecast_seal_sha256"]): row for row in label_rows}
    if set(label_by_hash) != seen_hashes:
        missing = sorted(seen_hashes - set(label_by_hash))
        extra = sorted(set(label_by_hash) - seen_hashes)
        raise SealedEvaluationError(
            f"outcome-label cohort does not exactly match forecasts; missing={missing[:3]}, extra={extra[:3]}"
        )

    state_probability = {state: [] for state in STATES}
    state_threshold: dict[str, float | None] = {state: None for state in STATES}
    state_permission: dict[str, str | None] = {state: None for state in STATES}
    y: list[int] = []
    unresolved_count = 0
    ineligible_count = 0
    for seal in validated:
        label = label_by_hash[str(seal["forecast_seal_sha256"])]
        if str(label["issued_at"]) != str(seal["issued_at"]):
            raise SealedEvaluationError("label issue time does not match forecast seal")
        if not label["outcome_resolved"]:
            unresolved_count += 1
            continue
        if label["eligible_new_crossing_issue"] is not True:
            ineligible_count += 1
            continue
        if label["label"] not in (0, 1):
            raise SealedEvaluationError("eligible resolved forecast is missing a binary label")
        y.append(int(label["label"]))
        for state in STATES:
            state_probability[state].append(float(seal["probabilities"][state]))
            threshold = float(seal["thresholds"][state][policy])
            if state_threshold[state] is None:
                state_threshold[state] = threshold
            elif not math.isclose(float(state_threshold[state]), threshold, rel_tol=0, abs_tol=1e-15):
                raise SealedEvaluationError("frozen threshold changed within sealed cohort")
            permission = str(seal["operator_permissions"][state])
            if state_permission[state] is None:
                state_permission[state] = permission
            elif state_permission[state] != permission:
                raise SealedEvaluationError("operator permission changed within sealed cohort")

    positives = int(sum(y))
    yy = np.asarray(y, dtype=int)
    results: dict[str, Any] = {}
    for state, values in state_probability.items():
        p = np.asarray(values, dtype=float)
        if len(p):
            threshold = float(state_threshold[state])
            numerical_crossing = p >= threshold
            numerical_metrics = _binary_metrics(yy, numerical_crossing)
            permission = str(state_permission[state])
            permitted = permission != "ABSTAIN"
            exposed_alert = numerical_crossing if permitted else np.zeros(len(p), dtype=bool)
            alert_metrics = _binary_metrics(yy, exposed_alert)
            brier = float(np.mean((p - yy.astype(float)) ** 2))
            n_review = max(1, int(math.ceil(len(p) * review_fraction)))
            actual_review_fraction = n_review / len(p)
            rank = np.argsort(-p, kind="mergesort")[:n_review]
            capture = int(np.sum(yy[rank]))
            capture_rate = capture / positives if positives else math.nan
            enrichment = (
                capture_rate / actual_review_fraction
                if positives and actual_review_fraction > 0
                else math.nan
            )
        else:
            permission = None
            permitted = False
            numerical_metrics = {}
            alert_metrics = {}
            brier = math.nan
            n_review = 0
            actual_review_fraction = math.nan
            capture = 0
            capture_rate = math.nan
            enrichment = math.nan
        results[state] = {
            "rows": len(p),
            "positives": positives,
            "numerical_threshold_metrics": numerical_metrics,
            "operator_permission": permission,
            "alerts_permitted_by_policy": permitted,
            "permission_filtered_alert_metrics": alert_metrics,
            "brier": brier,
            "requested_review_fraction": review_fraction,
            "actual_review_fraction": actual_review_fraction,
            "review_rows": n_review,
            "review_positive_capture": capture,
            "review_positive_capture_rate": capture_rate,
            "review_enrichment_vs_random": enrichment,
        }

    support_sufficient = positives >= minimum_positive_support
    claim_status = (
        "POSITIVE_SUPPORT_SUFFICIENT_BUT_INDEPENDENCE_UNVERIFIED"
        if support_sufficient
        else "INSUFFICIENT_POSITIVE_SUPPORT_DO_NOT_CLAIM_FINAL_SKILL"
    )
    return {
        "format": EVALUATION_FORMAT,
        "target": TARGET,
        "policy": policy,
        "forecast_count": len(validated),
        "resolved_eligible_rows": len(y),
        "unresolved_rows": unresolved_count,
        "resolved_ineligible_rows": ineligible_count,
        "positives": positives,
        "minimum_positive_support": minimum_positive_support,
        "positive_support_gate_passed": support_sufficient,
        "independence_verified": False,
        "trusted_pre_outcome_witness_verified": False,
        "provider_source_attestation_verified": False,
        "claim_status": claim_status,
        "package_manifest_sha256": next(iter(package_hashes)),
        "architecture_id": next(iter(architecture_ids)),
        "states": results,
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
        "sampled_crossing_equivalence_to_catalogue_definition_established": False,
    }
