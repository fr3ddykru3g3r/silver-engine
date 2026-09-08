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
TARGET_THRESHOLD_PFU = 10.0
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
    """Parse and order primary proton samples without hiding bad measurements.

    Invalid flux values are intentionally retained so that only forecast windows
    touching them become unresolved. Dropping them would hide gaps; rejecting the
    whole series would let an unrelated bad sample poison every forecast in a
    batch.
    """
    times = pd.to_datetime(list(proton_times), utc=True, errors="coerce")
    flux = pd.to_numeric(pd.Series(list(proton_flux)), errors="coerce").to_numpy(dtype=float)
    if len(times) != len(flux) or len(times) < 2 or bool(pd.isna(times).any()):
        raise SealedEvaluationError("proton outcome series is invalid")

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
    issue_flux = float(flux[issue_idx])
    if not math.isfinite(issue_flux) or issue_flux < 0:
        return {
            "resolved": False,
            "reason": "ISSUE_SUPPORT_INVALID_FLUX",
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

    covered_flux = flux[issue_idx : end_idx + 1]
    if not np.isfinite(covered_flux).all() or np.any(covered_flux < 0):
        return {
            "resolved": False,
            "reason": "OUTCOME_INVALID_OR_NONFINITE_FLUX",
            "issue_support_lag_seconds": float(issue_lag.total_seconds()),
        }

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
    threshold_pfu: float = TARGET_THRESHOLD_PFU,
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
    if isinstance(threshold_pfu, bool) or not isinstance(threshold_pfu, (int, float)):
        raise SealedEvaluationError("frozen target requires threshold_pfu=10")
    if not math.isfinite(float(threshold_pfu)) or float(threshold_pfu) != TARGET_THRESHOLD_PFU:
        raise SealedEvaluationError("frozen target requires threshold_pfu=10")
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
        eligible = current_flux < TARGET_THRESHOLD_PFU
        row["current_flux_pfu"] = current_flux
        row["eligible_new_crossing_issue"] = bool(eligible)
        if not eligible:
            rows.append(row)
            continue

        label = 0
        previous = current_flux
        for idx in range(issue_idx + 1, end_idx + 1):
            value = float(flux[idx])
            if previous < TARGET_THRESHOLD_PFU <= value:
                label = 1
                row["first_crossing_utc"] = times[idx].isoformat()
                break
            previous = value
        row["label"] = label
        rows.append(row)

    payload = {
        "format": LABEL_FORMAT,
        "target": TARGET,
        "threshold_pfu": TARGET_THRESHOLD_PFU,
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
        "sampled_crossing_semantics": "PREVIOUS_LT10_AND_CURRENT_GE10_ON_PRIMARY_SERIES",
        "official_event_catalogue_equivalence_established": False,
        "rows": rows,
    }
    payload["label_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_label_receipt(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate V2 sampled-outcome receipt integrity and frozen target semantics."""
    if not isinstance(receipt, Mapping) or receipt.get("format") != LABEL_FORMAT:
        raise SealedEvaluationError("unsupported label receipt; V2 required")
    if receipt.get("target") != TARGET:
        raise SealedEvaluationError("label receipt target mismatch")
    if receipt.get("threshold_pfu") != TARGET_THRESHOLD_PFU:
        raise SealedEvaluationError("label receipt threshold does not match frozen 10 pfu target")
    if receipt.get("official_event_catalogue_equivalence_established") is not False:
        raise SealedEvaluationError("label receipt overstates official catalogue equivalence")
    unsigned = dict(receipt)
    claimed = unsigned.pop("label_receipt_sha256", None)
    if _sha(claimed, "label_receipt_sha256") != sha256_bytes(_canonical_json(unsigned)):
        raise SealedEvaluationError("label receipt digest mismatch")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise SealedEvaluationError("label receipt rows missing")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SealedEvaluationError("label row invalid")
        key = _sha(row.get("forecast_seal_sha256"), "forecast_seal_sha256")
        if key in seen:
            raise SealedEvaluationError("duplicate label row")
        seen.add(key)
        resolved = row.get("outcome_resolved")
        if not isinstance(resolved, bool):
            raise SealedEvaluationError("label row outcome_resolved must be boolean")
        eligible = row.get("eligible_new_crossing_issue")
        label = row.get("label")
        if not resolved:
            if label is not None or eligible is not None:
                raise SealedEvaluationError("unresolved label row cannot contain an outcome")
        elif eligible is False:
            if label is not None:
                raise SealedEvaluationError("ineligible already-active issue cannot have a label")
        elif eligible is True:
            if label not in (0, 1) or isinstance(label, bool):
                raise SealedEvaluationError("eligible resolved label row must have binary label")
        else:
            raise SealedEvaluationError("resolved label row requires eligibility")
    if receipt.get("forecast_count") != len(rows):
        raise SealedEvaluationError("label receipt forecast_count mismatch")
    if receipt.get("resolved_count") != sum(int(row["outcome_resolved"]) for row in rows):
        raise SealedEvaluationError("label receipt resolved_count mismatch")
    if receipt.get("unresolved_count") != sum(int(not row["outcome_resolved"]) for row in rows):
        raise SealedEvaluationError("label receipt unresolved_count mismatch")
    if receipt.get("positive_count") != sum(int(row.get("label") == 1) for row in rows):
        raise SealedEvaluationError("label receipt positive_count mismatch")
    return [dict(row) for row in rows]


def threshold_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = p >= float(threshold)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    # Undefined rates are represented as JSON null, never NaN and never fake zero.
    pod = tp / (tp + fn) if tp + fn else None
    far = fp / (tp + fp) if tp + fp else 0.0
    tss = (tp / (tp + fn) if tp + fn else 0.0) - (fp / (fp + tn) if fp + tn else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "pod": pod, "far": far, "tss": tss}


def brier(y_true: Sequence[int], probability: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    return float(np.mean((p - y) ** 2))


def evaluate_sealed_cohort(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    label_receipt: Mapping[str, Any],
    review_fraction: float = 0.05,
    minimum_positive_support: int = 20,
) -> dict[str, Any]:
    """Evaluate only resolved, eligible rows under frozen policies.

    Passing the positive-support gate is necessary but not sufficient for an
    independent-evaluation claim. ``independence_verified`` deliberately remains
    false until a separate trusted witness/custodian mechanism is validated.
    """
    if not forecast_seals:
        raise SealedEvaluationError("no forecast seals supplied")
    if not 0 < float(review_fraction) <= 1:
        raise SealedEvaluationError("review_fraction must be in (0,1]")
    if not isinstance(minimum_positive_support, int) or minimum_positive_support < 1:
        raise SealedEvaluationError("minimum_positive_support must be positive")

    labels = validate_label_receipt(label_receipt)
    label_by_hash = {row["forecast_seal_sha256"]: row for row in labels}
    forecast_hashes: set[str] = set()
    forecast_issues: set[str] = set()
    package_identity: tuple[str, str] | None = None
    frozen_thresholds: dict[str, dict[str, float]] | None = None
    frozen_permissions: dict[str, str] | None = None
    state_probability = {state: [] for state in STATES}
    y: list[int] = []
    unresolved = 0
    ineligible = 0

    validated_forecasts = []
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        seal_hash = str(seal["forecast_seal_sha256"])
        issue = str(seal["issued_at"])
        if seal_hash in forecast_hashes:
            raise SealedEvaluationError("duplicate forecast seal within cohort")
        if issue in forecast_issues:
            raise SealedEvaluationError("duplicate forecast issue time within cohort")
        forecast_hashes.add(seal_hash)
        forecast_issues.add(issue)
        validated_forecasts.append(seal)

        identity = (str(seal["package_manifest_sha256"]), str(seal["architecture_id"]))
        if package_identity is None:
            package_identity = identity
        elif identity != package_identity:
            raise SealedEvaluationError("model package changed within sealed cohort")
        thresholds = seal["thresholds"]
        permissions = seal["operator_permissions"]
        if frozen_thresholds is None:
            frozen_thresholds = thresholds
            frozen_permissions = permissions
        elif thresholds != frozen_thresholds:
            raise SealedEvaluationError("frozen threshold changed within sealed cohort")
        elif permissions != frozen_permissions:
            raise SealedEvaluationError("operator permission changed within sealed cohort")

    if set(label_by_hash) != forecast_hashes:
        raise SealedEvaluationError("forecast seals and label receipt rows must exactly match")

    eligible_forecasts = []
    for seal in validated_forecasts:
        label = label_by_hash[seal["forecast_seal_sha256"]]
        if not label["outcome_resolved"]:
            unresolved += 1
            continue
        if label["eligible_new_crossing_issue"] is False:
            ineligible += 1
            continue
        y.append(int(label["label"]))
        eligible_forecasts.append(seal)
        for state in STATES:
            state_probability[state].append(float(seal["probabilities"][state]))

    if not eligible_forecasts:
        raise SealedEvaluationError("no resolved eligible forecasts to evaluate")

    results = {}
    yy = np.asarray(y, dtype=int)
    positives = int(np.sum(yy))
    for state in STATES:
        p = np.asarray(state_probability[state], dtype=float)
        by_policy = {
            policy: threshold_metrics(yy, p, frozen_thresholds[state][policy])
            for policy in POLICIES
        }
        permission = frozen_permissions[state]
        alerts_permitted = permission != "ABSTAIN"
        alert_by_policy = {}
        for policy in POLICIES:
            numerical = by_policy[policy]
            if alerts_permitted:
                alert_by_policy[policy] = dict(numerical)
            else:
                alert_by_policy[policy] = threshold_metrics(yy, np.zeros_like(p), 1.0)

        n_review = max(1, min(len(p), int(math.ceil(review_fraction * len(p)))))
        rank = np.argsort(-p, kind="mergesort")[:n_review]
        capture = int(np.sum(yy[rank]))
        actual_review_fraction = n_review / len(p)
        capture_rate = capture / positives if positives else None
        enrichment = (
            capture_rate / actual_review_fraction
            if capture_rate is not None and actual_review_fraction
            else None
        )
        results[state] = {
            "rows": int(len(p)),
            "positives": positives,
            "probability_brier": brier(yy, p),
            "numerical_threshold_metrics": by_policy,
            "permission_filtered_alert_metrics": alert_by_policy,
            "operator_permission": permission,
            "alerts_permitted_by_policy": alerts_permitted,
            "review_rows": n_review,
            "actual_review_fraction": actual_review_fraction,
            "review_positive_capture": capture,
            "review_capture_rate": capture_rate,
            "review_enrichment_vs_random": enrichment,
        }

    support_ok = positives >= minimum_positive_support
    payload = {
        "format": EVALUATION_FORMAT,
        "target": TARGET,
        "sealed_forecasts": len(validated_forecasts),
        "eligible_rows": len(eligible_forecasts),
        "unresolved_outcome_rows": unresolved,
        "ineligible_already_active_rows": ineligible,
        "positives": positives,
        "minimum_positive_support": minimum_positive_support,
        "positive_support_gate_passed": support_ok,
        "independence_verified": False,
        "trusted_pre_outcome_witness_verified": False,
        "provider_source_attestation_verified": False,
        "claim_status": (
            "POSITIVE_SUPPORT_SUFFICIENT_BUT_INDEPENDENCE_UNVERIFIED"
            if support_ok
            else "INSUFFICIENT_POSITIVE_SUPPORT_DO_NOT_CLAIM_FINAL_SKILL"
        ),
        "review_metric_scope": "RETROSPECTIVE_RANKING_DIAGNOSTIC_NOT_A_DEPLOYED_FIXED_DAILY_BUDGET_POLICY",
        "states": results,
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
    }
    payload["evaluation_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload
