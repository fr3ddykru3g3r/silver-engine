"""Sealed prospective evaluation primitives for IRIS-SEP.

A forecast is sealed before its 24-hour outcome is known. The seal binds the
frozen model package, exact feature row, trusted-source authentication receipt,
causal feature-derivation receipt, state probabilities and frozen thresholds.
Outcome derivation happens later from primary >10 MeV proton observations.

This module contains no training, calibration or threshold selection path.
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
TARGET = "NEW_GT10MEV_GE10PFU_CROSSING_WITHIN_24H"
HORIZON = timedelta(hours=24)


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
    # A forecast seal created hours later is not evidence of a forecast that
    # existed at issue time. Allow only a small serialization/IO interval.
    if (sealed - issue).total_seconds() > 300:
        raise SealedEvaluationError("forecast must be sealed within five minutes of issuance")
    if set(probabilities) != {"FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON"}:
        raise SealedEvaluationError("all four availability-state probabilities are required")
    if set(thresholds) != set(probabilities) or set(operator_permissions) != set(probabilities):
        raise SealedEvaluationError("threshold/permission state coverage mismatch")

    probability_payload = {state: _probability(value, f"probability {state}") for state, value in probabilities.items()}
    threshold_payload: dict[str, dict[str, float]] = {}
    for state, policies in thresholds.items():
        if set(policies) != {"MAX_TSS", "POD80_MIN_FAR"}:
            raise SealedEvaluationError("both frozen threshold policies required for every state")
        threshold_payload[state] = {
            policy: _probability(value, f"threshold {state}/{policy}")
            for policy, value in policies.items()
        }

    payload = {
        "format": FORMAT,
        "target": TARGET,
        "issued_at": issue.isoformat(),
        "sealed_at": sealed.isoformat(),
        "outcome_window_end": (issue + HORIZON).isoformat(),
        "architecture_id": str(architecture_id),
        "package_manifest_sha256": _sha(package_manifest_sha256, "package_manifest_sha256"),
        "feature_row_sha256": _sha(feature_row_sha256, "feature_row_sha256"),
        "source_authentication_sha256": _sha(source_authentication_sha256, "source_authentication_sha256"),
        "causal_feature_derivation_sha256": _sha(
            causal_feature_derivation_sha256,
            "causal_feature_derivation_sha256",
        ),
        "probabilities": probability_payload,
        "thresholds": threshold_payload,
        "operator_permissions": {state: str(value) for state, value in operator_permissions.items()},
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
    }
    payload["forecast_seal_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_forecast_seal(seal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or seal.get("format") != FORMAT or seal.get("target") != TARGET:
        raise SealedEvaluationError("unsupported forecast seal")
    claimed = seal.get("forecast_seal_sha256")
    unsigned = dict(seal)
    unsigned.pop("forecast_seal_sha256", None)
    if _sha(claimed, "forecast_seal_sha256") != sha256_bytes(_canonical_json(unsigned)):
        raise SealedEvaluationError("forecast seal digest mismatch")
    issue = _time(seal.get("issued_at"), "issued_at")
    sealed = _time(seal.get("sealed_at"), "sealed_at")
    if sealed < issue or (sealed - issue).total_seconds() > 300:
        raise SealedEvaluationError("forecast was not sealed at issue time")
    _sha(seal.get("package_manifest_sha256"), "package_manifest_sha256")
    _sha(seal.get("feature_row_sha256"), "feature_row_sha256")
    _sha(seal.get("source_authentication_sha256"), "source_authentication_sha256")
    _sha(seal.get("causal_feature_derivation_sha256"), "causal_feature_derivation_sha256")
    if seal.get("runtime_training_allowed") is not False or seal.get("runtime_recalibration_allowed") is not False or seal.get("runtime_rethresholding_allowed") is not False:
        raise SealedEvaluationError("forecast seal permits runtime model changes")
    rebuilt = build_forecast_seal(
        issued_at=issue, sealed_at=sealed,
        package_manifest_sha256=seal["package_manifest_sha256"],
        feature_row_sha256=seal["feature_row_sha256"],
        source_authentication_sha256=seal["source_authentication_sha256"],
        causal_feature_derivation_sha256=seal["causal_feature_derivation_sha256"],
        probabilities=seal.get("probabilities", {}), thresholds=seal.get("thresholds", {}),
        operator_permissions=seal.get("operator_permissions", {}), architecture_id=seal.get("architecture_id", ""),
    )
    if rebuilt != dict(seal):
        raise SealedEvaluationError("forecast seal semantic mismatch")
    return dict(seal)


def derive_new_crossing_labels(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    proton_times: Sequence[Any],
    proton_flux: Sequence[Any],
    threshold_pfu: float = 10.0,
) -> dict[str, Any]:
    """Derive NEW-crossing outcomes from later primary proton observations.

    Issue times already at/above threshold are ineligible. For an eligible issue,
    the label is one when the sampled >10 MeV flux crosses from below threshold
    to at/above threshold during ``(issue, issue+24h]``.
    """
    if isinstance(threshold_pfu, bool) or threshold_pfu != 10.0:
        raise SealedEvaluationError("frozen target requires threshold_pfu=10")
    times = pd.to_datetime(list(proton_times), utc=True, errors="coerce")
    flux = pd.to_numeric(pd.Series(list(proton_flux)), errors="coerce").to_numpy(dtype=float)
    if len(times) != len(flux) or len(times) < 2 or bool(pd.isna(times).any()):
        raise SealedEvaluationError("proton outcome series is invalid")
    order = np.argsort(times.asi8)
    times = pd.DatetimeIndex(times[order])
    flux = flux[order]
    if times.has_duplicates:
        raise SealedEvaluationError("duplicate proton outcome timestamps")
    # Do not drop missing/invalid samples: that hides gaps in the outcome record.
    valid_flux = np.isfinite(flux) & (flux >= 0)
    cadence = pd.Timedelta(minutes=5)
    rows = []
    seen = set()
    seen_issues = set()
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        key = seal["forecast_seal_sha256"]
        if key in seen:
            raise SealedEvaluationError("duplicate forecast seal")
        seen.add(key)
        issue = pd.Timestamp(_time(seal["issued_at"], "issued_at"))
        if issue in seen_issues:
            raise SealedEvaluationError("duplicate forecast issue")
        seen_issues.add(issue)
        end = issue + pd.Timedelta(hours=24)
        prior = np.flatnonzero(times <= issue)
        current = int(prior[-1]) if len(prior) else None
        current_flux = None
        eligible = None
        label = None
        crossing = None
        reason = "STALE_OR_MISSING_ISSUE_OBSERVATION"
        if current is not None and valid_flux[current] and issue - times[current] <= cadence:
            current_flux = float(flux[current])
            eligible = current_flux < threshold_pfu
            if not eligible:
                reason = "INELIGIBLE_ALREADY_ABOVE_THRESHOLD"
            else:
                future = np.flatnonzero((times > issue) & (times <= end))
                # Negative and positive labels use the same complete-horizon gate.
                # No inference across instrument gaps or an immature outcome window.
                idx = np.r_[current, future]
                complete = (len(future) > 0 and times[future[-1]] == end
                            and valid_flux[idx].all()
                            and (pd.Series(times[idx]).diff().dropna() <= cadence).all())
                reason = "INCOMPLETE_OR_INVALID_OUTCOME_WINDOW"
                if complete:
                    above = future[flux[future] >= threshold_pfu]
                    label = int(len(above) > 0)
                    crossing = times[above[0]].isoformat() if len(above) else None
                    reason = "COMPLETE_SAMPLED_24H_OUTCOME"
        rows.append({
            "forecast_seal_sha256": key,
            "issued_at": issue.isoformat(),
            "eligible_new_crossing_issue": eligible,
            "label": label,
            "current_flux_pfu": current_flux,
            "first_crossing_utc": crossing,
            "outcome_status": reason,
        })

    payload = {
        "format": LABEL_FORMAT,
        "target": TARGET,
        "threshold_pfu": float(threshold_pfu),
        "forecast_count": len(rows),
        "eligible_count": sum(row["eligible_new_crossing_issue"] is True for row in rows),
        "unresolved_count": sum(row["label"] is None and row["eligible_new_crossing_issue"] is not False for row in rows),
        "maximum_sample_gap_seconds": 300,
        "endpoint_rule": "OBSERVATION_REQUIRED_AT_HORIZON_END",
        "label_semantics": "SAMPLED_CROSSING_NOT_YET_VALIDATED_AGAINST_CLEAR_EVENT_CATALOGUE",
        "positive_count": sum(int(row["label"] == 1) for row in rows),
        "rows": rows,
    }
    payload["label_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_label_receipt(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check internal integrity; a hash is not independent source authentication."""
    if not isinstance(receipt, Mapping) or receipt.get("format") != LABEL_FORMAT or receipt.get("target") != TARGET:
        raise SealedEvaluationError("unsupported label receipt")
    unsigned = dict(receipt)
    claimed = unsigned.pop("label_receipt_sha256", None)
    if claimed != sha256_bytes(_canonical_json(unsigned)):
        raise SealedEvaluationError("label receipt digest mismatch")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise SealedEvaluationError("label receipt rows missing")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SealedEvaluationError("invalid label row")
        key = _sha(row.get("forecast_seal_sha256"), "forecast_seal_sha256")
        if key in seen:
            raise SealedEvaluationError("duplicate label row")
        seen.add(key)
        label = row.get("label")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise SealedEvaluationError("invalid binary label")
        if label is not None and (row.get("eligible_new_crossing_issue") is not True or row.get("outcome_status") != "COMPLETE_SAMPLED_24H_OUTCOME"):
            raise SealedEvaluationError("label lacks complete eligible outcome")
    return rows


def threshold_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise SealedEvaluationError("metric inputs must be aligned non-empty vectors")
    if not set(np.unique(y)).issubset({0, 1}):
        raise SealedEvaluationError("labels must be binary")
    pred = p >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else math.nan
    pofd = fp / (fp + tn) if fp + tn else math.nan
    tss = pod - pofd if math.isfinite(pod) and math.isfinite(pofd) else math.nan
    far = fp / (tp + fp) if tp + fp else math.nan
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "pod": pod, "far": far, "tss": tss}


def evaluate_sealed_cohort(
    *,
    forecast_seals: Sequence[Mapping[str, Any]],
    label_receipt: Mapping[str, Any],
    policy: str = "MAX_TSS",
    review_fraction: float = 0.05,
    minimum_positive_support: int = 20,
) -> dict[str, Any]:
    """Score immutable forecasts without tuning any parameter on outcomes."""
    if policy not in {"MAX_TSS", "POD80_MIN_FAR"}:
        raise SealedEvaluationError("unknown frozen policy")
    if not 0 < review_fraction <= 1:
        raise SealedEvaluationError("review_fraction must be in (0,1]")
    if not isinstance(minimum_positive_support, int) or minimum_positive_support < 1:
        raise SealedEvaluationError("minimum_positive_support must be positive")

    labels = validate_label_receipt(label_receipt)
    label_by_hash = {row.get("forecast_seal_sha256"): row for row in labels if isinstance(row, Mapping)}
    state_probability = {state: [] for state in ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")}
    state_threshold = {state: None for state in state_probability}
    y = []
    seen = set()
    package = None
    seen_issues = set()
    unresolved = 0
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        key = seal["forecast_seal_sha256"]
        if key in seen:
            raise SealedEvaluationError("duplicate forecast seal")
        seen.add(key)
        issue = _time(seal["issued_at"], "issued_at")
        if issue in seen_issues:
            raise SealedEvaluationError("duplicate forecast issue")
        seen_issues.add(issue)
        identity = (seal["package_manifest_sha256"], seal["architecture_id"])
        if package is not None and package != identity:
            raise SealedEvaluationError("model package changed within sealed cohort")
        package = identity
        label = label_by_hash.get(key)
        if label is None:
            raise SealedEvaluationError("forecast missing from label receipt")
        if label.get("eligible_new_crossing_issue") is False:
            continue
        if label.get("label") is None:
            unresolved += 1
            continue
        if label.get("label") not in (0, 1):
            raise SealedEvaluationError("eligible forecast is missing a binary label")
        y.append(int(label["label"]))
        for state in state_probability:
            state_probability[state].append(float(seal["probabilities"][state]))
            threshold = float(seal["thresholds"][state][policy])
            if state_threshold[state] is None:
                state_threshold[state] = threshold
            elif not math.isclose(float(state_threshold[state]), threshold, rel_tol=0, abs_tol=1e-15):
                raise SealedEvaluationError("frozen threshold changed within sealed cohort")

    positives = int(sum(y))
    results: dict[str, Any] = {}
    for state, values in state_probability.items():
        metrics = threshold_metrics(y, values, float(state_threshold[state])) if y else {}
        p = np.asarray(values, dtype=float)
        yy = np.asarray(y, dtype=float)
        brier = float(np.mean((p - yy) ** 2)) if len(p) else math.nan
        n_review = max(1, int(math.ceil(len(p) * review_fraction))) if len(p) else 0
        if n_review:
            rank = np.argsort(-p, kind="mergesort")[:n_review]
            capture = int(np.sum(yy[rank]))
            capture_rate = capture / positives if positives else math.nan
            random_expectation = n_review / len(p)
            enrichment = capture_rate / random_expectation if positives and random_expectation else math.nan
        else:
            capture = 0
            capture_rate = math.nan
            enrichment = math.nan
        results[state] = {
            "rows": len(p),
            "positives": positives,
            "threshold_metrics": metrics,
            "brier": brier,
            "review_fraction": review_fraction,
            "review_rows": n_review,
            "review_positive_capture": capture,
            "review_positive_capture_rate": capture_rate,
            "review_enrichment_vs_random": enrichment,
        }

    enough = positives >= minimum_positive_support
    return {
        "format": "IRIS_SEP_SEALED_COHORT_EVALUATION_V1",
        "target": TARGET,
        "policy": policy,
        "eligible_rows": len(y),
        "positives": positives,
        "minimum_positive_support": minimum_positive_support,
        "support_gate_passed": enough,
        "claim_status": "NUMERICAL_SUPPORT_ONLY_INDEPENDENCE_UNVERIFIED" if enough else "INSUFFICIENT_POSITIVE_SUPPORT_DO_NOT_CLAIM_FINAL_SKILL",
        "independent_evaluation_verified": False,
        "unresolved_outcome_rows": unresolved,
        "review_metric_scope": "RETROSPECTIVE_RANKING_NOT_A_DEPLOYABLE_DAILY_BUDGET_POLICY",
        "threshold_metric_scope": "COUNTERFACTUAL_PROBABILITY_SCORES_NOT_PERMISSION_FILTERED_ALERTS",
        "seal_limitation": "Self-reported timestamps and digests require externally witnessed pre-outcome publication; they do not prove independence.",
        "states": results,
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
    }
