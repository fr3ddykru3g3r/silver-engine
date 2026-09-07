"""Sealed prospective evaluation primitives for IRIS-SEP.

A forecast is sealed before its 24-hour outcome is known. The seal binds the
frozen model package, exact feature row, trusted-source authentication receipt,
state probabilities and frozen thresholds. Outcome derivation happens later
from primary >10 MeV proton observations.

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


FORMAT = "IRIS_SEP_SEALED_FORECAST_V1"
LABEL_FORMAT = "IRIS_SEP_SEALED_NEW_CROSSING_LABELS_V1"
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
    if seal.get("runtime_training_allowed") is not False or seal.get("runtime_recalibration_allowed") is not False or seal.get("runtime_rethresholding_allowed") is not False:
        raise SealedEvaluationError("forecast seal permits runtime model changes")
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
    if not math.isfinite(float(threshold_pfu)) or threshold_pfu <= 0:
        raise SealedEvaluationError("threshold_pfu must be positive")
    times = pd.to_datetime(list(proton_times), utc=True, errors="coerce")
    flux = pd.to_numeric(pd.Series(list(proton_flux)), errors="coerce").to_numpy(dtype=float)
    if len(times) != len(flux) or len(times) < 2 or bool(pd.isna(times).any()):
        raise SealedEvaluationError("proton outcome series is invalid")
    order = np.argsort(times.asi8)
    times = pd.DatetimeIndex(times[order])
    flux = flux[order]
    finite = np.isfinite(flux)
    if not finite.all():
        times = times[finite]
        flux = flux[finite]
    if len(times) < 2:
        raise SealedEvaluationError("insufficient finite proton outcome samples")

    rows = []
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        issue = pd.Timestamp(_time(seal["issued_at"], "issued_at"))
        end = issue + pd.Timedelta(hours=24)
        prior_idx = np.where(times <= issue)[0]
        if len(prior_idx) == 0:
            raise SealedEvaluationError("no proton observation at/before one forecast issue")
        current_index = int(prior_idx[-1])
        current_flux = float(flux[current_index])
        eligible = current_flux < threshold_pfu
        label = None
        first_crossing = None
        if eligible:
            # Include the current sample as the predecessor and search future
            # samples through the closed 24-hour horizon.
            future_idx = np.where((times > issue) & (times <= end))[0]
            previous = current_flux
            label = 0
            for idx in future_idx:
                value = float(flux[idx])
                if previous < threshold_pfu <= value:
                    label = 1
                    first_crossing = times[idx].isoformat()
                    break
                previous = value
        rows.append({
            "forecast_seal_sha256": seal["forecast_seal_sha256"],
            "issued_at": issue.isoformat(),
            "eligible_new_crossing_issue": bool(eligible),
            "label": label,
            "current_flux_pfu": current_flux,
            "first_crossing_utc": first_crossing,
        })

    payload = {
        "format": LABEL_FORMAT,
        "target": TARGET,
        "threshold_pfu": float(threshold_pfu),
        "forecast_count": len(rows),
        "eligible_count": sum(int(row["eligible_new_crossing_issue"]) for row in rows),
        "positive_count": sum(int(row["label"] == 1) for row in rows),
        "rows": rows,
    }
    payload["label_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


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

    labels = label_receipt.get("rows") if isinstance(label_receipt, Mapping) else None
    if not isinstance(labels, list):
        raise SealedEvaluationError("label receipt rows missing")
    label_by_hash = {row.get("forecast_seal_sha256"): row for row in labels if isinstance(row, Mapping)}
    state_probability = {state: [] for state in ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")}
    state_threshold = {state: None for state in state_probability}
    y = []
    for raw in forecast_seals:
        seal = validate_forecast_seal(raw)
        label = label_by_hash.get(seal["forecast_seal_sha256"])
        if label is None or not label.get("eligible_new_crossing_issue"):
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
            random_expectation = review_fraction
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
        "claim_status": "INDEPENDENT_EVALUATION_SUPPORT_SUFFICIENT" if enough else "INSUFFICIENT_POSITIVE_SUPPORT_DO_NOT_CLAIM_FINAL_SKILL",
        "states": results,
        "runtime_training_allowed": False,
        "runtime_recalibration_allowed": False,
        "runtime_rethresholding_allowed": False,
    }
