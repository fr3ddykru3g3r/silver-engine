from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from iris_report.iris_sep.src.iris_sep.sealed_evaluation import (
    SealedEvaluationError,
    build_forecast_seal,
    derive_new_crossing_labels,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "event_label_semantics_contract_2026-09-09.json"
ISSUE = datetime(2026, 9, 9, 0, 0, tzinfo=timezone.utc)
STATES = ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")


def _seal(issue=ISSUE):
    return build_forecast_seal(
        issued_at=issue,
        sealed_at=issue + timedelta(seconds=20),
        package_manifest_sha256="a" * 64,
        feature_row_sha256="b" * 64,
        source_authentication_sha256="c" * 64,
        causal_feature_derivation_sha256="d" * 64,
        probabilities={state: 0.1 for state in STATES},
        thresholds={state: {"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4} for state in STATES},
        operator_permissions={
            "FULL": "NORMAL_ONLY_IF_ADMISSION_PASSES",
            "NO_XRS": "DEGRADED",
            "NO_PROTON": "DEGRADED",
            "NO_XRS_OR_PROTON": "ABSTAIN",
        },
        architecture_id="event-semantics-fixture",
    )


def _series(default=1.0, *, start=None, end=None):
    times = []
    flux = []
    current = start or ISSUE - timedelta(minutes=5)
    end = end or ISSUE + timedelta(hours=24)
    while current <= end:
        times.append(current)
        flux.append(float(default))
        current += timedelta(minutes=5)
    return times, flux


def test_contract_preserves_catalogue_equivalence_boundary():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert data["format"] == "IRIS_SEP_EVENT_LABEL_SEMANTICS_CONTRACT_V1"
    boundary = data["scientific_boundary"]
    assert boundary["sampled_primary_series_crossing_definition_verified_in_code"] is True
    assert boundary["official_event_catalogue_equivalence_established"] is False


def test_target_threshold_cannot_change_under_same_target_identifier():
    times, flux = _series()
    with pytest.raises(SealedEvaluationError, match="frozen target requires threshold_pfu=10"):
        derive_new_crossing_labels(
            forecast_seals=[_seal()],
            proton_times=times,
            proton_flux=flux,
            threshold_pfu=20.0,
        )


def test_exactly_ten_from_below_counts_as_positive_crossing():
    times, flux = _series()
    cross = ISSUE + timedelta(hours=1)
    idx = times.index(cross)
    flux[idx] = 10.0
    for j in range(idx + 1, len(flux)):
        flux[j] = 10.0
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["eligible_new_crossing_issue"] is True
    assert row["label"] == 1
    assert row["first_crossing_utc"] == cross.isoformat()


def test_crossing_at_exact_24h_endpoint_counts():
    times, flux = _series()
    endpoint = ISSUE + timedelta(hours=24)
    flux[times.index(endpoint)] = 10.0
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["label"] == 1
    assert row["first_crossing_utc"] == endpoint.isoformat()


def test_issue_support_exactly_at_threshold_is_ineligible_not_negative():
    times, flux = _series()
    issue_idx = times.index(ISSUE)
    flux[issue_idx] = 10.0
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["outcome_resolved"] is True
    assert row["eligible_new_crossing_issue"] is False
    assert row["label"] is None


def test_crossing_after_24h_is_not_counted():
    times, flux = _series()
    times.append(ISSUE + timedelta(hours=24, minutes=5))
    flux.append(12.0)
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["label"] == 0


def test_issue_support_older_than_five_minutes_is_unresolved():
    times, flux = _series(start=ISSUE - timedelta(minutes=10))
    for timestamp in (ISSUE, ISSUE - timedelta(minutes=5)):
        idx = times.index(timestamp)
        del times[idx]
        del flux[idx]
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["outcome_resolved"] is False
    assert row["resolution_reason"] == "ISSUE_SUPPORT_STALE"
    assert row["label"] is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_flux_inside_one_window_is_unresolved_not_a_batch_crash(invalid):
    second_issue = ISSUE + timedelta(days=2)
    times, flux = _series(
        start=ISSUE - timedelta(minutes=5),
        end=second_issue + timedelta(hours=24),
    )
    bad_time = ISSUE + timedelta(hours=6)
    flux[times.index(bad_time)] = invalid
    result = derive_new_crossing_labels(
        forecast_seals=[_seal(ISSUE), _seal(second_issue)],
        proton_times=times,
        proton_flux=flux,
    )
    first, second = result["rows"]
    assert first["outcome_resolved"] is False
    assert first["resolution_reason"] == "OUTCOME_INVALID_OR_NONFINITE_FLUX"
    assert first["label"] is None
    assert second["outcome_resolved"] is True
    assert second["label"] == 0


def test_invalid_issue_support_is_unresolved():
    times, flux = _series()
    flux[times.index(ISSUE)] = float("nan")
    result = derive_new_crossing_labels(
        forecast_seals=[_seal()], proton_times=times, proton_flux=flux
    )
    row = result["rows"][0]
    assert row["outcome_resolved"] is False
    assert row["resolution_reason"] == "ISSUE_SUPPORT_INVALID_FLUX"
    assert row["label"] is None
