from __future__ import annotations

from datetime import datetime, timezone
import copy

import pytest

from iris_report.iris_sep.tools.capture_prospective_input_snapshot_v1 import (
    SnapshotError,
    energy_key,
    select_proton_input,
)
from iris_report.iris_sep.tools.build_past_proton_proxy_prediction_v1 import (
    feature_schema,
    record_hash,
    verify_existing_chain,
)
from iris_report.iris_sep.tools.verify_prospective_prediction_ledger_v1 import (
    LedgerError,
    digest,
    verify,
)


def minimal_source_config(max_age: int = 15) -> dict:
    return {
        "sources": {
            "primary_integral_protons": {
                "source_id": "NOAA_SWPC_PRIMARY_PROTON_7D",
                "units": "pfu",
                "target_energy_labels": [">=10 MeV", ">10 MeV"],
            }
        },
        "past_proton_input_gate": {"maximum_observation_age_minutes_at_issue": max_age},
    }


def test_energy_normalization_is_bounded_not_fuzzy() -> None:
    assert energy_key("  >=10   MeV ") == ">=10 MeV"
    assert energy_key("≥10 MeV") == ">=10 MeV"
    assert energy_key("30 MeV") == "30 MeV"


def test_latest_preissue_proton_value_is_selected_and_future_is_ignored() -> None:
    issue = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)
    payload = [
        {"time_tag": "2026-09-11T23:50:00Z", "satellite": 19, "flux": 1.2, "energy": ">=10 MeV"},
        {"time_tag": "2026-09-11T23:55:00Z", "satellite": 19, "flux": 2.3, "energy": ">=10 MeV"},
        {"time_tag": "2026-09-12T00:05:00Z", "satellite": 19, "flux": 99.0, "energy": ">=10 MeV"},
        {"time_tag": "2026-09-11T23:59:00Z", "satellite": 19, "flux": 50.0, "energy": ">=30 MeV"},
    ]
    selected = select_proton_input(payload, issue, minimal_source_config())
    assert selected["observation_time_utc"] == "2026-09-11T23:55:00Z"
    assert selected["flux_pfu"] == 2.3
    assert selected["age_minutes_at_issue"] == 5.0
    assert selected["age_gate_passed"] is True


def test_stale_value_fails_age_gate_without_substitution() -> None:
    issue = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)
    payload = [{"time_tag": "2026-09-11T23:30:00Z", "satellite": 19, "flux": 2.0, "energy": ">=10 MeV"}]
    selected = select_proton_input(payload, issue, minimal_source_config(max_age=15))
    assert selected["age_gate_passed"] is False


def test_ambiguous_latest_proton_measurement_is_rejected() -> None:
    issue = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)
    payload = [
        {"time_tag": "2026-09-11T23:55:00Z", "satellite": 18, "flux": 1.0, "energy": ">=10 MeV"},
        {"time_tag": "2026-09-11T23:55:00Z", "satellite": 19, "flux": 2.0, "energy": ">=10 MeV"},
    ]
    with pytest.raises(SnapshotError):
        select_proton_input(payload, issue, minimal_source_config())


def test_feature_schema_is_deterministic_and_minimal() -> None:
    rule = {
        "model_id": "past_proton_active_proxy",
        "input": {
            "selection": "latest preissue value",
            "maximum_observation_age_minutes_at_issue": 15,
        },
        "decision": {"missing_stale_or_invalid_input": "ABSTAIN"},
    }
    source = minimal_source_config()
    a = feature_schema(rule, source)
    b = feature_schema(copy.deepcopy(rule), copy.deepcopy(source))
    assert a == b
    assert len(a["features"]) == 1
    assert a["features"][0]["source_id"] == "NOAA_SWPC_PRIMARY_PROTON_7D"


def make_record(issue: str, pred: str, observed: str, previous: str | None) -> dict:
    body = {
        "format": "IRIS_SEP_PROSPECTIVE_PREDICTION_RECORD_V1",
        "study_id": "IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1",
        "issue_time_utc": issue,
        "prediction_timestamp_utc": pred,
        "model_id": "past_proton_active_proxy",
        "probability": 0.0,
        "alert": 0,
        "abstain": False,
        "selected_input": {
            "observation_time_utc": observed,
            "satellite": "19",
            "energy": ">=10 MeV",
            "flux_pfu": 1.0,
            "age_minutes_at_issue": 5.0,
        },
        "rule_sha256": "1" * 64,
        "feature_schema_sha256": "2" * 64,
        "source_contract_sha256": "3" * 64,
        "snapshot_receipt_sha256": "4" * 64,
        "previous_record_sha256": previous,
        "protected_outcomes_accessed": False,
    }
    row = dict(body)
    row["record_sha256"] = digest(body)
    return row


def test_prediction_chain_verifies_and_tamper_fails() -> None:
    first = make_record(
        "2026-09-12T00:00:00Z", "2026-09-11T23:58:00Z", "2026-09-11T23:55:00Z", None
    )
    second = make_record(
        "2026-09-13T00:00:00Z", "2026-09-12T23:58:00Z", "2026-09-12T23:55:00Z", first["record_sha256"]
    )
    result = verify([first, second])
    assert result["status"] == "PASS"
    assert result["records"] == 2
    tampered = copy.deepcopy(second)
    tampered["selected_input"]["flux_pfu"] = 99.0
    with pytest.raises(LedgerError):
        verify([first, tampered])


def test_builder_chain_verifier_rejects_duplicate_issue() -> None:
    first = make_record(
        "2026-09-12T00:00:00Z", "2026-09-11T23:58:00Z", "2026-09-11T23:55:00Z", None
    )
    duplicate = copy.deepcopy(first)
    duplicate["previous_record_sha256"] = first["record_sha256"]
    duplicate.pop("record_sha256")
    duplicate["record_sha256"] = record_hash({k: v for k, v in duplicate.items() if k != "record_sha256"})
    with pytest.raises(Exception):
        verify_existing_chain([first, duplicate])
