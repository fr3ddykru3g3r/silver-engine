from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iris_sep.contracts import (  # noqa: E402
    ContractViolation,
    assert_role_access,
    assert_unique_issue_ids,
    build_freeze_receipt,
    load_contract,
    parse_utc,
)


def test_contract_is_frozen_and_hashed() -> None:
    contract = load_contract(ROOT / "config" / "benchmark_contract.json")
    assert contract.sha256
    assert contract.roles[-1] == "locked_test"
    assert contract.allowed_operations("validation_threshold") == ("operating_threshold",)


def test_locked_test_is_fail_closed_during_tuning() -> None:
    with pytest.raises(ContractViolation, match="inaccessible"):
        assert_role_access("locked_test", phase="tuning", outcome_requested=False)


def test_locked_test_is_available_only_for_final_evaluation() -> None:
    assert_role_access("locked_test", phase="final_evaluation", outcome_requested=True)


def test_duplicate_issue_ids_are_rejected() -> None:
    with pytest.raises(ContractViolation, match="duplicate"):
        assert_unique_issue_ids([{"issue_id": "x"}, {"issue_id": "x"}])


def test_timestamps_must_be_timezone_aware() -> None:
    assert parse_utc("2026-09-04T12:00:00Z").isoformat().endswith("+00:00")
    with pytest.raises(ContractViolation, match="timezone-aware"):
        parse_utc("2026-09-04T12:00:00")


def test_freeze_receipt_never_claims_test_access() -> None:
    contract = load_contract(ROOT / "config" / "benchmark_contract.json")
    receipt = build_freeze_receipt(contract)
    assert receipt["status"] == "PASS"
    assert receipt["test_outcomes_accessed"] is False
    json.dumps(receipt)
