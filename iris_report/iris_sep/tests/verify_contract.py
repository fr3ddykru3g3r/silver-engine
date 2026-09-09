"""Dependency-free verification for the frozen benchmark contract."""

from pathlib import Path
import sys


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


def require_violation(callable_, message: str) -> None:
    try:
        callable_()
    except ContractViolation:
        return
    raise AssertionError(message)


contract = load_contract(ROOT / "config" / "benchmark_contract_v2.json")
assert contract.raw["forecast"]["issue_cadence_hours"] == 24
assert contract.raw["claim_boundary"]["hourly_operational_claim"] is False
assert contract.roles[-1] == "locked_test"
assert contract.allowed_operations("validation_threshold") == ("operating_threshold",)
require_violation(
    lambda: assert_role_access("locked_test", phase="tuning", outcome_requested=False),
    "locked test was accessible during tuning",
)
assert_role_access("locked_test", phase="final_evaluation", outcome_requested=True)
require_violation(
    lambda: assert_unique_issue_ids([{"issue_id": "x"}, {"issue_id": "x"}]),
    "duplicate issue identifiers were accepted",
)
assert parse_utc("2026-09-04T12:00:00Z").isoformat().endswith("+00:00")
require_violation(
    lambda: parse_utc("2026-09-04T12:00:00"),
    "timezone-naive issue timestamp was accepted",
)
assert build_freeze_receipt(contract)["test_outcomes_accessed"] is False
print("IRIS_SEP_CONTRACT_PASS", contract.sha256)
