"""Fail-closed access and integrity checks for the IRIS-SEP benchmark.

This module deliberately knows nothing about test outcomes. Its job is to make
illegal tuning-time access difficult and auditable before model code exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractViolation(RuntimeError):
    """Raised when data or evaluation behavior violates the frozen contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkContract:
    raw: Mapping[str, Any]
    path: Path
    sha256: str

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.raw["partitioning"]["roles"])

    def assert_frozen(self) -> None:
        if self.raw.get("status") not in {
            "FROZEN_BEFORE_DATA_ACCESS",
            "FROZEN_AFTER_SCHEMA_METADATA_BEFORE_LOCKED_TEST_ACCESS",
        }:
            raise ContractViolation("benchmark contract is not frozen")
        if self.roles[-1] != "locked_test":
            raise ContractViolation("locked_test must be the final partition role")
        if self.raw["selection"].get("test_labels_available_during_tuning") is not False:
            raise ContractViolation("test labels must be unavailable during tuning")
        if self.raw["selection"].get("test_predictions_available_during_tuning") is not False:
            raise ContractViolation("test predictions must be unavailable during tuning")

    def allowed_operations(self, role: str) -> tuple[str, ...]:
        try:
            return tuple(self.raw["fitting_permissions"][role])
        except KeyError as exc:
            raise ContractViolation(f"unknown partition role: {role}") from exc


def load_contract(path: str | Path) -> BenchmarkContract:
    contract_path = Path(path).resolve()
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = BenchmarkContract(raw=raw, path=contract_path, sha256=sha256_file(contract_path))
    contract.assert_frozen()
    return contract


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractViolation(f"timestamp is not timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


def assert_unique_issue_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        issue_id = str(row.get("issue_id", "")).strip()
        if not issue_id:
            raise ContractViolation("every row requires a nonempty issue_id")
        if issue_id in seen:
            duplicates.add(issue_id)
        seen.add(issue_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise ContractViolation(f"duplicate issue_id values: {preview}")


def assert_role_access(role: str, *, phase: str, outcome_requested: bool) -> None:
    """Enforce the one-way locked-test boundary.

    ``phase`` is either ``tuning`` or ``final_evaluation``. Even feature-only
    previews of the locked test are rejected during tuning to prevent cohort
    tailoring and accidental prediction peeking.
    """

    if phase not in {"tuning", "final_evaluation"}:
        raise ContractViolation(f"unknown phase: {phase}")
    if role == "locked_test" and phase != "final_evaluation":
        raise ContractViolation("locked_test is inaccessible during tuning")
    if outcome_requested and role == "locked_test" and phase != "final_evaluation":
        raise ContractViolation("locked-test outcomes are inaccessible during tuning")


def build_freeze_receipt(contract: BenchmarkContract) -> dict[str, Any]:
    return {
        "receipt_type": "IRIS_SEP_BENCHMARK_CONTRACT_FREEZE",
        "contract_id": contract.raw["contract_id"],
        "contract_sha256": contract.sha256,
        "status": "PASS",
        "test_outcomes_accessed": False,
    }
