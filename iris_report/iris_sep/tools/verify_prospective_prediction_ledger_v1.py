#!/usr/bin/env python3
"""Verify the hash-chained predictor-only prospective SEP ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_ID = "IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1"
MODEL_ID = "past_proton_active_proxy"


class LedgerError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise LedgerError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"invalid JSON at line {line_no}") from exc
    if not rows:
        raise LedgerError("ledger is empty")
    return rows


def verify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = None
    seen = set()
    previous_issue = None
    for index, row in enumerate(rows):
        if row.get("format") != "IRIS_SEP_PROSPECTIVE_PREDICTION_RECORD_V1":
            raise LedgerError(f"wrong format at row {index}")
        if row.get("study_id") != STUDY_ID or row.get("model_id") != MODEL_ID:
            raise LedgerError(f"wrong study/model at row {index}")
        if row.get("protected_outcomes_accessed") is not False:
            raise LedgerError(f"protected-outcome boundary missing at row {index}")
        if row.get("abstain") is not False:
            raise LedgerError("this V1 ledger stores emitted proxy predictions only")
        issue = parse_utc(row["issue_time_utc"])
        pred = parse_utc(row["prediction_timestamp_utc"])
        observed = parse_utc(row["selected_input"]["observation_time_utc"])
        if pred > issue or observed > issue:
            raise LedgerError(f"post-issue information at row {index}")
        if (issue.hour, issue.minute, issue.second, issue.microsecond) != (0, 0, 0, 0):
            raise LedgerError(f"issue is not 00:00 UTC at row {index}")
        if previous_issue is not None and issue <= previous_issue:
            raise LedgerError("ledger issue times are not strictly increasing")
        previous_issue = issue
        key = (row["issue_time_utc"], row["model_id"])
        if key in seen:
            raise LedgerError(f"duplicate issue/model: {key}")
        seen.add(key)
        stored = row.get("record_sha256")
        body = {k: v for k, v in row.items() if k != "record_sha256"}
        if stored != digest(body):
            raise LedgerError(f"record hash mismatch at row {index}")
        if body.get("previous_record_sha256") != previous:
            raise LedgerError(f"chain mismatch at row {index}")
        for field in ("rule_sha256", "feature_schema_sha256", "source_contract_sha256", "snapshot_receipt_sha256"):
            value = str(row.get(field, ""))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise LedgerError(f"invalid {field} at row {index}")
        alert = row.get("alert")
        probability = row.get("probability")
        if alert not in (0, 1) or probability not in (0.0, 1.0) or int(probability) != alert:
            raise LedgerError(f"deterministic rule output mismatch at row {index}")
        previous = stored
    return {
        "status": "PASS",
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "records": len(rows),
        "first_issue_time_utc": rows[0]["issue_time_utc"],
        "last_issue_time_utc": rows[-1]["issue_time_utc"],
        "terminal_record_sha256": rows[-1]["record_sha256"],
        "protected_outcomes_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(load(args.ledger))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
