#!/usr/bin/env python3
"""Create one pre-issue deterministic past-proton proxy prediction.

Input is a verified candidate snapshot receipt produced before the issue time.
The output ledger is hash chained and predictor-only; no protected target is
queried, derived, or stored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "config" / "past_proton_active_proxy_rule_v1.json"
SOURCE_CONTRACT = ROOT / "config" / "prospective_live_input_sources_v1.json"


class PredictionError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise PredictionError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def feature_schema(rule: dict[str, Any], source_contract: dict[str, Any]) -> dict[str, Any]:
    source = source_contract["sources"]["primary_integral_protons"]
    return {
        "model_id": rule["model_id"],
        "features": [{
            "name": "primary_goes_integral_proton_flux_ge10mev_pfu",
            "source_id": source["source_id"],
            "units": source["units"],
            "selection": rule["input"]["selection"],
            "maximum_observation_age_minutes_at_issue": rule["input"]["maximum_observation_age_minutes_at_issue"]
        }],
        "missing_action": rule["decision"]["missing_stale_or_invalid_input"]
    }


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PredictionError(f"ledger line {number} is invalid JSON") from exc
    return rows


def record_hash(record_without_hash: dict[str, Any]) -> str:
    return sha256_bytes(canonical(record_without_hash))


def verify_existing_chain(rows: list[dict[str, Any]]) -> None:
    previous = None
    seen = set()
    for index, row in enumerate(rows):
        stored = row.get("record_sha256")
        body = {k: v for k, v in row.items() if k != "record_sha256"}
        if stored != record_hash(body):
            raise PredictionError(f"existing ledger hash mismatch at row {index}")
        if body.get("previous_record_sha256") != previous:
            raise PredictionError(f"existing ledger chain mismatch at row {index}")
        key = (body.get("issue_time_utc"), body.get("model_id"))
        if key in seen:
            raise PredictionError(f"duplicate issue/model already in ledger: {key}")
        seen.add(key)
        previous = stored


def run(snapshot_dir: Path, ledger: Path) -> dict[str, Any]:
    receipt_path = snapshot_dir / "snapshot_receipt.json"
    if not receipt_path.is_file():
        raise PredictionError("snapshot_receipt.json missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "CANDIDATE_PREISSUE_INPUT_RECEIPT":
        raise PredictionError("snapshot is not candidate pre-issue evidence")
    if receipt.get("protected_outcomes_accessed") is not False:
        raise PredictionError("snapshot receipt does not assert protected outcomes remained untouched")

    for source in receipt.get("sources", {}).values():
        raw = snapshot_dir / source["raw_filename"]
        if not raw.is_file() or sha256_file(raw) != source["raw_sha256"]:
            raise PredictionError(f"raw input hash mismatch: {source.get('raw_filename')}")

    rule_bytes = RULE.read_bytes()
    rule = json.loads(rule_bytes)
    source_bytes = SOURCE_CONTRACT.read_bytes()
    source_contract = json.loads(source_bytes)
    if rule.get("model_id") != "past_proton_active_proxy":
        raise PredictionError("unexpected deterministic rule")
    if rule.get("status") != "RULE_FROZEN_INPUT_CAUSALITY_NOT_YET_CERTIFIED":
        raise PredictionError("rule status changed")

    issue = parse_utc(receipt["issue_time_utc"])
    completed = parse_utc(receipt["collector_completed_at_utc"])
    if completed > issue:
        raise PredictionError("snapshot completed after issue time")
    selected = receipt["selected_past_proton_input"]
    observed = parse_utc(selected["observation_time_utc"])
    if observed > issue or not selected.get("age_gate_passed", False):
        raise PredictionError("selected proton input is future or stale")
    flux = float(selected["flux_pfu"])
    if not math.isfinite(flux) or flux < 0:
        raise PredictionError("selected proton flux is invalid")

    threshold = float(rule["decision"]["threshold_pfu"])
    alert = int(flux >= threshold)
    probability = float(alert)
    created = datetime.now(timezone.utc)
    if created > issue:
        raise PredictionError("prediction creation occurred after issue time; fail closed")

    rows = load_ledger(ledger)
    verify_existing_chain(rows)
    key = (iso_z(issue), rule["model_id"])
    if any((row.get("issue_time_utc"), row.get("model_id")) == key for row in rows):
        raise PredictionError(f"prediction already exists for {key}")
    previous = rows[-1]["record_sha256"] if rows else None
    schema = feature_schema(rule, source_contract)
    body = {
        "format": "IRIS_SEP_PROSPECTIVE_PREDICTION_RECORD_V1",
        "study_id": rule["study_id"],
        "issue_time_utc": iso_z(issue),
        "prediction_timestamp_utc": iso_z(created),
        "model_id": rule["model_id"],
        "probability": probability,
        "alert": alert,
        "abstain": False,
        "selected_input": {
            "observation_time_utc": selected["observation_time_utc"],
            "satellite": selected["satellite"],
            "energy": selected["energy"],
            "flux_pfu": flux,
            "age_minutes_at_issue": selected["age_minutes_at_issue"]
        },
        "rule_sha256": sha256_bytes(rule_bytes),
        "feature_schema_sha256": sha256_bytes(canonical(schema)),
        "source_contract_sha256": sha256_bytes(source_bytes),
        "snapshot_receipt_sha256": sha256_file(receipt_path),
        "previous_record_sha256": previous,
        "protected_outcomes_accessed": False
    }
    row = dict(body)
    row["record_sha256"] = record_hash(body)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    row = run(args.snapshot_dir, args.ledger)
    print(json.dumps({
        "issue_time_utc": row["issue_time_utc"],
        "model_id": row["model_id"],
        "alert": row["alert"],
        "record_sha256": row["record_sha256"],
        "protected_outcomes_accessed": False
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
