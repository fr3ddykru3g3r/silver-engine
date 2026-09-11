#!/usr/bin/env python3
"""Capture a pre-issue NOAA/SWPC predictor snapshot for prospective SEP work.

This tool is deliberately predictor-only. It never downloads or derives protected
SEP outcomes, event counts, episode identities, or model scores. Raw public input
responses are preserved byte-for-byte with retrieval timestamps and SHA-256.

A successful snapshot is evidence that a particular value was visible to this
collector before a particular issue time. It does NOT by itself certify a source's
general publication latency and does not change features_verified_causal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "prospective_live_input_sources_v1.json"
USER_AGENT = "IRIS-SEP-prospective-input-snapshot/1.0 (+student research audit)"
TIMEOUT = (20, 90)


class SnapshotError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise SnapshotError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def fetch(s: requests.Session, url: str) -> tuple[bytes, datetime, dict[str, str]]:
    response = s.get(url, timeout=TIMEOUT)
    retrieved = datetime.now(timezone.utc)
    response.raise_for_status()
    body = response.content
    if not body:
        raise SnapshotError(f"empty response: {url}")
    headers = {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower() in {"date", "last-modified", "etag", "content-type", "content-length"}
    }
    return body, retrieved, headers


def schema_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        rows = [row for row in value if isinstance(row, dict)]
        return {
            "kind": "list",
            "row_count": len(value),
            "dictionary_row_count": len(rows),
            "keys_first_500": sorted({k for row in rows[:500] for k in row}),
        }
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(value)}
    return {"kind": type(value).__name__}


def validate_required_keys(payload: Any, required: list[str], source_id: str) -> None:
    if not isinstance(payload, list) or not payload:
        raise SnapshotError(f"{source_id}: expected non-empty JSON list")
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        raise SnapshotError(f"{source_id}: no dictionary rows")
    available = {key for row in rows[:500] for key in row}
    missing = sorted(set(required) - available)
    if missing:
        raise SnapshotError(f"{source_id}: required schema keys missing: {missing}")


def energy_key(value: Any) -> str:
    return " ".join(str(value).strip().split()).replace("≥", ">=")


def select_proton_input(payload: Any, issue: datetime, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise SnapshotError("proton payload is not a list")
    allowed = {energy_key(x) for x in config["sources"]["primary_integral_protons"]["target_energy_labels"]}
    candidates = []
    for row in payload:
        if not isinstance(row, dict) or energy_key(row.get("energy")) not in allowed:
            continue
        try:
            observed = parse_utc(str(row["time_tag"]))
            flux = float(row["flux"])
        except (KeyError, TypeError, ValueError, SnapshotError):
            continue
        if observed > issue or not math.isfinite(flux) or flux < 0:
            continue
        candidates.append((observed, str(row.get("satellite", "")), flux, energy_key(row.get("energy"))))
    if not candidates:
        raise SnapshotError("no finite >=10 MeV primary-proton observation at or before issue time")
    latest_time = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == latest_time]
    signatures = {(item[1], item[2], item[3]) for item in latest}
    if len(signatures) != 1:
        raise SnapshotError("latest >=10 MeV observation is not unique")
    observed, satellite, flux, energy = latest[0]
    age_minutes = (issue - observed).total_seconds() / 60.0
    maximum = float(config["past_proton_input_gate"]["maximum_observation_age_minutes_at_issue"])
    return {
        "observation_time_utc": iso_z(observed),
        "satellite": satellite,
        "energy": energy,
        "flux_pfu": flux,
        "age_minutes_at_issue": age_minutes,
        "maximum_allowed_age_minutes": maximum,
        "age_gate_passed": 0.0 <= age_minutes <= maximum,
    }


def run(output: Path, issue: datetime) -> dict[str, Any]:
    config_bytes = CONTRACT.read_bytes()
    config = json.loads(config_bytes)
    if config.get("study_id") != "IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1":
        raise SnapshotError("unexpected prospective source contract")
    if config.get("status") != "SOURCE_READINESS_ONLY_NOT_EXECUTION_FROZEN":
        raise SnapshotError("source contract status changed")
    if (issue.hour, issue.minute, issue.second, issue.microsecond) != (0, 0, 0, 0):
        raise SnapshotError("issue_time must be exactly 00:00:00 UTC")
    if output.exists():
        raise SnapshotError("output directory must be new and immutable")
    output.mkdir(parents=True)

    receipts: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    retrievals: list[datetime] = []
    with session() as s:
        for key in ("primary_integral_protons", "primary_xrs", "instrument_sources"):
            spec = config["sources"][key]
            body, retrieved, headers = fetch(s, spec["url"])
            retrievals.append(retrieved)
            filename = f"raw_{spec['source_id']}.json"
            (output / filename).write_bytes(body)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise SnapshotError(f"{spec['source_id']}: invalid JSON") from exc
            if spec.get("required_keys"):
                validate_required_keys(payload, list(spec["required_keys"]), spec["source_id"])
            payloads[key] = payload
            receipts[spec["source_id"]] = {
                "url": spec["url"],
                "retrieved_at_utc": iso_z(retrieved),
                "raw_filename": filename,
                "raw_sha256": sha256_bytes(body),
                "raw_bytes": len(body),
                "response_headers": headers,
                "schema": schema_summary(payload),
            }

    latest_retrieval = max(retrievals)
    proton = select_proton_input(payloads["primary_integral_protons"], issue, config)
    retrieval_gate = latest_retrieval <= issue
    candidate_gate = bool(retrieval_gate and proton["age_gate_passed"])
    status = "CANDIDATE_PREISSUE_INPUT_RECEIPT" if candidate_gate else "BLOCKED_NOT_PREISSUE_READY"
    receipt = {
        "format": "IRIS_SEP_PROSPECTIVE_INPUT_SNAPSHOT_V1",
        "study_id": config["study_id"],
        "status": status,
        "issue_time_utc": iso_z(issue),
        "collector_completed_at_utc": iso_z(latest_retrieval),
        "contract_sha256": sha256_bytes(config_bytes),
        "sources": receipts,
        "selected_past_proton_input": proton,
        "gates": {
            "all_source_retrievals_no_later_than_issue": retrieval_gate,
            "selected_proton_age_within_frozen_limit": proton["age_gate_passed"],
            "candidate_preissue_snapshot": candidate_gate,
            "features_verified_causal_changed": False
        },
        "protected_outcomes_accessed": False,
        "claim_boundary": "One timestamped public-input snapshot; not a general latency certification and not forecast-skill evidence."
    }
    path = output / "snapshot_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--issue-time", required=True, help="UTC issue timestamp, exactly 00:00:00Z")
    args = parser.parse_args()
    receipt = run(args.output_dir, parse_utc(args.issue_time))
    print(json.dumps({
        "status": receipt["status"],
        "issue_time_utc": receipt["issue_time_utc"],
        "selected_observation_time_utc": receipt["selected_past_proton_input"]["observation_time_utc"],
        "selected_observation_age_minutes": receipt["selected_past_proton_input"]["age_minutes_at_issue"],
        "protected_outcomes_accessed": False
    }, sort_keys=True))
    return 0 if receipt["status"] == "CANDIDATE_PREISSUE_INPUT_RECEIPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
