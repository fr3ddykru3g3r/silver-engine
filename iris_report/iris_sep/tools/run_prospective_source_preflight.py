"""Collect and authenticate all registered IRIS-SEP prospective source families.

This is a preflight, not a forecast.  It snapshots NOAA/SWPC, JSOC CEA-NRT,
HEK/GOES flares, DONKI CMEs and CDAW CMEs, creates acquisition receipts, runs
the trusted-source authentication contract, and then applies the frozen causal
interface disposition.  It always writes a receipt, even when one source fails.

A forecast is explicitly forbidden when the frozen 259-feature interface cannot
be formed causally.  No missing SHARP field is filled, substituted or backcast.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from iris_report.iris_sep.src.iris_sep.source_authentication import (
    build_acquisition_receipt,
    authenticate_acquisition_receipts,
)


REGISTRY = Path(__file__).resolve().parents[1] / "config" / "trusted_source_registry_v1.json"
INTERFACE_DISPOSITION = Path(__file__).resolve().parents[1] / "architecture" / "prospective_causal_interface_disposition_2026-09-08.json"
REQUIRED_SOURCE_IDS = (
    "NOAA_SWPC_PRIMARY_PROTON_7D",
    "NOAA_SWPC_PRIMARY_XRS_7D",
    "NOAA_SWPC_GOES_INSTRUMENT_SOURCES",
    "JSOC_HMI_SHARP_NRT",
    "LMSAL_HEK_GOES_FLARES",
    "NASA_CCMC_DONKI_CME",
    "NASA_GSFC_CDAW_CME",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _http_snapshot(url: str, *, params: dict[str, str] | None = None) -> tuple[bytes, datetime]:
    response = requests.get(
        url,
        params=params,
        timeout=90,
        headers={"User-Agent": "IRIS-SEP-prospective-preflight/1.0"},
    )
    response.raise_for_status()
    return response.content, datetime.now(timezone.utc)


def _noaa(source_id: str, endpoint: str, output: Path) -> dict[str, Any]:
    data, retrieved = _http_snapshot(endpoint)
    _save(output / f"{source_id}.json", data)
    payload = json.loads(data)
    times = [row.get("time_tag") for row in payload if isinstance(row, dict) and row.get("time_tag")]
    return build_acquisition_receipt(
        source_id=source_id,
        retrieved_at=retrieved,
        artifact_sha256=_sha(data),
        artifact_bytes=len(data),
        source_url=endpoint,
        observation_first_utc=min(times) if times else None,
        observation_last_utc=max(times) if times else None,
        metadata={"rows": len(payload) if isinstance(payload, list) else None},
    )


def _jsoc(output: Path, start: datetime, end: datetime) -> dict[str, Any]:
    import drms

    series = "hmi.sharp_cea_720s_nrt"
    required = [
        "HARPNUM", "T_REC", "NOAA_AR", "AREA", "AREA_ACR", "CAR_ROT", "CMASKL",
        "CRLN_OBS", "CRLT_OBS", "DSUN_OBS", "LAT_FWT", "LAT_MAX", "LAT_MIN",
        "LON_FWT", "LON_MAX", "LON_MIN", "MEANALP", "MEANGAM", "MEANGBH",
        "MEANGBL", "MEANGBT", "MEANGBZ", "MEANJZD", "MEANJZH", "MEANPOT",
        "MEANSHR", "NACR", "NPIX", "RSUN_OBS", "R_VALUE", "SAVNCPP", "SHRGT45",
        "SIZE", "SIZE_ACR", "TOTPOT", "TOTUSJZ", "USFLUX", "USFLUXL"
    ]
    client = drms.Client()
    info = client.info(series)
    available = {str(v) for v in info.keywords.index}
    missing = sorted(set(required) - available)
    query_start = start.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hours = max(1, int((end - start).total_seconds() // 3600) + 1)
    recordset = f"{series}[][{query_start}/{hours}h]"
    query_keys = [key for key in required if key in available]
    frame = client.query(recordset, key=",".join(query_keys))
    raw = frame.to_csv(index=False).encode("utf-8")
    retrieved = datetime.now(timezone.utc)
    _save(output / "JSOC_HMI_SHARP_NRT.csv", raw)
    first = str(frame["T_REC"].iloc[0]) if len(frame) and "T_REC" in frame else None
    last = str(frame["T_REC"].iloc[-1]) if len(frame) and "T_REC" in frame else None
    return build_acquisition_receipt(
        source_id="JSOC_HMI_SHARP_NRT",
        retrieved_at=retrieved,
        artifact_sha256=_sha(raw),
        artifact_bytes=len(raw),
        source_url="https://jsoc.stanford.edu",
        query_identity=recordset,
        observation_first_utc=first,
        observation_last_utc=last,
        metadata={
            "rows": int(len(frame)),
            "required_keywords": required,
            "available_required_keywords": query_keys,
            "missing_required_keywords": missing,
        },
    )


def _hek(output: Path, start: datetime, end: datetime) -> dict[str, Any]:
    from sunpy.net import Fido, attrs as a

    result = Fido.search(
        a.Time(start, end),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES",
    )
    rows: list[dict[str, Any]] = []
    try:
        table = result["hek"]
    except Exception:
        table = result[0] if len(result) else []
    fields = ("event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum")
    for row in table:
        item = {}
        for field in fields:
            try:
                value = row[field]
            except Exception:
                value = None
            item[field] = None if value is None else str(value)
        rows.append(item)
    raw = _canonical(rows)
    retrieved = datetime.now(timezone.utc)
    _save(output / "LMSAL_HEK_GOES_FLARES.json", raw)
    starts = [r["event_starttime"] for r in rows if r.get("event_starttime")]
    return build_acquisition_receipt(
        source_id="LMSAL_HEK_GOES_FLARES",
        retrieved_at=retrieved,
        artifact_sha256=_sha(raw),
        artifact_bytes=len(raw),
        source_url="https://www.lmsal.com",
        query_identity="event_type=FL;observatory=GOES",
        observation_first_utc=min(starts) if starts else None,
        observation_last_utc=max(starts) if starts else None,
        metadata={"rows": len(rows)},
    )


def _donki(output: Path, start: datetime, end: datetime) -> dict[str, Any]:
    endpoint = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
    params = {"startDate": start.date().isoformat(), "endDate": end.date().isoformat()}
    data, retrieved = _http_snapshot(endpoint, params=params)
    _save(output / "NASA_CCMC_DONKI_CME.json", data)
    payload = json.loads(data)
    times = [row.get("startTime") for row in payload if isinstance(row, dict) and row.get("startTime")]
    return build_acquisition_receipt(
        source_id="NASA_CCMC_DONKI_CME",
        retrieved_at=retrieved,
        artifact_sha256=_sha(data),
        artifact_bytes=len(data),
        source_url=endpoint,
        query_identity="startDate/endDate bounded to pre-issue lookback",
        observation_first_utc=min(times) if times else None,
        observation_last_utc=max(times) if times else None,
        metadata={"rows": len(payload) if isinstance(payload, list) else None, "params": params},
    )


def _cdaw(output: Path, issue: datetime) -> dict[str, Any]:
    yyyy, mm = issue.strftime("%Y"), issue.strftime("%m")
    url = f"https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{yyyy}_{mm}/univ{yyyy}_{mm}.html"
    data, retrieved = _http_snapshot(url)
    _save(output / "NASA_GSFC_CDAW_CME.html", data)
    return build_acquisition_receipt(
        source_id="NASA_GSFC_CDAW_CME",
        retrieved_at=retrieved,
        artifact_sha256=_sha(data),
        artifact_bytes=len(data),
        source_url=url,
        query_identity=f"monthly_catalog={yyyy}-{mm}",
        metadata={"month": f"{yyyy}-{mm}"},
    )


def run(output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise ValueError("output must be a new immutable directory")
    output.mkdir(parents=True)

    started = datetime.now(timezone.utc)
    lookback_start = started - timedelta(hours=24)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    interface = json.loads(INTERFACE_DISPOSITION.read_text(encoding="utf-8"))
    receipts: list[dict[str, Any]] = []
    failures: dict[str, dict[str, str]] = {}

    def attempt(source_id: str, fn):
        try:
            receipts.append(fn())
        except Exception as exc:  # evidence capture is more important than early abort
            failures[source_id] = {"error_type": type(exc).__name__, "error": str(exc)[:2000]}

    attempt("NOAA_SWPC_PRIMARY_PROTON_7D", lambda: _noaa(
        "NOAA_SWPC_PRIMARY_PROTON_7D",
        registry["sources"]["NOAA_SWPC_PRIMARY_PROTON_7D"]["endpoint"], output))
    attempt("NOAA_SWPC_PRIMARY_XRS_7D", lambda: _noaa(
        "NOAA_SWPC_PRIMARY_XRS_7D",
        registry["sources"]["NOAA_SWPC_PRIMARY_XRS_7D"]["endpoint"], output))
    attempt("NOAA_SWPC_GOES_INSTRUMENT_SOURCES", lambda: _noaa(
        "NOAA_SWPC_GOES_INSTRUMENT_SOURCES",
        registry["sources"]["NOAA_SWPC_GOES_INSTRUMENT_SOURCES"]["endpoint"], output))
    attempt("JSOC_HMI_SHARP_NRT", lambda: _jsoc(output, lookback_start, started))
    attempt("LMSAL_HEK_GOES_FLARES", lambda: _hek(output, lookback_start, started))
    attempt("NASA_CCMC_DONKI_CME", lambda: _donki(output, lookback_start, started))
    attempt("NASA_GSFC_CDAW_CME", lambda: _cdaw(output, started))

    # The issue time is defined only after every acquisition attempt has ended,
    # so successful acquisitions necessarily precede the hypothetical issue.
    issue_time = datetime.now(timezone.utc)
    auth = authenticate_acquisition_receipts(
        issue_time=issue_time,
        receipts=receipts,
        required_source_ids=REQUIRED_SOURCE_IDS,
    )

    jsoc_receipt = next((r for r in receipts if r.get("source_id") == "JSOC_HMI_SHARP_NRT"), None)
    missing_nrt = [] if jsoc_receipt is None else list(jsoc_receipt.get("metadata", {}).get("missing_required_keywords", []))
    interface_allows_skill = bool(interface.get("prospective_skill_allowed", False))
    source_complete = auth["authenticated_for_prospective_use"]
    forecast_admissible = bool(source_complete and interface_allows_skill and not missing_nrt)

    blockers = []
    if failures:
        blockers.append("ONE_OR_MORE_SOURCE_ACQUISITIONS_FAILED")
    if not source_complete:
        blockers.append("SOURCE_AUTHENTICATION_INCOMPLETE")
    if missing_nrt:
        blockers.append("FROZEN_259_FEATURE_INTERFACE_NOT_AVAILABLE_FROM_CEA_NRT")
    if not interface_allows_skill:
        blockers.append("FROZEN_INTERFACE_DISPOSITION_RETROSPECTIVE_ONLY")

    payload = {
        "format": "IRIS_SEP_PROSPECTIVE_SOURCE_PREFLIGHT_V1",
        "started_utc": started.isoformat(),
        "hypothetical_issue_time_utc": issue_time.isoformat(),
        "lookback_start_utc": lookback_start.isoformat(),
        "required_source_ids": list(REQUIRED_SOURCE_IDS),
        "successful_receipt_count": len(receipts),
        "failures": failures,
        "source_authentication": auth,
        "jsoc_nrt_missing_required_keywords": missing_nrt,
        "causal_interface_disposition": interface.get("disposition"),
        "forecast_probability_emitted": False,
        "prospective_skill_forecast_admissible": forecast_admissible,
        "blockers": sorted(set(blockers)),
        "no_missing_value_fill_used": True,
        "no_definitive_to_nrt_substitution_used": True,
        "no_runtime_training_used": True,
        "claim_boundary": "Preflight validates acquisition/authentication and interface admissibility only. It does not establish forecast skill.",
    }
    payload["preflight_sha256"] = _sha(_canonical(payload))
    (output / "acquisition_receipts.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    (output / "prospective_preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "format": result["format"],
        "successful_receipt_count": result["successful_receipt_count"],
        "forecast_probability_emitted": result["forecast_probability_emitted"],
        "prospective_skill_forecast_admissible": result["prospective_skill_forecast_admissible"],
        "blockers": result["blockers"],
        "failures": result["failures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
