"""Independent fresh-source audit for IRIS-SEP.

This tool validates *source availability and cadence*, not forecasting skill.
It deliberately does not train, calibrate, threshold, or score IRIS. A fresh
forecast is allowed only after the complete causal feature interface exists.

Public checks:
- NOAA/SWPC primary GOES integral proton 7-day JSON
- NOAA/SWPC primary GOES XRS 7-day JSON
- NOAA/SWPC GOES instrument-source declaration
- JSOC DRMS CEA SHARP and CEA SHARP-NRT series metadata and a bounded recent query

No JSOC export email is needed for metadata queries and no email is recorded.
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

NOAA_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"
NOAA_XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
NOAA_SOURCE_URL = "https://services.swpc.noaa.gov/json/goes/instrument-sources.json"
# These must stay aligned with trusted_source_registry_v1.json and with the
# promoted 251-feature solar interface. The model consumes CEA SHARP keyword
# summaries, not the non-CEA hmi.sharp_720s products.
JSOC_SERIES = ("hmi.sharp_cea_720s", "hmi.sharp_cea_720s_nrt")
REQUIRED_SHARP_KEYWORDS = (
    "AREA",
    "AREA_ACR",
    "CAR_ROT",
    "CMASKL",
    "CRLN_OBS",
    "CRLT_OBS",
    "DSUN_OBS",
    "LAT_FWT",
    "LAT_MAX",
    "LAT_MIN",
    "LON_FWT",
    "LON_MAX",
    "LON_MIN",
    "MEANALP",
    "MEANGAM",
    "MEANGBH",
    "MEANGBL",
    "MEANGBT",
    "MEANGBZ",
    "MEANJZD",
    "MEANJZH",
    "MEANPOT",
    "MEANSHR",
    "NACR",
    "NOAA_AR",
    "NPIX",
    "RSUN_OBS",
    "R_VALUE",
    "SAVNCPP",
    "SHRGT45",
    "SIZE",
    "SIZE_ACR",
    "TOTPOT",
    "TOTUSJZ",
    "USFLUX",
    "USFLUXL",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_json(url: str, timeout: int = 60) -> tuple[Any, bytes]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "IRIS-SEP-fresh-source-audit/1.0"},
    )
    response.raise_for_status()
    payload = response.content
    return response.json(), payload


def _parse_times(records: list[dict[str, Any]]) -> pd.DatetimeIndex:
    values = [row.get("time_tag") for row in records if row.get("time_tag") is not None]
    times = pd.to_datetime(values, utc=True, errors="coerce")
    if bool(pd.isna(times).any()):
        raise ValueError("source contains unparsable time_tag values")
    return pd.DatetimeIndex(times)


def _cadence_summary(times: pd.DatetimeIndex) -> dict[str, Any]:
    if len(times) < 2:
        return {
            "rows": int(len(times)),
            "first": None,
            "last": None,
            "median_cadence_seconds": None,
            "max_gap_seconds": None,
            "duplicate_timestamps": 0,
        }
    ordered = times.sort_values()
    # Do not assume DatetimeIndex.asi8 is nanoseconds. Pandas 3 can preserve a
    # microsecond-resolution dtype, so dividing the raw integer representation by
    # 1e9 can silently turn a real five-minute cadence into zero seconds.
    gaps = [float((b - a).total_seconds()) for a, b in zip(ordered[:-1], ordered[1:])]
    return {
        "rows": int(len(times)),
        "first": ordered[0].isoformat(),
        "last": ordered[-1].isoformat(),
        "median_cadence_seconds": float(pd.Series(gaps, dtype=float).median()),
        "max_gap_seconds": float(max(gaps)),
        "duplicate_timestamps": int(len(times) - len(pd.Index(times).unique())),
    }


def summarize_protons(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(records, list) or not records:
        raise ValueError("empty proton source")
    frame = pd.DataFrame(records)
    required = {"time_tag", "flux"}
    if not required.issubset(frame.columns):
        raise ValueError(f"proton source missing columns {sorted(required - set(frame.columns))}")
    energy_col = "energy" if "energy" in frame.columns else None
    if energy_col:
        labels = sorted(str(x) for x in frame[energy_col].dropna().unique())
        # SWPC currently uses strings such as ">=10 MeV". Prefer that channel;
        # if source wording changes, fail closed rather than silently selecting another energy.
        matches = [
            label
            for label in labels
            if "10" in label and "MeV" in label and (">" in label or "ge" in label.lower())
        ]
        if not matches:
            raise ValueError(f"could not identify >=10 MeV proton channel; labels={labels}")
        selected = matches[0]
        use = frame[frame[energy_col].astype(str) == selected].copy()
    else:
        selected = "UNDECLARED"
        use = frame.copy()
    use["flux"] = pd.to_numeric(use["flux"], errors="coerce")
    use = use[use["flux"].notna()].copy()
    times = _parse_times(use.to_dict("records"))
    flux = use["flux"].to_numpy(dtype=float)
    if len(flux) != len(times):
        raise ValueError("proton time/flux alignment failed")
    return {
        "channel": selected,
        "cadence": _cadence_summary(times),
        "finite_flux_rows": int(np_isfinite(flux).sum()),
        "max_flux_pfu": float(np_nanmax(flux)),
        "latest_flux_pfu": float(flux[-1]),
        "samples_at_or_above_10_pfu": int((flux >= 10.0).sum()),
        "threshold_10_pfu_active_at_latest_sample": bool(flux[-1] >= 10.0),
    }


def summarize_xrs(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(records, list) or not records:
        raise ValueError("empty XRS source")
    frame = pd.DataFrame(records)
    required = {"time_tag", "flux"}
    if not required.issubset(frame.columns):
        raise ValueError(f"XRS source missing columns {sorted(required - set(frame.columns))}")
    energy_col = "energy" if "energy" in frame.columns else None
    labels = sorted(str(x) for x in frame[energy_col].dropna().unique()) if energy_col else ["UNDECLARED"]
    per_band = {}
    for label in labels:
        use = frame[frame[energy_col].astype(str) == label].copy() if energy_col else frame.copy()
        use["flux"] = pd.to_numeric(use["flux"], errors="coerce")
        use = use[use["flux"].notna()].copy()
        times = _parse_times(use.to_dict("records"))
        flux = use["flux"].to_numpy(dtype=float)
        per_band[label] = {
            "cadence": _cadence_summary(times),
            "finite_flux_rows": int(np_isfinite(flux).sum()),
            "max_flux_w_m2": float(np_nanmax(flux)),
            "latest_flux_w_m2": float(flux[-1]),
        }
    return {"bands": per_band}


def np_isfinite(values):
    # Tiny dependency-local wrapper keeps the audit easy to inspect.
    import numpy as np

    return np.isfinite(values)


def np_nanmax(values):
    import numpy as np

    return np.nanmax(values)


def _keyword_names(info: Any) -> set[str]:
    keywords = getattr(info, "keywords", None)
    if keywords is None:
        return set()
    index = getattr(keywords, "index", None)
    if index is None:
        return set()
    return {str(value) for value in index}


def audit_jsoc() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "NOT_RUN", "series": {}}
    try:
        import drms
    except Exception as exc:  # pragma: no cover - exercised in network workflow
        return {"status": "DRMS_IMPORT_FAILED", "error": type(exc).__name__, "series": {}}
    try:
        client = drms.Client()
        missing_any: dict[str, list[str]] = {}
        for series in JSOC_SERIES:
            info = client.info(series)
            keyword_names = _keyword_names(info)
            missing = sorted(set(REQUIRED_SHARP_KEYWORDS) - keyword_names)
            result["series"][series] = {
                "primekeys": [str(x) for x in info.primekeys],
                "segment_count": int(len(info.segments)),
                "keyword_count": int(len(info.keywords)),
                "required_keyword_count": len(REQUIRED_SHARP_KEYWORDS),
                "missing_required_keywords": missing,
                "note": str(info.note),
            }
            if missing:
                missing_any[series] = missing
        if missing_any:
            raise ValueError(f"CEA SHARP series missing frozen-interface keywords: {missing_any}")

        # Query only one previous UTC day of NRT metadata. Empty HARPNUM selector
        # means all active-region records during the bounded time selector. Query
        # the exact CEA product consumed by the promoted feature interface.
        now = datetime.now(timezone.utc)
        day = (now - timedelta(days=1)).date()
        start = f"{day:%Y.%m.%d}_00:00:00_TAI"
        query = f"hmi.sharp_cea_720s_nrt[][{start}/1d]"
        keys = ",".join(("HARPNUM", "T_REC") + REQUIRED_SHARP_KEYWORDS)
        recent = client.query(query, key=keys)
        result["recent_query"] = {
            "recordset": query,
            "rows": int(len(recent)),
            "unique_harps": int(pd.to_numeric(recent.get("HARPNUM"), errors="coerce").nunique()) if len(recent) else 0,
            "first_t_rec": str(recent["T_REC"].iloc[0]) if len(recent) and "T_REC" in recent else None,
            "last_t_rec": str(recent["T_REC"].iloc[-1]) if len(recent) and "T_REC" in recent else None,
            "required_keyword_count": len(REQUIRED_SHARP_KEYWORDS),
        }
        if len(recent) == 0:
            raise ValueError("bounded CEA SHARP-NRT query returned zero rows")
        result["status"] = "PASSED"
    except Exception as exc:  # preserve source failure as evidence
        result["status"] = "QUERY_FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
    return result


def run(output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise ValueError("output directory must be new")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    sources = {}
    for name, url, summarizer in (
        ("goes_primary_integral_protons_7d", NOAA_PROTON_URL, summarize_protons),
        ("goes_primary_xrs_7d", NOAA_XRS_URL, summarize_xrs),
    ):
        data, raw = _fetch_json(url)
        (output / f"{name}.json").write_bytes(raw)
        sources[name] = {
            "url": url,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "summary": summarizer(data),
        }
    inst, raw = _fetch_json(NOAA_SOURCE_URL)
    (output / "goes_instrument_sources.json").write_bytes(raw)
    sources["goes_instrument_sources"] = {
        "url": NOAA_SOURCE_URL,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "payload": inst,
    }
    jsoc = audit_jsoc()
    receipt = {
        "format": "IRIS_SEP_FRESH_SOURCE_AUDIT_V1",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "FRESH_SOURCE_AVAILABILITY_AND_CADENCE_ONLY",
        "forecast_skill_recomputed": False,
        "fresh_forecast_probability_emitted": False,
        "reason_no_fresh_forecast": "Complete causally reconstructed IRIS feature vector was not part of this source audit.",
        "no_jsoc_export_email_recorded": True,
        "sources": sources,
        "jsoc": jsoc,
    }
    (output / "fresh_source_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "format": result["format"],
                "jsoc_status": result["jsoc"].get("status"),
                "proton": result["sources"]["goes_primary_integral_protons_7d"]["summary"],
                "fresh_forecast_probability_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
