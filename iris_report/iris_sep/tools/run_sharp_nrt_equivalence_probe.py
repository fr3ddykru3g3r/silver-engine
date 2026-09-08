"""Bounded source-equivalence probe for three frozen SHARP keywords.

Question
--------
Can the three active-region keyword values absent from ``hmi.sharp_cea_720s_nrt``
be obtained at issue time from the sibling CCD NRT series
``hmi.sharp_720s_nrt`` without changing their scientific meaning?

This tool does not forecast, train, calibrate, select a model, or open a locked
test.  It performs one predeclared source-interface check:

1. verify all four SHARP series advertise CMASKL, MEANGBL and USFLUXL;
2. verify the CCD NRT series contains recent finite values for those keywords;
3. on the fixed historical day 2024-05-10, match definitive CCD and CEA records
   by HARPNUM/T_REC and test whether the stored keyword values agree;
4. emit a receipt.  Source equivalence passes only if every fixed gate passes.

The probe is intentionally narrow.  Passing it would support using CCD-NRT as a
source-equivalent provider for these three stored SHARP keyword columns only; it
would not prove prospective forecast skill or make any other source causal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORMAT = "IRIS_SEP_SHARP_NRT_EQUIVALENCE_PROBE_V1"
KEYWORDS = ("CMASKL", "MEANGBL", "USFLUXL")
SERIES = {
    "ccd_definitive": "hmi.sharp_720s",
    "cea_definitive": "hmi.sharp_cea_720s",
    "ccd_nrt": "hmi.sharp_720s_nrt",
    "cea_nrt": "hmi.sharp_cea_720s_nrt",
}
HISTORICAL_RECORDSET_DATE = "2024.05.10_00:00:00_TAI"
HISTORICAL_WINDOW = "1d"
MIN_MATCHED_FINITE_ROWS = 20
MIN_RECENT_FINITE_ROWS_PER_KEYWORD = 20
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-9


class SharpEquivalenceProbeError(ValueError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _keyword_names(info: Any) -> set[str]:
    keywords = getattr(info, "keywords", None)
    index = getattr(keywords, "index", None)
    if index is None:
        return set()
    return {str(value) for value in index}


def _recordset(series: str, start: str, duration: str) -> str:
    return f"{series}[][{start}/{duration}]"


def _query(client: Any, series: str, start: str, duration: str) -> tuple[str, pd.DataFrame]:
    recordset = _recordset(series, start, duration)
    keys = ",".join(("HARPNUM", "T_REC") + KEYWORDS)
    frame = client.query(recordset, key=keys)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return recordset, frame


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"HARPNUM", "T_REC", *KEYWORDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SharpEquivalenceProbeError(f"query missing fields: {missing}")
    out = frame.copy()
    out["HARPNUM"] = pd.to_numeric(out["HARPNUM"], errors="coerce")
    out["T_REC"] = out["T_REC"].astype(str)
    for keyword in KEYWORDS:
        out[keyword] = pd.to_numeric(out[keyword], errors="coerce")
    return out


def _finite_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        keyword: int(np.isfinite(frame[keyword].to_numpy(dtype=float, na_value=np.nan)).sum())
        for keyword in KEYWORDS
    }


def _comparison(ccd: pd.DataFrame, cea: pd.DataFrame) -> dict[str, Any]:
    left = _normalize(ccd)
    right = _normalize(cea)
    merged = left.merge(
        right,
        on=["HARPNUM", "T_REC"],
        how="inner",
        suffixes=("_ccd", "_cea"),
        validate="one_to_one",
    )
    result: dict[str, Any] = {
        "matched_rows": int(len(merged)),
        "keyword_results": {},
    }
    all_pass = len(merged) >= MIN_MATCHED_FINITE_ROWS
    for keyword in KEYWORDS:
        a = pd.to_numeric(merged[f"{keyword}_ccd"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{keyword}_cea"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        af = a[finite]
        bf = b[finite]
        finite_rows = int(finite.sum())
        if finite_rows:
            abs_diff = np.abs(af - bf)
            denom = np.maximum(np.maximum(np.abs(af), np.abs(bf)), 1.0)
            rel_diff = abs_diff / denom
            max_abs = float(np.max(abs_diff))
            max_rel = float(np.max(rel_diff))
            if keyword == "CMASKL":
                values_equal = bool(np.array_equal(af, bf))
            else:
                values_equal = bool(np.allclose(af, bf, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=False))
        else:
            max_abs = None
            max_rel = None
            values_equal = False
        passed = finite_rows >= MIN_MATCHED_FINITE_ROWS and values_equal
        all_pass = all_pass and passed
        result["keyword_results"][keyword] = {
            "finite_matched_rows": finite_rows,
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
            "passed": passed,
        }
    result["passed"] = bool(all_pass)
    return result


def run(output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise SharpEquivalenceProbeError("output directory must not already exist")
    output.mkdir(parents=True)

    try:
        import drms
    except Exception as exc:  # pragma: no cover - network workflow
        raise SharpEquivalenceProbeError("drms is required") from exc

    started = datetime.now(timezone.utc)
    client = drms.Client()

    metadata: dict[str, Any] = {}
    metadata_pass = True
    for name, series in SERIES.items():
        info = client.info(series)
        names = _keyword_names(info)
        missing = sorted(set(KEYWORDS) - names)
        metadata[name] = {
            "series": series,
            "keyword_count": int(len(names)),
            "missing_probe_keywords": missing,
            "passed": not missing,
        }
        metadata_pass = metadata_pass and not missing

    # Recent CCD-NRT availability: previous complete UTC day, chosen at runtime
    # before querying and recorded in the receipt. This is availability evidence,
    # not an outcome-selected date.
    recent_day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    recent_start = f"{recent_day:%Y.%m.%d}_00:00:00_TAI"
    recent_recordset, recent_raw = _query(client, SERIES["ccd_nrt"], recent_start, "1d")
    recent = _normalize(recent_raw)
    recent_counts = _finite_counts(recent)
    recent_pass = len(recent) > 0 and all(
        recent_counts[keyword] >= MIN_RECENT_FINITE_ROWS_PER_KEYWORD for keyword in KEYWORDS
    )
    recent.to_csv(output / "recent_ccd_nrt_keywords.csv", index=False)

    ccd_recordset, ccd_raw = _query(
        client, SERIES["ccd_definitive"], HISTORICAL_RECORDSET_DATE, HISTORICAL_WINDOW
    )
    cea_recordset, cea_raw = _query(
        client, SERIES["cea_definitive"], HISTORICAL_RECORDSET_DATE, HISTORICAL_WINDOW
    )
    ccd = _normalize(ccd_raw)
    cea = _normalize(cea_raw)
    ccd.to_csv(output / "historical_ccd_keywords.csv", index=False)
    cea.to_csv(output / "historical_cea_keywords.csv", index=False)
    comparison = _comparison(ccd, cea)

    # The causal feature builder consumes these as stored SHARP table columns and
    # then takes min/max/avg; it does not recompute them from CEA pixel geometry.
    frozen_feature_semantics_stored_keywords = True
    passed = bool(
        metadata_pass
        and recent_pass
        and comparison["passed"]
        and frozen_feature_semantics_stored_keywords
    )

    payload = {
        "format": FORMAT,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Can CCD-NRT supply CMASKL/MEANGBL/USFLUXL as source-equivalent stored keyword values for the frozen SHARP summary features?",
        "probe_keywords": list(KEYWORDS),
        "series_metadata": metadata,
        "metadata_gate_passed": bool(metadata_pass),
        "recent_ccd_nrt": {
            "recordset": recent_recordset,
            "rows": int(len(recent)),
            "finite_rows_by_keyword": recent_counts,
            "minimum_finite_rows_per_keyword": MIN_RECENT_FINITE_ROWS_PER_KEYWORD,
            "passed": bool(recent_pass),
        },
        "historical_definitive_comparison": {
            "fixed_recordset_date": HISTORICAL_RECORDSET_DATE,
            "ccd_recordset": ccd_recordset,
            "cea_recordset": cea_recordset,
            "minimum_matched_finite_rows": MIN_MATCHED_FINITE_ROWS,
            "float_rtol": FLOAT_RTOL,
            "float_atol": FLOAT_ATOL,
            **comparison,
        },
        "frozen_feature_semantics_are_stored_keyword_summaries": frozen_feature_semantics_stored_keywords,
        "source_equivalence_gate_passed": passed,
        "allowed_if_passed": "Use authenticated hmi.sharp_720s_nrt values only for CMASKL, MEANGBL and USFLUXL positions missing from CEA-NRT, with explicit mixed-series provenance.",
        "forbidden_even_if_passed": [
            "claim prospective forecast skill from this probe",
            "replace any other frozen SHARP quantity",
            "silently hide mixed-series provenance",
            "treat metadata equality as provider-byte attestation",
        ],
        "forecast_probability_emitted": False,
        "training_performed": False,
        "locked_test_accessed": False,
    }
    payload["receipt_sha256"] = _sha256(_canonical_json(payload))
    (output / "sharp_nrt_equivalence_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "format": result["format"],
        "metadata_gate_passed": result["metadata_gate_passed"],
        "recent_ccd_nrt_passed": result["recent_ccd_nrt"]["passed"],
        "historical_comparison_passed": result["historical_definitive_comparison"]["passed"],
        "source_equivalence_gate_passed": result["source_equivalence_gate_passed"],
        "receipt_sha256": result["receipt_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
