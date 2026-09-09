"""Bounded source-readiness probe for IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1.

The probe does no model fitting, derives no labels, and computes no forecast
skill. It checks three predeclared years/months against the frozen source
contract, records file hashes and metadata, and fails closed if the archive
schema contradicts the preregistered study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "config" / "freshness_crossover_study_v1_preregistration_2026-09-09.json"
PROBE = ROOT / "config" / "freshness_crossover_source_probe_v1_2026-09-09.json"
FORMAT = "IRIS_SEP_FRESHNESS_CROSSOVER_SOURCE_READINESS_V1"
USER_AGENT = "IRIS-SEP-freshness-source-readiness/1.0"
TIMEOUT = 60


class SourceProbeError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fetch_bytes(session: requests.Session, url: str, *, maximum: int) -> bytes:
    response = session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    body = response.content
    if not body:
        raise SourceProbeError(f"empty response from {url}")
    if len(body) > maximum:
        raise SourceProbeError(f"download exceeds bound for {url}: {len(body)} > {maximum}")
    return body


def _doy_datetime(year: int, doy: int, hour: int, minute: int) -> datetime:
    if not 1 <= doy <= 366 or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise SourceProbeError("invalid OMNI timestamp fields")
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour, minutes=minute)


def inspect_omni_5min_ascii(body: bytes, *, expected_year: int, fill_value: float) -> dict[str, Any]:
    text = body.decode("ascii", errors="strict")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise SourceProbeError("OMNI annual file has no records")

    first = lines[0].split()
    last = lines[-1].split()
    if len(first) < 4 or len(last) < 4:
        raise SourceProbeError("OMNI records lack timestamp fields")

    first_time = _doy_datetime(*(int(first[i]) for i in range(4)))
    last_time = _doy_datetime(*(int(last[i]) for i in range(4)))
    if first_time.year != expected_year or last_time.year != expected_year:
        raise SourceProbeError("OMNI annual file year does not match requested year")

    widths: set[int] = set()
    nonfill_gt10 = 0
    negative_gt10 = 0
    parsed_rows = 0
    for line in lines:
        fields = line.split()
        widths.add(len(fields))
        if len(fields) < 7:
            raise SourceProbeError("OMNI 5-minute row too short for appended GOES columns")
        try:
            gt10, gt30, gt60 = map(float, fields[-3:])
        except ValueError as exc:
            raise SourceProbeError("OMNI appended proton columns are not numeric") from exc
        parsed_rows += 1
        if gt10 != fill_value:
            nonfill_gt10 += 1
            if gt10 < 0:
                negative_gt10 += 1
        for value in (gt30, gt60):
            if value != fill_value and value < 0:
                raise SourceProbeError("negative non-fill integral proton flux")

    if len(widths) != 1:
        raise SourceProbeError(f"OMNI annual file has inconsistent record widths: {sorted(widths)}")
    if nonfill_gt10 == 0:
        raise SourceProbeError("OMNI annual file contains no non-fill >10 MeV values")
    if negative_gt10:
        raise SourceProbeError("OMNI annual file contains negative non-fill >10 MeV values")

    return {
        "rows": parsed_rows,
        "record_columns": next(iter(widths)),
        "first_timestamp": first_time.isoformat(),
        "last_timestamp": last_time.isoformat(),
        "nonfill_gt10_rows": nonfill_gt10,
        "negative_gt10_rows": negative_gt10,
        "appended_columns": ["gt10", "gt30", "gt60"],
    }


def parse_xrs_candidates(html: str, *, satellite: int, preferred_extension: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    required = (f"g{satellite}", "xrs", "1m")
    candidates = []
    for link in links:
        name = Path(link).name.lower()
        if all(token in name for token in required) and name.endswith(preferred_extension.lower()):
            candidates.append(link)
    return sorted(set(candidates))


def _attrs(variable: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in variable.ncattrs():
        value = variable.getncattr(name)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = [item.item() if hasattr(item, "item") else item for item in value]
        else:
            out[name] = str(value)
    return out


def _text_blob(name: str, attrs: dict[str, Any]) -> str:
    values = [name]
    for key in ("long_name", "standard_name", "description", "units"):
        value = attrs.get(key)
        if value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def inspect_xrs_netcdf(path: Path) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise SourceProbeError("netCDF4 is required for XRS probing") from exc

    with netCDF4.Dataset(path, mode="r") as ds:
        variables: dict[str, dict[str, Any]] = {}
        for name, var in ds.variables.items():
            attrs = _attrs(var)
            variables[name] = {
                "dimensions": list(var.dimensions),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "units": attrs.get("units"),
                "long_name": attrs.get("long_name"),
                "standard_name": attrs.get("standard_name"),
                "description": attrs.get("description"),
                "fill_value": attrs.get("_FillValue"),
                "flag_values": attrs.get("flag_values"),
                "flag_meanings": attrs.get("flag_meanings"),
            }
        globals_out = {name: ds.getncattr(name) for name in ds.ncattrs()}

    xrs_a: list[str] = []
    xrs_b: list[str] = []
    times: list[str] = []
    quality: list[str] = []
    for name, meta in variables.items():
        text = _text_blob(name, meta)
        compact = re.sub(r"[^a-z0-9]", "", text)
        if any(token in compact for token in ("xrsa", "xraychannela")) and any(token in text for token in ("flux", "irradiance")):
            xrs_a.append(name)
        if any(token in compact for token in ("xrsb", "xraychannelb")) and any(token in text for token in ("flux", "irradiance")):
            xrs_b.append(name)
        units = str(meta.get("units") or "").lower()
        if name.lower() == "time" or "time" in str(meta.get("standard_name") or "").lower() or " since " in units:
            times.append(name)
        if any(token in text for token in ("quality", "flag", "status")):
            quality.append(name)

    return {
        "global_attribute_names": sorted(globals_out),
        "variable_count": len(variables),
        "xrs_a_candidates": sorted(set(xrs_a)),
        "xrs_b_candidates": sorted(set(xrs_b)),
        "time_candidates": sorted(set(times)),
        "quality_candidates": sorted(set(quality)),
        "variables": variables,
    }


def probe_xrs_month(
    session: requests.Session,
    *,
    directory_template: str,
    month: str,
    satellite: int,
    preferred_extension: str,
    maximum: int,
    output_dir: Path,
) -> dict[str, Any]:
    year, mon = month.split("-")
    directory = directory_template.format(year=year, month=mon)
    listing = fetch_bytes(session, directory, maximum=3_000_000).decode("utf-8", errors="replace")
    candidates = parse_xrs_candidates(listing, satellite=satellite, preferred_extension=preferred_extension)
    if len(candidates) != 1:
        return {
            "month": month,
            "directory": directory,
            "candidate_files": candidates,
            "passed": False,
            "reason": "EXACTLY_ONE_GOES15_XRS_1M_NETCDF_REQUIRED",
        }

    url = urljoin(directory, candidates[0])
    body = fetch_bytes(session, url, maximum=maximum)
    local = output_dir / Path(candidates[0]).name
    local.write_bytes(body)
    inventory = inspect_xrs_netcdf(local)
    passed = bool(
        inventory["xrs_a_candidates"]
        and inventory["xrs_b_candidates"]
        and inventory["time_candidates"]
        and inventory["quality_candidates"]
    )
    return {
        "month": month,
        "directory": directory,
        "url": url,
        "filename": local.name,
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "inventory": inventory,
        "passed": passed,
        "reason": "XRS_A_B_TIME_QUALITY_METADATA_PRESENT" if passed else "REQUIRED_XRS_METADATA_MISSING",
    }


def run(output: Path) -> dict[str, Any]:
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    if study.get("study_id") != "IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1":
        raise SourceProbeError("unexpected study contract")
    if probe.get("status") != "FROZEN_BEFORE_BOUNDED_FILE_PROBE":
        raise SourceProbeError("probe config is not frozen")
    if study.get("labels_inspected_at_freeze") is not False or study.get("model_scores_inspected_at_freeze") is not False:
        raise SourceProbeError("study contract does not preserve pre-outcome freeze")

    output.mkdir(parents=True, exist_ok=False)
    sample_dir = output / "sample_files"
    sample_dir.mkdir()
    session = requests.Session()
    maximum = int(probe["probe_boundaries"]["maximum_single_download_bytes"])

    proton_rows = []
    for year in probe["proton"]["sample_years"]:
        url = probe["proton"]["annual_url_template"].format(year=year)
        body = fetch_bytes(session, url, maximum=maximum)
        summary = inspect_omni_5min_ascii(body, expected_year=int(year), fill_value=float(probe["proton"]["fill_value"]))
        proton_rows.append({
            "year": int(year),
            "url": url,
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "summary": summary,
            "passed": True,
        })

    xrs_rows = []
    for month in probe["xrs"]["sample_months"]:
        xrs_rows.append(
            probe_xrs_month(
                session,
                directory_template=probe["xrs"]["monthly_directory_template"],
                month=month,
                satellite=int(probe["xrs"]["satellite"]),
                preferred_extension=str(probe["xrs"]["preferred_extension"]),
                maximum=maximum,
                output_dir=sample_dir,
            )
        )

    proton_pass = all(row.get("passed") is True for row in proton_rows)
    xrs_pass = all(row.get("passed") is True for row in xrs_rows)
    passed = proton_pass and xrs_pass
    receipt: dict[str, Any] = {
        "format": FORMAT,
        "study_id": study["study_id"],
        "study_contract_sha256": sha256_bytes(STUDY.read_bytes()),
        "probe_config_sha256": sha256_bytes(PROBE.read_bytes()),
        "proton": proton_rows,
        "xrs": xrs_rows,
        "gates": {
            "proton_sample_schema_passed": proton_pass,
            "xrs_sample_schema_passed": xrs_pass,
            "source_readiness_passed": passed,
        },
        "training_allowed": passed,
        "labels_derived": False,
        "forecast_skill_computed": False,
        "locked_test_accessed": False,
        "independent_final_evidence": False,
        "failure_action": None if passed else "STOP_BEFORE_LABEL_DERIVATION_OR_MODEL_FIT",
        "claim_boundary": probe["probe_boundaries"]["claim_boundary"],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    (output / "source_readiness.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    receipt = run(args.output)
    print(json.dumps({"gates": receipt["gates"], "receipt_sha256": receipt["receipt_sha256"]}, indent=2))
    return 0 if receipt["gates"]["source_readiness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
