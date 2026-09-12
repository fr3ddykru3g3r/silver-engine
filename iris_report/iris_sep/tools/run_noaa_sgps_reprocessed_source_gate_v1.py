"""Fail-closed, source-only gate for NSRRD-V1. No labels or models are imported."""
from __future__ import annotations

import argparse
from calendar import monthrange
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "noaa_sgps_reprocessed_retrospective_development_v1_source_contract_2026-09-12.json"
FORMAT = "IRIS_SEP_NSRRD_V1_SOURCE_GATE_RESULT"
TIMEOUT = 45
MAX_BYTES = 12_000_000


class SourceGateError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch(session: requests.Session, url: str, maximum: int = MAX_BYTES) -> bytes:
    response = session.get(url, timeout=TIMEOUT, headers={"User-Agent": "IRIS-SEP-NSRRD-source-gate/1.0"})
    response.raise_for_status()
    body = response.content
    if not body or len(body) > maximum:
        raise SourceGateError(f"unexpected response size for {url}: {len(body)}")
    return body


def parse_science_daily_links(html: str, year_month: str) -> list[str]:
    if not re.fullmatch(r"\d{4}-\d{2}", year_month):
        raise SourceGateError("year_month must be YYYY-MM")
    token = year_month.replace("-", "")
    pattern = rf"sci_sgps-l2-avg1m_g16_d{token}[0-3][0-9]_v[0-9A-Za-z._-]+\.nc"
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    return sorted({Path(item).name for item in links if re.fullmatch(pattern, Path(item).name)})


def filename_date(name: str) -> date:
    match = re.search(r"_d(\d{8})_", name)
    if not match:
        raise SourceGateError(f"daily date missing from {name}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def expected_dates(year_month: str, start: date, end: date) -> list[date]:
    year, month = map(int, year_month.split("-"))
    return [date(year, month, day) for day in range(1, monthrange(year, month)[1] + 1)
            if start <= date(year, month, day) <= end]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "filled"):
        value = value.filled(float("nan"))
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def inspect_netcdf(body: bytes) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise SourceGateError("netCDF4 is required") from exc
    with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
        handle.write(body)
        handle.flush()
        with netCDF4.Dataset(handle.name) as dataset:
            variables: dict[str, Any] = {}
            for name, var in dataset.variables.items():
                attrs = {key: _jsonable(var.getncattr(key)) for key in var.ncattrs()}
                item = {
                    "dimensions": list(var.dimensions), "shape": list(var.shape), "dtype": str(var.dtype),
                    "units": attrs.get("units"), "long_name": attrs.get("long_name"),
                    "fill_value": attrs.get("_FillValue"),
                }
                if name in {"DiffProtonLowerEnergy", "DiffProtonUpperEnergy", "DiffProtonEffectiveEnergy"}:
                    item["values"] = _jsonable(var[:])
                variables[name] = item
            dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
    return {"dimensions": dimensions, "variables": variables}


def flatten_finite(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, list):
        for item in value:
            result.extend(flatten_finite(item))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result.append(float(value))
    return result


def to_mev(values: list[float], units: str | None) -> list[float]:
    token = (units or "").lower()
    factor = 0.001 if "kev" in token else 1000.0 if "gev" in token else 1.0
    return [value * factor for value in values]


def historical_semantics(inventory: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    variables = inventory["variables"]
    flux = variables.get(contract["required_flux_variable"])
    energies = [variables.get(name) for name in contract["required_energy_variables"]]
    quality = [name for name in contract["required_quality_variables"] if name in variables]
    time_vars = [name for name in contract["required_time_variables_one_of"] if name in variables]
    def minus_x_values(meta: dict[str, Any] | None) -> list[float]:
        if not meta:
            return []
        values = meta.get("values")
        shape = meta.get("shape", [])
        if shape == [2, contract["required_channel_count"]] and isinstance(values, list):
            values = values[0]
        return to_mev(flatten_finite(values), meta.get("units"))
    lower = minus_x_values(energies[0])
    upper = minus_x_values(energies[1])
    effective = minus_x_values(energies[2])
    shape = (flux or {}).get("shape", [])
    dimensions = (flux or {}).get("dimensions", [])
    channel_count = contract["required_channel_count"]
    return {
        "flux_dimensions": dimensions,
        "flux_shape": shape,
        "lower_mev": lower,
        "upper_mev": upper,
        "effective_mev": effective,
        "quality_variables": quality,
        "time_variables": time_vars,
        "has_sensor_and_channels": 2 in shape and channel_count in shape,
        "energy_count_13": len(lower) == len(upper) == len(effective) == channel_count,
        "quality_and_fill_present": bool(quality and flux and flux.get("fill_value") is not None),
        "passed": bool(flux and time_vars and quality and 2 in shape and channel_count in shape
                       and len(lower) == len(upper) == len(effective) == channel_count
                       and flux.get("fill_value") is not None),
    }


def live_bands(rows: Any, required_channels: list[str]) -> dict[str, Any]:
    pattern = re.compile(r"^\s*([0-9.]+)\s*-\s*([0-9.]+)\s*(keV|MeV)\s*$", re.I)
    mapping: dict[str, list[float]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("channel") not in required_channels:
            continue
        match = pattern.match(str(row.get("energy", "")))
        if not match:
            continue
        factor = 0.001 if match.group(3).lower() == "kev" else 1.0
        mapping[row["channel"]] = [float(match.group(1)) * factor, float(match.group(2)) * factor]
    return {"mapping": mapping, "passed": set(mapping) == set(required_channels)}


def bands_match(lower: list[float], upper: list[float], live: dict[str, list[float]], channels: list[str], tolerance: float) -> dict[str, Any]:
    pairs = []
    passed = len(lower) == len(upper) == len(channels) and set(live) == set(channels)
    for index, channel in enumerate(channels):
        if not passed:
            break
        observed = [lower[index], upper[index]]
        expected = live[channel]
        okay = all(abs(a - b) <= tolerance * max(abs(b), 1e-12) for a, b in zip(observed, expected))
        pairs.append({"channel": channel, "historical_mev": observed, "live_mev": expected, "passed": okay})
        passed = passed and okay
    return {"pairs": pairs, "passed": passed}


def month_tokens(start: date, end: date) -> list[str]:
    year, month = start.year, start.month
    result = []
    while (year, month) <= (end.year, end.month):
        result.append(f"{year:04d}-{month:02d}")
        month = month + 1
        if month == 13:
            year, month = year + 1, 1
    return result


def run(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text())
    if contract.get("status") != "FROZEN_BEFORE_SOURCE_GATE_BEFORE_LABELS_BEFORE_TRAINING":
        raise SourceGateError("contract is not frozen")
    output.mkdir(parents=True, exist_ok=False)
    start, end = [date.fromisoformat(value) for value in contract["historical_interval"]]
    session = requests.Session()
    inventories = []
    listing_hashes = []
    files_by_month: dict[str, list[str]] = {}
    missing_dates = []
    for token in month_tokens(start, end):
        year, month = token.split("-")
        url = urljoin(contract["historical_sgps_root"], f"{year}/{month}/")
        body = fetch(session, url, 4_000_000)
        links = parse_science_daily_links(body.decode(errors="replace"), token)
        files_by_month[token] = links
        listing_hashes.append({"month": token, "url": url, "sha256": sha256_bytes(body), "file_count": len(links)})
        present = {filename_date(name) for name in links}
        missing_dates.extend(str(value) for value in expected_dates(token, start, end) if value not in present)
        if token in contract["anchor_months"] and links:
            for name in sorted({links[0], links[-1]}):
                file_body = fetch(session, urljoin(url, name))
                semantic = historical_semantics(inspect_netcdf(file_body), contract)
                inventories.append({"month": token, "file": name, "sha256": sha256_bytes(file_body), "bytes": len(file_body), "semantics": semantic})

    live_body = fetch(session, contract["live_differential_url"], 8_000_000)
    live_rows = json.loads(live_body)
    live = live_bands(live_rows, contract["required_live_channels"])
    reference = inventories[0]["semantics"] if inventories else {"lower_mev": [], "upper_mev": []}
    match = bands_match(reference["lower_mev"], reference["upper_mev"], live["mapping"], contract["required_live_channels"], contract["energy_relative_tolerance"])
    signature = lambda item: (item["semantics"]["flux_dimensions"], item["semantics"]["flux_shape"], item["semantics"]["lower_mev"], item["semantics"]["upper_mev"])
    gates = {
        "complete_daily_inventory": not missing_dates and all(files_by_month.values()),
        "eight_frozen_anchor_files_inspected": len(inventories) == 2 * len(contract["anchor_months"]),
        "anchor_semantics_passed": bool(inventories) and all(item["semantics"]["passed"] for item in inventories),
        "anchor_schema_and_energy_stable": bool(inventories) and all(signature(item) == signature(inventories[0]) for item in inventories),
        "live_13_channel_schema_passed": live["passed"],
        "historical_to_live_energy_match_passed": match["passed"],
        "live_sensor_reduction_verified_for_both_yaw_states": False,
    }
    gates["source_interface_ready"] = all(gates.values())
    receipt = {
        "format": FORMAT, "study_id": contract["study_id"], "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), "listing_inventory": listing_hashes,
        "missing_dates": missing_dates, "anchor_files": inventories,
        "live": {"url": contract["live_differential_url"], "sha256": sha256_bytes(live_body), **live},
        "energy_binding": match,
        "sensor_binding": {
            "status": "PENDING_BOTH_YAW_STATES",
            "reason": "NOAA documentation supports a westward reduction, but this gate has not yet immutably verified the live feed against both upright and yaw-flipped L2 sensor series.",
            "training_allowed": False
        },
        "gates": gates,
        "decision": contract["pass_action"] if gates["source_interface_ready"] else contract["failure_action"],
        "labels_derived": False, "training_performed": False, "forecast_skill_computed": False,
        "protected_outcomes_accessed": False, "claim_boundary": contract["claim_boundary"],
    }
    (output / "nsrrd_v1_source_gate.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    receipt = run(parser.parse_args().output)
    print(json.dumps({"decision": receipt["decision"], "gates": receipt["gates"]}, indent=2, sort_keys=True))
    return 0 if receipt["gates"]["source_interface_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
