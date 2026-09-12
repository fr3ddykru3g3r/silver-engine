"""Bounded source-only SGPS L1b -> SWPC live differential-proton binding probe.

No event labels, model training, forecast scoring, or protected outcomes are read.
The probe fails closed if archive energy bands or directional/sensor reduction are
ambiguous rather than inventing a mapping after the fact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "sgps_l1b_live_binding_v1_2026-09-12.json"
FORMAT = "IRIS_SEP_SGPS_L1B_LIVE_BINDING_RESULT_V1"
USER_AGENT = "IRIS-SEP-sgps-l1b-live-binding/1.0"
TIMEOUT = 45
MAX_BYTES = 30_000_000
RELATIVE_ENERGY_TOLERANCE = 0.02


class BindingProbeError(RuntimeError):
    pass


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def fetch_bytes(session: requests.Session, url: str, maximum: int = MAX_BYTES) -> bytes:
    response = session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    body = response.content
    if not body or len(body) > maximum:
        raise BindingProbeError(f"unexpected response size for {url}: {len(body)}")
    return body


def parse_month_netcdf_links(html: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+\.nc)["\']', html, flags=re.I)
    return sorted({Path(link).name for link in links if Path(link).name.lower().endswith(".nc")})


def _jsonable(value: Any) -> Any:
    if hasattr(value, "filled"):
        value = value.filled(float("nan"))
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect_netcdf(body: bytes) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise BindingProbeError("netCDF4 is required") from exc

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(body)
        tmp.flush()
        with netCDF4.Dataset(tmp.name, "r") as ds:
            dims = {name: len(dim) for name, dim in ds.dimensions.items()}
            variables: dict[str, Any] = {}
            for name, var in ds.variables.items():
                attrs = {}
                for attr in var.ncattrs():
                    attrs[attr] = _jsonable(var.getncattr(attr))
                entry = {
                    "dimensions": list(var.dimensions),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "units": attrs.get("units"),
                    "long_name": attrs.get("long_name"),
                    "standard_name": attrs.get("standard_name"),
                    "description": attrs.get("description"),
                    "fill_value": attrs.get("_FillValue"),
                }
                if var.size <= 100 and str(var.dtype).lower() not in {"str", "object"}:
                    try:
                        entry["values"] = _jsonable(var[:])
                    except Exception:
                        entry["values"] = None
                variables[name] = entry
            globals_out = {name: _jsonable(ds.getncattr(name)) for name in ds.ncattrs()}
    return {"dimensions": dims, "variables": variables, "global_attributes": globals_out}


def metadata_text(name: str, meta: dict[str, Any]) -> str:
    return " ".join(
        str(v)
        for v in (
            name,
            meta.get("long_name"),
            meta.get("standard_name"),
            meta.get("description"),
            meta.get("units"),
        )
        if v is not None
    ).lower()


def identify_archive_semantics(inventory: dict[str, Any]) -> dict[str, Any]:
    flux_candidates: list[str] = []
    energy_candidates: list[str] = []
    time_candidates: list[str] = []
    quality_candidates: list[str] = []
    for name, meta in inventory["variables"].items():
        text = metadata_text(name, meta)
        if "proton" in text and "differential" in text and "flux" in text:
            flux_candidates.append(name)
        if "proton" in text and "energy" in text and any(k in text for k in ("differential", "band", "channel")):
            energy_candidates.append(name)
        units = str(meta.get("units") or "").lower()
        if name.lower() == "time" or " since " in units or "time" in str(meta.get("standard_name") or "").lower():
            time_candidates.append(name)
        if any(token in text for token in ("quality", "dqf", "valid sample", "validsample", "status flag")):
            quality_candidates.append(name)

    channel_dims = set()
    ambiguous_extra_dims = set()
    dims = inventory["dimensions"]
    for name in flux_candidates:
        meta = inventory["variables"][name]
        for dim in meta["dimensions"]:
            n = dims.get(dim)
            if n == 13:
                channel_dims.add(dim)
            elif n not in (None, 1) and "time" not in dim.lower():
                ambiguous_extra_dims.add((dim, n))

    bands = derive_archive_energy_bands(inventory, energy_candidates)
    return {
        "flux_candidates": sorted(flux_candidates),
        "energy_candidates": sorted(energy_candidates),
        "time_candidates": sorted(time_candidates),
        "quality_candidates": sorted(quality_candidates),
        "channel_dimensions_length_13": sorted(channel_dims),
        "non_time_non_channel_dimensions": [
            {"dimension": dim, "size": size} for dim, size in sorted(ambiguous_extra_dims)
        ],
        "derived_energy_bands_mev": bands,
        "has_unambiguous_13_channel_flux": bool(flux_candidates and channel_dims and not ambiguous_extra_dims),
        "has_13_energy_bands": len(bands) == 13,
        "metadata_ready": bool(time_candidates and quality_candidates),
    }


def _flatten_numeric(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_numeric(item))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        out.append(float(value))
    return out


def _to_mev(values: Iterable[float], units: str | None) -> list[float]:
    units_l = (units or "").lower().replace(" ", "")
    factor = 1.0
    if "kev" in units_l:
        factor = 1e-3
    elif "gev" in units_l:
        factor = 1e3
    return [float(v) * factor for v in values]


def derive_archive_energy_bands(inventory: dict[str, Any], candidates: list[str]) -> list[list[float]]:
    """Derive 13 [low, high] bands only from explicit small energy arrays.

    Accepted forms are a 13x2 bounds variable or separate 13-value lower/upper
    arrays. Effective-energy-only metadata is recorded by the caller but is not
    sufficient for a numeric band match.
    """
    lower: list[float] | None = None
    upper: list[float] | None = None
    for name in candidates:
        meta = inventory["variables"][name]
        values = meta.get("values")
        if values is None:
            continue
        text = metadata_text(name, meta)
        flat = _to_mev(_flatten_numeric(values), meta.get("units"))
        shape = meta.get("shape") or []
        if shape in ([13, 2], [2, 13]) and len(flat) == 26:
            if shape == [13, 2]:
                return [[flat[i * 2], flat[i * 2 + 1]] for i in range(13)]
            return [[flat[i], flat[13 + i]] for i in range(13)]
        if len(flat) == 13 and "lower" in text:
            lower = flat
        if len(flat) == 13 and "upper" in text:
            upper = flat
    if lower is not None and upper is not None:
        return [[lo, hi] for lo, hi in zip(lower, upper)]
    return []


def parse_live_energy_band(text: str) -> list[float] | None:
    """Parse a live energy label such as '1020-1860 keV' or '1.02-1.86 MeV'."""
    cleaned = text.lower().replace("–", "-").replace("—", "-")
    nums = [float(x) for x in re.findall(r"(?<![a-z])([0-9]+(?:\.[0-9]+)?)", cleaned)]
    if len(nums) < 2:
        return None
    factor = 1.0
    if "kev" in cleaned:
        factor = 1e-3
    elif "gev" in cleaned:
        factor = 1e3
    return [nums[0] * factor, nums[1] * factor]


def summarize_live(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        return {"passed": False, "reason": "LIVE_JSON_NOT_NONEMPTY_LIST"}
    rows = [row for row in value if isinstance(row, dict)]
    keys = sorted({k for row in rows[:500] for k in row})
    energy_labels = sorted({str(row.get("energy")) for row in rows if row.get("energy") is not None})
    channels = sorted({str(row.get("channel")) for row in rows if row.get("channel") is not None})
    parsed = []
    for label in energy_labels:
        band = parse_live_energy_band(label)
        if band is not None:
            parsed.append({"label": label, "band_mev": band})
    required = {"time_tag", "flux", "energy", "channel"}
    return {
        "keys": keys,
        "row_count": len(rows),
        "energy_labels": energy_labels,
        "channels": channels,
        "parsed_energy_bands": parsed,
        "passed": required.issubset(set(keys)) and len(parsed) == 13,
    }


def bands_match(archive: list[list[float]], live: list[list[float]], tolerance: float) -> dict[str, Any]:
    if len(archive) != 13 or len(live) != 13:
        return {"passed": False, "pairs": [], "reason": "THIRTEEN_BANDS_REQUIRED"}
    archive_sorted = sorted(archive)
    live_sorted = sorted(live)
    pairs = []
    passed = True
    for a, b in zip(archive_sorted, live_sorted):
        diffs = []
        for av, bv in zip(a, b):
            denom = max(abs(av), abs(bv), 1e-12)
            diffs.append(abs(av - bv) / denom)
        ok = max(diffs) <= tolerance
        passed = passed and ok
        pairs.append({"archive_mev": a, "live_mev": b, "relative_differences": diffs, "passed": ok})
    return {"passed": passed, "pairs": pairs, "relative_tolerance": tolerance}


def probe_month(session: requests.Session, root: str, month: str) -> dict[str, Any]:
    year, mon = month.split("-")
    directory = urljoin(root, f"{year}/{mon}/")
    listing = fetch_bytes(session, directory, maximum=4_000_000).decode("utf-8", errors="replace")
    links = parse_month_netcdf_links(listing)
    if not links:
        return {"month": month, "directory": directory, "file_count": 0, "passed": False, "reason": "NO_NETCDF_FILES"}
    selected = links[0]
    body = fetch_bytes(session, urljoin(directory, selected))
    inventory = inspect_netcdf(body)
    semantics = identify_archive_semantics(inventory)
    return {
        "month": month,
        "directory": directory,
        "file_count": len(links),
        "selected_file": selected,
        "selection_rule": "LEXICOGRAPHIC_FIRST_NETCDF_FILE",
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "semantics": semantics,
        "passed": bool(
            semantics["has_unambiguous_13_channel_flux"]
            and semantics["has_13_energy_bands"]
            and semantics["metadata_ready"]
        ),
    }


def run(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_SOURCE_ONLY_PROBE":
        raise BindingProbeError("binding contract not frozen")
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()

    archive_rows = [probe_month(session, contract["historical_root"], month) for month in contract["anchor_months"]]
    live_body = fetch_bytes(session, contract["live_url"])
    live_value = json.loads(live_body)
    live = summarize_live(live_value)

    routing_body = fetch_bytes(session, contract["routing_url"])
    routing_value = json.loads(routing_body)
    routing_keys = sorted(routing_value.keys()) if isinstance(routing_value, dict) else sorted({k for r in routing_value[:100] if isinstance(r, dict) for k in r}) if isinstance(routing_value, list) else []

    archive_reference = next((row for row in archive_rows if row.get("semantics", {}).get("derived_energy_bands_mev")), None)
    archive_bands = archive_reference["semantics"]["derived_energy_bands_mev"] if archive_reference else []
    live_bands = [row["band_mev"] for row in live.get("parsed_energy_bands", [])]
    match = bands_match(archive_bands, live_bands, RELATIVE_ENERGY_TOLERANCE)

    all_archive = all(row.get("passed") is True for row in archive_rows)
    gates = {
        "all_anchor_months_retrievable_and_semantic": all_archive,
        "live_differential_schema_and_13_bands": bool(live.get("passed")),
        "numeric_energy_band_match": bool(match.get("passed")),
        "routing_schema_nonempty": bool(routing_keys),
    }
    gates["binding_supported"] = all(gates.values())

    result = {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "training_performed": False,
        "labels_derived": False,
        "forecast_skill_computed": False,
        "protected_outcomes_accessed": False,
        "archive": archive_rows,
        "live": {
            **live,
            "url": contract["live_url"],
            "bytes": len(live_body),
            "sha256": sha256_bytes(live_body),
        },
        "routing": {
            "url": contract["routing_url"],
            "bytes": len(routing_body),
            "sha256": sha256_bytes(routing_body),
            "keys": routing_keys,
        },
        "energy_match": match,
        "gates": gates,
        "decision": "L1B_BINDING_ACCEPTED_FOR_SEPARATE_FROZEN_STUDY" if gates["binding_supported"] else "L1B_BINDING_REJECTED_DO_NOT_TRAIN",
        "claim_boundary": contract["claim_boundary"],
    }
    (output / "sgps_l1b_live_binding_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"gates": result["gates"], "decision": result["decision"]}, indent=2, sort_keys=True))
    return 0 if result["gates"]["binding_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
