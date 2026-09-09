"""Bounded source-readiness probe for NOAA_CAUSAL_REDUCED_INPUT_V1.

This tool does no training and does not derive labels. It verifies that the
preregistered GOES-16 operational archive and current SWPC live interfaces can
support a separately named causal XRS+proton study without guessing variables,
interpolating gaps, or silently substituting products.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "noaa_causal_reduced_input_v1_data_contract_2026-09-09.json"
FORMAT = "IRIS_SEP_NOAA_REDUCED_INPUT_SOURCE_READINESS_V1"
USER_AGENT = "IRIS-SEP-source-readiness/1.0"
TIMEOUT = 45
MAX_FILE_BYTES = 8_000_000


class SourceReadinessError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_daily_file_links(html: str, product: str, satellite: int, day: str) -> list[str]:
    """Return deterministic operational daily NetCDF candidates from a directory page."""
    if not re.fullmatch(r"\d{8}", day):
        raise SourceReadinessError("day must be YYYYMMDD")
    prefix = rf"dn_{re.escape(product)}_g{satellite}_d{day}_v[0-9A-Za-z._-]+\.nc"
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    return sorted({link for link in links if re.fullmatch(prefix, Path(link).name)})


def _attrs(variable: Any) -> dict[str, Any]:
    out = {}
    for name in variable.ncattrs():
        value = variable.getncattr(name)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = [
                item.item() if hasattr(item, "item") else item
                for item in value
            ]
        else:
            out[name] = str(value)
    return out


def inspect_netcdf(path: Path) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise SourceReadinessError("netCDF4 is required for the online source probe") from exc

    with netCDF4.Dataset(path, mode="r") as ds:
        variables = {}
        for name, var in ds.variables.items():
            attrs = _attrs(var)
            variables[name] = {
                "dimensions": list(var.dimensions),
                "dtype": str(var.dtype),
                "shape": list(var.shape),
                "units": attrs.get("units"),
                "long_name": attrs.get("long_name"),
                "standard_name": attrs.get("standard_name"),
                "description": attrs.get("description"),
                "fill_value": attrs.get("_FillValue"),
                "flag_values": attrs.get("flag_values"),
                "flag_meanings": attrs.get("flag_meanings"),
            }
        globals_out = {}
        for key in ds.ncattrs():
            value = ds.getncattr(key)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            globals_out[key] = value if isinstance(value, (str, int, float, bool, list)) else str(value)
    return {"global_attributes": globals_out, "variables": variables}


def _text_blob(name: str, meta: dict[str, Any]) -> str:
    parts = [name]
    for key in ("long_name", "standard_name", "description", "units"):
        value = meta.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def identify_xrs_candidates(variables: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result = {"A": [], "B": []}
    for name, meta in variables.items():
        text = _text_blob(name, meta)
        if "flux" not in text and "irradiance" not in text:
            continue
        compact = re.sub(r"[^a-z0-9]", "", text)
        if "xrsa" in compact or "xraychannela" in compact:
            result["A"].append(name)
        if "xrsb" in compact or "xraychannelb" in compact:
            result["B"].append(name)
    for key in result:
        result[key] = sorted(set(result[key]))
    return result


def identify_proton_integral_candidates(variables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept only metadata that explicitly supports integral proton semantics near 10 MeV.

    This is intentionally conservative. A differential SGPS vector is not silently
    relabelled as >10 MeV integral flux. If metadata does not make the semantics
    explicit, the probe records the inventory and fails for a later human-audited,
    preregistered deterministic derivation.
    """
    candidates = []
    for name, meta in variables.items():
        text = _text_blob(name, meta)
        if "proton" not in text or "integral" not in text:
            continue
        has_10_mev = bool(
            re.search(r"(?:>|gt|greater\s+than|above)\s*10(?:\.0+)?\s*mev", text)
            or re.search(r"10(?:\.0+)?\s*mev", text)
        )
        if has_10_mev:
            candidates.append({"variable": name, "metadata_text": text})
    return sorted(candidates, key=lambda row: row["variable"])


def quality_variables(variables: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    for name, meta in variables.items():
        text = _text_blob(name, meta)
        if any(token in text for token in ("quality", "flag", "status")):
            out.append(name)
    return sorted(set(out))


def time_variables(variables: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    for name, meta in variables.items():
        text = _text_blob(name, meta)
        units = str(meta.get("units") or "").lower()
        if name.lower() == "time" or "time" in str(meta.get("standard_name") or "").lower() or " since " in units:
            out.append(name)
    return sorted(set(out))


def fetch_bytes(session: requests.Session, url: str, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    response = session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    body = response.content
    if not body or len(body) > maximum:
        raise SourceReadinessError(f"unexpected file size for {url}: {len(body)}")
    return body


def fetch_json(session: requests.Session, url: str) -> tuple[Any, dict[str, Any]]:
    body = fetch_bytes(session, url, maximum=12_000_000)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SourceReadinessError(f"invalid JSON from {url}") from exc
    return parsed, {"url": url, "bytes": len(body), "sha256": sha256_bytes(body)}


def _json_schema_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        keys = sorted({key for row in value[:200] if isinstance(row, dict) for key in row})
        return {"kind": "list", "rows": len(value), "keys": keys}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(value)}
    return {"kind": type(value).__name__}


def probe_archive_day(
    session: requests.Session,
    *,
    root: str,
    product: str,
    satellite: int,
    iso_day: str,
    output_dir: Path,
) -> dict[str, Any]:
    d = date.fromisoformat(iso_day)
    compact = d.strftime("%Y%m%d")
    directory = urljoin(root, f"{d:%Y}/{d:%m}/")
    listing = fetch_bytes(session, directory, maximum=3_000_000).decode("utf-8", errors="replace")
    candidates = parse_daily_file_links(listing, product, satellite, compact)
    if len(candidates) != 1:
        return {
            "date": iso_day,
            "directory": directory,
            "candidate_files": candidates,
            "passed": False,
            "reason": "EXACTLY_ONE_OPERATIONAL_DAILY_FILE_REQUIRED",
        }
    url = urljoin(directory, candidates[0])
    body = fetch_bytes(session, url)
    local = output_dir / candidates[0]
    local.write_bytes(body)
    inventory = inspect_netcdf(local)
    variables = inventory["variables"]
    return {
        "date": iso_day,
        "directory": directory,
        "url": url,
        "filename": candidates[0],
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "time_variables": time_variables(variables),
        "quality_variables": quality_variables(variables),
        "variable_count": len(variables),
        "inventory": inventory,
        "passed": bool(time_variables(variables)),
        "reason": "TIME_METADATA_PRESENT" if time_variables(variables) else "TIME_METADATA_MISSING",
    }


def run(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_SOURCE_READINESS_PROBE_AND_BEFORE_TRAINING":
        raise SourceReadinessError("data contract is not frozen for readiness probing")
    if contract.get("study_id") != "NOAA_CAUSAL_REDUCED_INPUT_V1":
        raise SourceReadinessError("unexpected study id")

    output.mkdir(parents=True, exist_ok=False)
    files_dir = output / "sample_files"
    files_dir.mkdir()

    session = requests.Session()
    live = {}
    for key, url in contract["live_products"].items():
        if key == "purpose":
            continue
        value, receipt = fetch_json(session, url)
        receipt["schema"] = _json_schema_summary(value)
        live[key] = receipt

    sample_dates = contract["source_readiness_probe"]["fixed_sample_dates"]
    archive = {"xrs": [], "proton": []}
    for family, spec in (("xrs", contract["historical_products"]["xrs"]), ("proton", contract["historical_products"]["proton"])):
        product = "xrsf-l2-avg1m" if family == "xrs" else "sgps-l2-avg1m"
        for iso_day in sample_dates:
            archive[family].append(
                probe_archive_day(
                    session,
                    root=spec["root"],
                    product=product,
                    satellite=16,
                    iso_day=iso_day,
                    output_dir=files_dir,
                )
            )

    xrs_candidates = []
    proton_candidates = []
    all_xrs_quality = True
    all_proton_quality = True
    for row in archive["xrs"]:
        if row.get("inventory"):
            candidate = identify_xrs_candidates(row["inventory"]["variables"])
            row["xrs_candidates"] = candidate
            xrs_candidates.append(candidate)
            all_xrs_quality = all_xrs_quality and bool(row["quality_variables"])
    for row in archive["proton"]:
        if row.get("inventory"):
            candidate = identify_proton_integral_candidates(row["inventory"]["variables"])
            row["integral_gt10mev_candidates"] = candidate
            proton_candidates.append(candidate)
            all_proton_quality = all_proton_quality and bool(row["quality_variables"])

    archive_retrieval_passed = all(
        row.get("passed") is True
        for family in archive.values()
        for row in family
    )
    xrs_semantics_passed = bool(xrs_candidates) and all(
        bool(row["A"]) and bool(row["B"])
        for row in xrs_candidates
    )
    proton_semantics_passed = bool(proton_candidates) and all(bool(row) for row in proton_candidates)
    live_schema_passed = all(
        receipt["schema"].get("kind") in {"list", "dict"}
        and bool(receipt["schema"].get("keys"))
        for receipt in live.values()
    )
    quality_metadata_passed = all_xrs_quality and all_proton_quality

    passed = bool(
        archive_retrieval_passed
        and xrs_semantics_passed
        and proton_semantics_passed
        and live_schema_passed
        and quality_metadata_passed
    )
    receipt = {
        "format": FORMAT,
        "study_id": contract["study_id"],
        "data_contract_sha256": sha256_bytes(CONTRACT.read_bytes()),
        "historical_interval": contract["historical_interval"],
        "fixed_roles": contract["fixed_roles"],
        "sample_dates": sample_dates,
        "live": live,
        "archive": archive,
        "gates": {
            "archive_retrieval_passed": archive_retrieval_passed,
            "xrs_semantics_passed": xrs_semantics_passed,
            "proton_integral_gt10mev_semantics_passed": proton_semantics_passed,
            "quality_metadata_passed": quality_metadata_passed,
            "live_schema_passed": live_schema_passed,
            "source_readiness_passed": passed,
        },
        "training_allowed": passed,
        "labels_derived": False,
        "forecast_skill_computed": False,
        "locked_test_accessed": False,
        "failure_action": None if passed else "STOP_TRAINING_AND_RESOLVE_SOURCE_SEMANTICS",
        "claim_boundary": "Source readiness only; no forecast skill or operational claim.",
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
