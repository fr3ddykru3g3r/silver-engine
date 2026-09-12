"""Source-interface readiness probe for NOAA_CAUSAL_REDUCED_INPUT_V2.

This probe is intentionally bounded. It performs no training, derives no model
labels, computes no forecast skill, and does not touch the protected post-2025-09-10
outcome pool. It verifies only that the frozen V2 predictor and official-label
interfaces are technically usable before the separate pretraining completeness gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html as html_lib
import io
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "noaa_causal_reduced_input_v2_data_contract_2026-09-12.json"
FORMAT = "IRIS_SEP_NOAA_REDUCED_INPUT_SOURCE_READINESS_V2"
USER_AGENT = "IRIS-SEP-source-readiness-v2/1.0"
TIMEOUT = 45
MAX_FILE_BYTES = 12_000_000


class SourceReadinessV2Error(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(session: requests.Session, url: str, *, maximum: int = MAX_FILE_BYTES) -> bytes:
    response = session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    body = response.content
    if not body or len(body) > maximum:
        raise SourceReadinessV2Error(f"unexpected file size for {url}: {len(body)}")
    return body


def fetch_json(session: requests.Session, url: str) -> tuple[Any, dict[str, Any]]:
    body = fetch_bytes(session, url)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SourceReadinessV2Error(f"invalid JSON from {url}") from exc
    return value, {"url": url, "bytes": len(body), "sha256": sha256_bytes(body)}


def schema_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        keys = sorted({key for row in value[:500] if isinstance(row, dict) for key in row})
        return {"kind": "list", "rows": len(value), "keys": keys}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(value)}
    return {"kind": type(value).__name__, "keys": []}


def parse_operational_daily_links(html: str, product: str, satellite: int, year_month: str) -> list[str]:
    """Return operational daily files in one frozen month, without reading labels."""
    if not re.fullmatch(r"\d{4}-\d{2}", year_month):
        raise SourceReadinessV2Error("year_month must be YYYY-MM")
    yyyymm = year_month.replace("-", "")
    pattern = rf"dn_{re.escape(product)}_g{satellite}_d{yyyymm}[0-3][0-9]_v[0-9A-Za-z._-]+\.nc"
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    return sorted({Path(link).name for link in links if re.fullmatch(pattern, Path(link).name)})


def inspect_netcdf_bytes(body: bytes) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as exc:
        raise SourceReadinessV2Error("netCDF4 is required") from exc

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(body)
        tmp.flush()
        with netCDF4.Dataset(tmp.name, mode="r") as ds:
            variables: dict[str, dict[str, Any]] = {}
            for name, var in ds.variables.items():
                attrs: dict[str, Any] = {}
                for attr in var.ncattrs():
                    value = var.getncattr(attr)
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        attrs[attr] = value
                    elif isinstance(value, (list, tuple)):
                        attrs[attr] = [item.item() if hasattr(item, "item") else item for item in value]
                    else:
                        attrs[attr] = str(value)
                variables[name] = {
                    "dimensions": list(var.dimensions),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "units": attrs.get("units"),
                    "long_name": attrs.get("long_name"),
                    "standard_name": attrs.get("standard_name"),
                    "fill_value": attrs.get("_FillValue"),
                }
            globals_out: dict[str, Any] = {}
            for attr in ds.ncattrs():
                value = ds.getncattr(attr)
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                globals_out[attr] = value if isinstance(value, (str, int, float, bool, list)) else str(value)
    return {"variables": variables, "global_attributes": globals_out}


def _text(name: str, meta: dict[str, Any]) -> str:
    return " ".join(
        str(v) for v in [name, meta.get("long_name"), meta.get("standard_name"), meta.get("units")] if v is not None
    ).lower()


def xrs_semantics(inventory: dict[str, Any]) -> dict[str, Any]:
    variables = inventory["variables"]
    a, b, time_vars, quality = [], [], [], []
    for name, meta in variables.items():
        text = re.sub(r"[^a-z0-9]+", "", _text(name, meta))
        raw = _text(name, meta)
        if ("xrsa" in text or "xraychannela" in text) and ("flux" in raw or "irradiance" in raw):
            a.append(name)
        if ("xrsb" in text or "xraychannelb" in text) and ("flux" in raw or "irradiance" in raw):
            b.append(name)
        units = str(meta.get("units") or "").lower()
        if name.lower() == "time" or " since " in units or "time" in str(meta.get("standard_name") or "").lower():
            time_vars.append(name)
        if any(token in raw for token in ("quality", "flag", "status", "valid")):
            quality.append(name)
    return {
        "xrs_a": sorted(set(a)),
        "xrs_b": sorted(set(b)),
        "time_variables": sorted(set(time_vars)),
        "quality_variables": sorted(set(quality)),
        "passed": bool(a and b and time_vars and quality),
    }


def sgps_differential_semantics(inventory: dict[str, Any]) -> dict[str, Any]:
    variables = inventory["variables"]
    flux = []
    lower = []
    upper = []
    effective = []
    time_vars = []
    quality = []
    forbidden_integral_gt10 = []
    for name, meta in variables.items():
        raw = _text(name, meta)
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if "differential" in raw and "proton" in raw and "flux" in raw:
            flux.append(name)
        if "diffprotonlowerenergy" in compact or ("lower" in raw and "energy" in raw and "proton" in raw):
            lower.append(name)
        if "diffprotonupperenergy" in compact or ("upper" in raw and "energy" in raw and "proton" in raw):
            upper.append(name)
        if "diffprotoneffectiveenergy" in compact or ("effective" in raw and "energy" in raw and "proton" in raw):
            effective.append(name)
        units = str(meta.get("units") or "").lower()
        if name.lower() == "time" or " since " in units or "time" in str(meta.get("standard_name") or "").lower():
            time_vars.append(name)
        if any(token in raw for token in ("quality", "flag", "valid", "dqf")):
            quality.append(name)
        if "integral" in raw and "proton" in raw and re.search(r"(?:>|gt|above|greaterthan)?10(?:\.0+)?mev", compact):
            forbidden_integral_gt10.append(name)
    passed = bool(flux and (lower or effective) and (upper or effective) and time_vars and quality and not forbidden_integral_gt10)
    return {
        "differential_flux": sorted(set(flux)),
        "lower_energy": sorted(set(lower)),
        "upper_energy": sorted(set(upper)),
        "effective_energy": sorted(set(effective)),
        "time_variables": sorted(set(time_vars)),
        "quality_variables": sorted(set(quality)),
        "direct_integral_gt10_candidates": sorted(set(forbidden_integral_gt10)),
        "passed": passed,
    }


def probe_anchor_month(
    session: requests.Session,
    *,
    root: str,
    product: str,
    satellite: int,
    year_month: str,
    semantic_fn,
) -> dict[str, Any]:
    year, month = year_month.split("-")
    directory = urljoin(root, f"{year}/{month}/")
    listing_body = fetch_bytes(session, directory, maximum=4_000_000)
    links = parse_operational_daily_links(listing_body.decode("utf-8", errors="replace"), product, satellite, year_month)
    if not links:
        return {
            "year_month": year_month,
            "directory": directory,
            "operational_file_count": 0,
            "passed": False,
            "reason": "NO_OPERATIONAL_DAILY_FILE_IN_FROZEN_MONTH",
        }
    selected = links[0]
    url = urljoin(directory, selected)
    body = fetch_bytes(session, url)
    inventory = inspect_netcdf_bytes(body)
    semantics = semantic_fn(inventory)
    return {
        "year_month": year_month,
        "directory": directory,
        "selection_rule": "LEXICOGRAPHIC_FIRST_OPERATIONAL_DAILY_FILE",
        "operational_file_count": len(links),
        "selected_file": selected,
        "url": url,
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "semantics": semantics,
        "passed": bool(semantics.get("passed")),
        "reason": "SEMANTICS_PRESENT" if semantics.get("passed") else "SEMANTICS_MISSING",
    }


def event_catalog_semantics(html: str) -> dict[str, Any]:
    decoded = html_lib.unescape(html)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", decoded)).lower()
    checks = {
        "title": "solar proton events affecting the earth environment" in text,
        "begin_time_header": "begin time" in text,
        "maximum_header": "maximum time" in text and ">10 mev maximum" in text,
        "start_definition": "start of a proton event" in text and "3 consecutive" in text and "10 pfu" in text,
        "end_definition": "end of an event" in text and "last time" in text and "10 pfu" in text,
    }
    return {"checks": checks, "passed": all(checks.values())}


def extract_pdf_text(body: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SourceReadinessV2Error("pypdf is required") from exc
    reader = PdfReader(io.BytesIO(body))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def post_event_report_semantics(text: str) -> dict[str, Any]:
    flat = re.sub(r"\s+", " ", text).lower()
    has_proton = "proton" in flat and ("10 mev" in flat or "10mev" in flat)
    has_start = bool(re.search(r"\bbegan\b|\bbegin\b|\bstarted\b", flat))
    has_end = bool(re.search(r"\bended\b|\bend(?:ed)? at\b|decreased below event threshold|dropped below event threshold", flat))
    return {
        "has_proton_language": has_proton,
        "has_start_language": has_start,
        "has_end_language": has_end,
        "passed": bool(has_proton and has_start and has_end),
    }


def run(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_V2_SOURCE_READINESS_AND_BEFORE_TRAINING":
        raise SourceReadinessV2Error("V2 data contract is not frozen")
    if contract.get("study_id") != "NOAA_CAUSAL_REDUCED_INPUT_V2":
        raise SourceReadinessV2Error("unexpected study id")

    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()

    anchors = contract["source_readiness_probe"]["predictor_anchor_months"]
    xrs_rows = [
        probe_anchor_month(
            session,
            root=contract["historical_products"]["xrs"]["root"],
            product="xrsf-l2-avg1m",
            satellite=16,
            year_month=month,
            semantic_fn=xrs_semantics,
        )
        for month in anchors
    ]
    sgps_rows = [
        probe_anchor_month(
            session,
            root=contract["historical_products"]["proton_predictors"]["root"],
            product="sgps-l2-avg1m",
            satellite=16,
            year_month=month,
            semantic_fn=sgps_differential_semantics,
        )
        for month in anchors
    ]

    live: dict[str, Any] = {}
    live_requirements = {
        "xrs": {"time_tag", "flux"},
        "proton_predictors": {"time_tag", "flux", "energy"},
        "routing": set(),
    }
    live_pass = True
    for key, url in contract["live_products"].items():
        if key == "compatibility_rule":
            continue
        value, receipt = fetch_json(session, url)
        summary = schema_summary(value)
        receipt["schema"] = summary
        keys = set(summary.get("keys", []))
        required = live_requirements[key]
        passed = summary.get("kind") in {"list", "dict"} and bool(keys) and required.issubset(keys)
        receipt["passed"] = passed
        receipt["required_keys"] = sorted(required)
        live[key] = receipt
        live_pass = live_pass and passed

    catalog_body = fetch_bytes(session, contract["label_products"]["event_catalog"], maximum=2_000_000)
    catalog_sem = event_catalog_semantics(catalog_body.decode("utf-8", errors="replace"))
    catalog = {
        "url": contract["label_products"]["event_catalog"],
        "bytes": len(catalog_body),
        "sha256": sha256_bytes(catalog_body),
        "semantics": catalog_sem,
    }

    witness_urls = [
        "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/weekly_reports/PRFs_of_SGD/2024/02/prf2528.pdf",
        "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/weekly_reports/PRFs_of_SGD/2024/05/prf2541.pdf"
    ]
    witnesses = []
    for url in witness_urls:
        body = fetch_bytes(session, url, maximum=8_000_000)
        text = extract_pdf_text(body)
        sem = post_event_report_semantics(text)
        witnesses.append({
            "url": url,
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "semantics": sem,
        })

    gates = {
        "xrs_anchor_semantics_passed": all(row.get("passed") is True for row in xrs_rows),
        "sgps_differential_anchor_semantics_passed": all(row.get("passed") is True for row in sgps_rows),
        "official_event_catalog_semantics_passed": bool(catalog_sem.get("passed")),
        "official_post_event_end_interface_passed": all(row["semantics"].get("passed") is True for row in witnesses),
        "live_schema_passed": live_pass,
    }
    gates["source_interface_ready"] = all(gates.values())

    receipt = {
        "format": FORMAT,
        "study_id": contract["study_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "protected_outcome_boundary": "2025-09-10T00:00:00Z",
        "protected_outcomes_accessed": False,
        "training_performed": False,
        "labels_derived": False,
        "forecast_skill_computed": False,
        "xrs_anchor_months": xrs_rows,
        "sgps_differential_anchor_months": sgps_rows,
        "live": live,
        "event_catalog": catalog,
        "post_event_report_witnesses": witnesses,
        "gates": gates,
        "next_gate": "PRETRAINING_COMPLETENESS_REQUIRED_BEFORE_ANY_MODEL_FIT",
        "claim_boundary": contract["claim_boundary"],
    }
    canonical = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output / "source_readiness_v2.json").write_text(canonical, encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = run(args.output)
    print(json.dumps({
        "gates": receipt["gates"],
        "training_performed": receipt["training_performed"],
        "protected_outcomes_accessed": receipt["protected_outcomes_accessed"],
    }, indent=2, sort_keys=True))
    return 0 if receipt["gates"]["source_interface_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
