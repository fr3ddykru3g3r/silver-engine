"""Inventory GOES-16 SGPS L2 source regimes using filenames only.

This is a source-only pretraining gate. It reads directory listings from the
frozen historical interval, but no event catalogue, labels, model scores or
protected post-2025 outcomes. The result is used to choose a provenance-homogeneous
historical development interval before label derivation.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import requests

ROOT = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg1m/"
START = date(2020, 3, 1)
END = date(2025, 4, 1)
TIMEOUT = 45
UA = "IRIS-SEP-sgps-l2-source-regimes/1.0"
FILE_RE = re.compile(r"^(dn|sci)_sgps-l2-avg1m_g16_d(\d{8})_v([0-9A-Za-z.-]+)\.nc$")


def month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def fetch_listing(session: requests.Session, url: str) -> tuple[int, bytes]:
    r = session.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
    if r.status_code == 404:
        return 404, b""
    r.raise_for_status()
    if not r.content:
        raise RuntimeError(f"empty response: {url}")
    return r.status_code, r.content


def parse_listing(html: str) -> list[dict[str, str]]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    rows = []
    for link in links:
        name = Path(link).name
        m = FILE_RE.fullmatch(name)
        if m:
            rows.append({"name": name, "prefix": m.group(1), "day": m.group(2), "version": m.group(3)})
    return sorted(rows, key=lambda r: r["name"])


def regime_key(row: dict[str, Any]) -> str:
    prefixes = row["prefixes"]
    versions = row["versions"]
    if row.get("available", True) and len(prefixes) == 1 and len(versions) == 1:
        return f"{prefixes[0]}|{versions[0]}"
    return "MIXED_OR_EMPTY"


def contiguous_regimes(months: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in months:
        key = regime_key(row)
        if not out or out[-1]["key"] != key:
            out.append({"key": key, "start_month": row["year_month"], "end_month": row["year_month"], "months": 1, "complete_months": int(row["complete_daily_coverage"])})
        else:
            out[-1]["end_month"] = row["year_month"]
            out[-1]["months"] += 1
            out[-1]["complete_months"] += int(row["complete_daily_coverage"])
    return out


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    months = []
    for y, m in month_iter(START, END):
        ym = f"{y:04d}-{m:02d}"
        url = urljoin(ROOT, f"{y:04d}/{m:02d}/")
        status, body = fetch_listing(session, url)
        expected = calendar.monthrange(y, m)[1]
        if status == 404:
            months.append({
                "year_month": ym,
                "url": url,
                "http_status": 404,
                "available": False,
                "listing_sha256": None,
                "expected_days": expected,
                "file_count": 0,
                "unique_day_count": 0,
                "complete_daily_coverage": False,
                "prefixes": [],
                "versions": [],
                "first_file": None,
                "last_file": None,
            })
            continue

        files = parse_listing(body.decode("utf-8", errors="replace"))
        days = sorted({r["day"] for r in files})
        prefixes = sorted({r["prefix"] for r in files})
        versions = sorted({r["version"] for r in files})
        months.append({
            "year_month": ym,
            "url": url,
            "http_status": status,
            "available": True,
            "listing_sha256": hashlib.sha256(body).hexdigest(),
            "expected_days": expected,
            "file_count": len(files),
            "unique_day_count": len(days),
            "complete_daily_coverage": len(days) == expected,
            "prefixes": prefixes,
            "versions": versions,
            "first_file": files[0]["name"] if files else None,
            "last_file": files[-1]["name"] if files else None,
        })

    regimes = contiguous_regimes(months)
    eligible = [r for r in regimes if r["key"] != "MIXED_OR_EMPTY" and r["months"] >= 12 and r["complete_months"] == r["months"]]
    eligible_sorted = sorted(eligible, key=lambda r: (r["months"], r["end_month"]), reverse=True)
    receipt = {
        "study_id": "SGPS_L2_SOURCE_REGIMES_V1",
        "source_only": True,
        "labels_read": False,
        "training_performed": False,
        "forecast_skill_computed": False,
        "protected_outcomes_accessed": False,
        "historical_listing_interval": {"start_month": "2020-03", "end_month": "2025-04"},
        "months": months,
        "regimes": regimes,
        "longest_complete_homogeneous_regime": eligible_sorted[0] if eligible_sorted else None,
        "selection_rule": "If a new development study is opened, use the longest contiguous >=12-month regime with one filename prefix, one product version, and complete daily filename coverage. Ties resolve to the later end month. Missing monthly directories are recorded, not treated as fatal probe errors. This rule was fixed before event labels are derived.",
        "claim_boundary": "Filename/provenance inventory only; it does not establish measurement equivalence, live-feed equivalence, forecast skill, or independent validation."
    }
    (output / "sgps_l2_source_regimes_v1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return receipt


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    result=run(args.output)
    print(json.dumps({"regimes": result["regimes"], "longest_complete_homogeneous_regime": result["longest_complete_homogeneous_regime"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
