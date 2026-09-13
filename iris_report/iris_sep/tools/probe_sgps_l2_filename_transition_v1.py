"""Source-only diagnostic for the GOES-16 SGPS L2 filename/product transition.

No labels, event catalogue rows, model fitting, thresholds, forecast skill, or
protected post-2025 outcomes are touched. The sole purpose is to discover why
the frozen V2 source-readiness parser found 0 operational daily files in later
GOES-16 SGPS months despite the NOAA directory hierarchy existing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import requests

ROOT = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg1m/"
MONTHS = ["2020-11", "2022-03", "2024-05", "2025-03"]
TIMEOUT = 45
UA = "IRIS-SEP-sgps-l2-filename-transition/1.0"


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def fetch(session: requests.Session, url: str) -> bytes:
    r = session.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    if not r.content:
        raise RuntimeError(f"empty response: {url}")
    return r.content


def href_names(html: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    names = sorted({Path(x).name for x in links if Path(x).name.endswith(".nc")})
    return names


def classify(name: str) -> str:
    if name.startswith("dn_sgps-l2-avg1m_"):
        return "DN_OPERATIONAL"
    if name.startswith("sci_sgps-l2-avg1m_"):
        return "SCIENCE_PREFIX"
    if "sgps-l2-avg1m" in name:
        return "OTHER_SGPS_AVG1M"
    return "OTHER"


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    rows = []
    for ym in MONTHS:
        year, month = ym.split("-")
        url = urljoin(ROOT, f"{year}/{month}/")
        body = fetch(session, url)
        names = href_names(body.decode("utf-8", errors="replace"))
        counts: dict[str, int] = {}
        for name in names:
            counts[classify(name)] = counts.get(classify(name), 0) + 1
        rows.append({
            "year_month": ym,
            "url": url,
            "listing_bytes": len(body),
            "listing_sha256": sha256_bytes(body),
            "netcdf_count": len(names),
            "class_counts": counts,
            "first_files": names[:5],
            "last_files": names[-5:],
        })

    receipt = {
        "study_id": "SGPS_L2_FILENAME_TRANSITION_V1",
        "source_only": True,
        "labels_read": False,
        "training_performed": False,
        "forecast_skill_computed": False,
        "protected_outcomes_accessed": False,
        "root": ROOT,
        "months": rows,
        "decision_rule": "Use this receipt only to define a new source parser/contract before any label derivation or model training. Do not reinterpret a sci_ file as operational without a separate provenance check.",
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (output / "sgps_l2_filename_transition_v1.json").write_text(text, encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
