#!/usr/bin/env python3
"""Materialize the deterministic real-image subset used by the Colab runner.

Direct SUM URLs are not a reliable acquisition route for this project.  This
script uses the documented JSOC DRMS export interface, which requires the
student's own registered JSOC email, and stores one validated FITS file per
``sample_id``.  It is resumable: already-valid files are never re-downloaded.

The script deliberately downloads only the current runner's planned samples,
not every native SHARP record in the historical manifest.  The exact plan is
computed by ``iris-model/fit_cache.py`` and is checked again immediately before
training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "iris-model"
COMMON_DIR = REPO_ROOT / "common"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from fit_cache import is_fits_payload, is_readable_fits, required_records, verify_local_cache
from jsoc_time import parse_jsoc_trec_to_utc


TREC_RE = re.compile(r"\[(?P<harp>\d+)\]\[(?P<trec>[^\]]+)\]")


def normalize_trec(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip()).upper()


def record_key(value: object) -> tuple[int, str] | None:
    match = TREC_RE.search(str(value))
    if not match:
        return None
    return int(match.group("harp")), normalize_trec(match.group("trec"))


def scope_flags(scope: str) -> tuple[bool, bool, bool]:
    if scope == "base":
        return True, False, False
    if scope == "physics":
        return False, True, False
    if scope == "downstream":
        return False, False, True
    if scope == "all":
        return True, True, True
    raise ValueError(scope)


def load_targets(evidence_dir: Path, scope: str, seed: int) -> pd.DataFrame:
    run_base, run_physics, run_downstream = scope_flags(scope)
    requested = required_records(
        evidence_dir,
        run_base=run_base,
        run_physics=run_physics,
        run_downstream=run_downstream,
        seed=seed,
    )
    metadata_path = evidence_dir / "data" / "derived" / "sharp_metadata.csv.gz"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing SHARP metadata needed to form exact DRMS keys: {metadata_path}")
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata["harpnum"] = pd.to_numeric(metadata["HARPNUM"], errors="coerce")
    metadata["t_rec_utc"] = parse_jsoc_trec_to_utc(metadata["T_REC"])
    metadata = metadata.dropna(subset=["harpnum", "t_rec_utc"]).copy()
    metadata["harpnum"] = metadata["harpnum"].astype(int)
    requested["t_rec_utc"] = pd.to_datetime(requested["t_rec"], utc=True, errors="coerce")
    requested = requested.dropna(subset=["t_rec_utc"]).copy()
    keys = metadata[["harpnum", "t_rec_utc", "T_REC"]].drop_duplicates(
        ["harpnum", "t_rec_utc"]
    )
    out = requested.merge(keys, on=["harpnum", "t_rec_utc"], how="left", validate="many_to_one")
    missing = out["T_REC"].isna()
    if bool(missing.any()):
        examples = out.loc[missing, ["sample_id", "harpnum", "t_rec"]].head().to_dict("records")
        raise RuntimeError(
            f"{int(missing.sum())} planned samples have no exact T_REC in sharp_metadata; "
            f"examples={examples}"
        )
    out["tai_trec"] = out["T_REC"].map(normalize_trec)
    return out


def export_client(email: str):
    try:
        import drms
    except ImportError as exc:
        raise RuntimeError("Install the 'drms' package before JSOC acquisition") from exc
    if not email or "@" not in email:
        raise RuntimeError(
            "JSOC_EMAIL is missing. Register the student's email for JSOC exports, "
            "then set it in Colab with getpass without printing it."
        )
    # JSOC's checkAddress.sh currently advertises JSON but can return an empty
    # 200 response.  Do not let that compatibility probe prevent an otherwise
    # valid export request; JSOC validates the registered address at export.
    # Construct without email so drms does not eagerly call check_address().
    client = drms.Client()
    return client


def export_urls(client, recordset: str, email: str) -> pd.DataFrame:
    method = os.environ.get("JSOC_EXPORT_METHOD", "url_quick").strip()
    protocol = os.environ.get("JSOC_EXPORT_PROTOCOL", "as-is").strip()
    last = None
    for attempt in range(4):
      try:
        result = client.export(
            recordset,
            method=method,
            protocol=protocol,
            email=email,
        )
        urls = getattr(result, "urls", None)
        if urls is None:
            raise RuntimeError("JSOC returned no export URL table")
        urls = pd.DataFrame(urls).copy()
        if "url" not in urls.columns:
            raise RuntimeError(f"JSOC export URL table has no url column: {list(urls.columns)}")
        if "record" not in urls.columns:
            data = getattr(result, "data", None)
            if data is not None and "record" in data.columns and len(data) == len(urls):
                urls.insert(0, "record", data["record"].to_numpy())
        if "record" not in urls.columns:
            raise RuntimeError(f"JSOC export URL table has no record column: {list(urls.columns)}")
        return urls
      except Exception as exc:
        last = exc
        if attempt < 3:
            time.sleep(5 * (attempt + 1))
    try:
        raise last
    except Exception as exc:
        raise RuntimeError(
            f"JSOC export failed for {recordset}. Current transport is "
            f"method={method!r}, protocol={protocol!r}. If url_quick/as-is returns "
            "403, rerun with JSOC_EXPORT_METHOD=url and JSOC_EXPORT_PROTOCOL=fits."
        ) from exc


def normalize_download_url(value: object) -> str:
    url = str(value).strip()
    if url.startswith("/"):
        scheme = os.environ.get("JSOC_DOWNLOAD_SCHEME", "https").strip()
        return urljoin(f"{scheme}://jsoc.stanford.edu", url)
    return url


def download_fits(url: str, destination: Path, attempts: int = 4) -> int:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests before downloading JSOC FITS files") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "IRIS-ISEF-research/1.0 (registered JSOC export)"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with requests.get(url, headers=headers, timeout=(20, 60), stream=True, allow_redirects=True) as response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"transient HTTP {response.status_code}")
                response.raise_for_status()
                with part.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if not is_readable_fits(part):
                first = part.read_bytes()[:120]
                raise RuntimeError(
                    f"response is not a valid FITS payload ({part.stat().st_size} bytes; "
                    f"prefix={first!r})"
                )
            part.replace(destination)
            return int(destination.stat().st_size)
        except Exception as exc:
            last = exc
            try:
                part.unlink()
            except FileNotFoundError:
                pass
            if attempt + 1 < attempts:
                time.sleep(min(60.0, 3.0 * (2 ** attempt)))
    raise RuntimeError(f"download failed for {url}: {last}")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def finish_report(report: dict, output_dir: Path, scope: str) -> dict:
    report = dict(report)
    report["scope"] = scope
    (output_dir / "acquisition_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def acquire(evidence_dir: Path, output_dir: Path, scope: str, seed: int,
            email: str, max_harps: int | None = None,
            batch_size: int = 64, harp_min: int | None = None,
            harp_max: int | None = None, skip_verify: bool = False) -> dict:
    targets = load_targets(evidence_dir, scope, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "acquisition_log.jsonl"
    existing = {}
    try:
        existing = {
            str(k): v
            for k, v in {path.stem: path for path in output_dir.glob("*.fits")}.items()
            # Astropy-opening thousands of Drive files makes every resume spend
            # minutes before the first request.  Use the conservative FITS
            # header/size gate here; verify_local_cache(strict=True) performs
            # the full structural read exactly once before declaring PASS.
            if is_fits_payload(v)
        }
    except RuntimeError:
        raise
    remaining = targets[~targets.sample_id.astype(str).isin(existing)].copy()
    print(f"planned samples={len(targets):,}; already valid={len(targets)-len(remaining):,}; remaining={len(remaining):,}", flush=True)
    if remaining.empty:
        report_path = output_dir / "acquisition_report.json"
        return finish_report(verify_local_cache(
            evidence_dir, output_dir,
            run_base=scope_flags(scope)[0], run_physics=scope_flags(scope)[1],
            run_downstream=scope_flags(scope)[2], seed=seed,
            write_report=report_path,
        ), output_dir, scope)

    client = export_client(email)
    processed_harps = 0
    groups = list(remaining.groupby("harpnum", sort=True))
    groups = [(h, g) for h, g in groups
              if (harp_min is None or int(h) >= harp_min)
              and (harp_max is None or int(h) <= harp_max)]
    if max_harps is not None:
        groups = groups[:max_harps]
    work = pd.concat([group.sort_values("t_rec") for _, group in groups], ignore_index=True)
    for start_i in range(0, len(work), max(1, int(batch_size))):
        batch = work.iloc[start_i:start_i + max(1, int(batch_size))]
        downloaded = 0
        selectors = [f"hmi.sharp_cea_720s[{int(row.harpnum)}][{normalize_trec(row.tai_trec)}]{{magnetogram}}" for row in batch.itertuples(index=False)]
        recordset = ",".join(selectors)
        urls = export_urls(client, recordset, email)
        available = {}
        for _, item in urls.iterrows():
            key = record_key(item["record"])
            if key is None: continue
            if key in available: raise RuntimeError(f"Duplicate exported record key {key}")
            available[key] = normalize_download_url(item["url"])
        # JSOC may return only one HARP from a mixed-HARP url_quick request.
        # Retry omitted records in HARP-homogeneous requests before rejecting
        # the batch; the cache remains fail-closed.
        missing = [row for row in batch.itertuples(index=False)
                   if (int(row.harpnum), normalize_trec(row.tai_trec)) not in available]
        if missing:
            missing_df = pd.DataFrame([row._asdict() for row in missing])
            for harp, subset in missing_df.groupby("harpnum", sort=True):
                retry_selectors = [
                    f"hmi.sharp_cea_720s[{int(row.harpnum)}][{normalize_trec(row.tai_trec)}]{{magnetogram}}"
                    for row in subset.itertuples(index=False)
                ]
                retry_urls = export_urls(client, ",".join(retry_selectors), email)
                for _, item in retry_urls.iterrows():
                    key = record_key(item["record"])
                    if key is not None:
                        available[key] = normalize_download_url(item["url"])
        jobs = {}
        with ThreadPoolExecutor(max_workers=min(16, len(batch))) as pool:
            for row in batch.itertuples(index=False):
                key = (int(row.harpnum), normalize_trec(row.tai_trec))
                url = available.get(key)
                if url is None:
                    raise RuntimeError(
                        f"JSOC export omitted planned record sample_id={row.sample_id}, key={key}; "
                        f"returned_records={len(available)}. Refusing a partial cache."
                    )
                destination = output_dir / f"{row.sample_id}.fits"
                if destination.is_file() and is_fits_payload(destination): continue
                jobs[pool.submit(download_fits, url, destination)] = row
            for future in as_completed(jobs):
                row = jobs[future]
                url = available[(int(row.harpnum), normalize_trec(row.tai_trec))]
                size = future.result()
                append_jsonl(log_path, {"sample_id": str(row.sample_id), "harpnum": int(row.harpnum), "t_rec_tai": str(row.tai_trec), "url": url, "bytes": size, "recordset": recordset, "method": os.environ.get("JSOC_EXPORT_METHOD", "url_quick"), "protocol": os.environ.get("JSOC_EXPORT_PROTOCOL", "as-is")})
                downloaded += 1
        processed_harps += batch.harpnum.nunique()
        print(f"batch: requested={len(batch):,}; downloaded={downloaded:,}; processed_harps={processed_harps}", flush=True)

    if max_harps is not None:
        return {"status": "PARTIAL_SMOKE_ONLY", "processed_harps": processed_harps}
    if skip_verify:
        return {"status": "PARTIAL_WORKER", "processed_harps": processed_harps}
    run_base, run_physics, run_downstream = scope_flags(scope)
    return finish_report(verify_local_cache(
        evidence_dir, output_dir,
        run_base=run_base, run_physics=run_physics,
        run_downstream=run_downstream, seed=seed, strict=True,
        write_report=output_dir / "acquisition_report.json",
    ), output_dir, scope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["base", "physics", "downstream", "all"], default="base")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-harps", type=int, default=None, help="Only for non-scientific smoke testing")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--harp-min", type=int, default=None)
    parser.add_argument("--harp-max", type=int, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    email = os.environ.get("JSOC_EMAIL", "").strip()
    try:
        report = acquire(
            Path(args.evidence_dir), Path(args.output_dir), args.scope,
            args.seed, email, args.max_harps, args.batch_size,
            args.harp_min, args.harp_max, args.skip_verify,
        )
    except Exception as exc:
        print(f"ACQUISITION_FAILED: {exc}", file=sys.stderr, flush=True)
        raise
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
