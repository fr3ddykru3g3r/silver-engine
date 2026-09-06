#!/usr/bin/env python3
"""Repair a historical manifest created with the wrong JSOC time scale.

The integrity-locked historical artifact contains true JSOC ``T_REC`` values in
``sharp_metadata.csv.gz`` but an older manifest whose ``t_rec`` field treated
the TAI text as UTC.  This script creates a corrected manifest in place while
retaining the original sample identifier and source TAI key for provenance.

It is deliberately fail-closed: the metadata-to-manifest mapping must be
one-to-one, exact chronology must preserve the frozen connected-region
partitions, and primary labels are recomputed from the resolved event table
using the locked ``(t, t+24 h]`` rule plus global censoring of unresolved events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COMMON = Path(__file__).resolve().parents[1] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from jsoc_time import parse_jsoc_trec_to_utc


PRIMARY = {"train", "validation", "test"}
DAY24_NS = int(pd.Timedelta(hours=24).value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_utc_from_tai(values: pd.Series) -> pd.Series:
    """Reproduce only the old identifier convention, never a science timestamp."""
    text = (
        values.astype("string")
        .str.replace("_TAI", "", regex=False)
        .str.replace(".", "-", n=2, regex=False)
        .str.replace("_", " ", regex=False)
    )
    return pd.to_datetime(text, utc=True, errors="raise")


def canonical_sample_id(harpnums: pd.Series, times: pd.Series) -> pd.Series:
    stamps = times.dt.strftime("%Y%m%dT%H%M%S")
    fractional = times.dt.microsecond.astype(str).str.zfill(6).str.rstrip("0")
    stamps = stamps.where(fractional.eq(""), stamps + "." + fractional)
    return "H" + pd.to_numeric(harpnums, errors="raise").astype(int).astype(str) + "_" + stamps + "Z"


def quality_zero(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.isin({"0", "0x00000000"})


def chronology_partitions(rows: pd.DataFrame, time_col: str) -> tuple[dict[str, str], pd.Timestamp, pd.Timestamp]:
    eligible = rows[
        quality_zero(rows.quality)
        & pd.to_numeric(rows.cmd_deg, errors="coerce").abs().le(30.0)
        & rows.region_group_id.notna()
    ].copy()
    spans = (
        eligible.groupby("region_group_id")[time_col]
        .agg(start="min", end="max")
        .reset_index()
        .sort_values(["start", "region_group_id"])
        .reset_index(drop=True)
    )
    if len(spans) < 10:
        raise RuntimeError(f"Too few eligible connected groups: {len(spans)}")

    n = len(spans)
    i1 = max(1, min(n - 2, round(n * 0.6)))
    i2 = max(i1 + 1, min(n - 1, round(n * 0.8)))
    boundary_1 = spans.loc[i1, "start"]
    boundary_2 = spans.loc[i2, "start"]
    buffer = pd.Timedelta(hours=36)
    parts: dict[str, str] = {}
    for row in spans.itertuples(index=False):
        touches_1 = row.end >= boundary_1 - buffer and row.start <= boundary_1 + buffer
        touches_2 = row.end >= boundary_2 - buffer and row.start <= boundary_2 + buffer
        if touches_1 or touches_2:
            part = "excluded"
        elif row.end < boundary_1 - buffer:
            part = "train"
        elif row.start > boundary_1 + buffer and row.end < boundary_2 - buffer:
            part = "validation"
        elif row.start > boundary_2 + buffer:
            part = "test"
        else:
            part = "excluded"
        parts[str(row.region_group_id)] = part
    return parts, boundary_1, boundary_2


def parse_noaa_values(value: object) -> list[int]:
    if pd.isna(value):
        return []
    result = []
    for token in str(value).split(";"):
        if not token or token.lower() == "nan":
            continue
        try:
            result.append(int(float(token)))
        except ValueError as exc:
            raise ValueError(f"Invalid NOAA AR token {token!r}") from exc
    return result


def event_index(events: pd.DataFrame) -> tuple[dict[int, np.ndarray], np.ndarray]:
    events = events.copy()
    events["event_start"] = pd.to_datetime(events.event_start, utc=True, errors="raise")
    resolved: defaultdict[int, list[int]] = defaultdict(list)
    unresolved: list[int] = []
    for event in events.itertuples(index=False):
        if pd.isna(event.canonical_noaa_ar):
            unresolved.append(int(event.event_start.value))
        else:
            resolved[int(event.canonical_noaa_ar)].append(int(event.event_start.value))
    resolved_arrays = {key: np.sort(np.asarray(value, dtype="int64")) for key, value in resolved.items()}
    return resolved_arrays, np.sort(np.asarray(unresolved, dtype="int64"))


def event_in_window(starts: np.ndarray, timestamp_ns: int) -> bool:
    index = int(np.searchsorted(starts, timestamp_ns, side="right"))
    return index < len(starts) and int(starts[index]) <= timestamp_ns + DAY24_NS


def recompute_primary_labels(manifest: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    resolved, unresolved = event_index(events)
    labels: list[float] = []
    statuses: list[str] = []
    for row in manifest.itertuples(index=False):
        if str(row.partition) not in PRIMARY:
            labels.append(row.label_m1plus_24h)
            statuses.append(row.label_integrity_status)
            continue

        timestamp_ns = int(row.exact_time.value)
        positive = any(
            event_in_window(resolved.get(noaa, np.empty(0, dtype="int64")), timestamp_ns)
            for noaa in parse_noaa_values(row.noaa_ars)
        )
        if positive:
            labels.append(1.0)
            statuses.append("RESOLVED_OR_CLEAN")
        elif event_in_window(unresolved, timestamp_ns):
            labels.append(np.nan)
            statuses.append("CENSORED_UNRESOLVED_GLOBAL")
        else:
            labels.append(0.0)
            statuses.append("RESOLVED_OR_CLEAN")
    return pd.Series(labels, index=manifest.index, dtype="float64"), pd.Series(statuses, index=manifest.index, dtype="string")


def same_label(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.isna() & right.isna()) | left.eq(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()

    root = Path(args.evidence_dir)
    derived = root / "data" / "derived"
    manifest_path = derived / "training_manifest.csv.gz"
    metadata_path = derived / "sharp_metadata.csv.gz"
    events_path = derived / "resolved_m1plus_events.csv"
    receipt_path = derived / "tai_repair_audit.json"
    required = [manifest_path, metadata_path, events_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evidence files: " + ", ".join(missing))
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "PASS":
            print(json.dumps({"status": "ALREADY_REPAIRED", "receipt": str(receipt_path)}, indent=2))
            return

    manifest_sha_before = sha256_file(manifest_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    events = pd.read_csv(events_path, low_memory=False)
    required_manifest = {"sample_id", "t_rec", "harpnum", "region_group_id", "partition", "quality", "cmd_deg", "noaa_ars", "label_m1plus_24h", "label_integrity_status"}
    missing_columns = required_manifest - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")
    if not manifest.sample_id.astype(str).is_unique:
        raise ValueError("Manifest sample_id is not unique")
    metadata = metadata.copy()
    metadata["harpnum_int"] = pd.to_numeric(metadata.HARPNUM, errors="raise").astype(int)
    if metadata.duplicated(["harpnum_int", "T_REC"]).any():
        raise ValueError("Metadata (HARPNUM, T_REC) key is not unique")
    metadata["exact_time"] = parse_jsoc_trec_to_utc(metadata.T_REC)
    metadata["legacy_time"] = legacy_utc_from_tai(metadata.T_REC)
    metadata["legacy_sample_id"] = "H" + metadata.harpnum_int.astype(str) + "_" + metadata.legacy_time.dt.strftime("%Y%m%dT%H%M%SZ")
    if not metadata.legacy_sample_id.is_unique:
        raise ValueError("Metadata legacy sample identifier is not unique")
    if not metadata.exact_time.notna().all():
        raise ValueError("Metadata contains an invalid exact timestamp")

    joined = manifest.merge(
        metadata[["legacy_sample_id", "T_REC", "exact_time"]].rename(
            columns={"legacy_sample_id": "metadata_legacy_sample_id"}
        ),
        left_on="sample_id",
        right_on="metadata_legacy_sample_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined.exact_time.notna().all():
        bad = joined.loc[joined.exact_time.isna(), "sample_id"].head(5).tolist()
        raise RuntimeError(f"Could not map manifest rows to exact TAI metadata; examples: {bad}")
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("Manifest-to-metadata mapping was not complete")

    old_time = pd.to_datetime(joined.t_rec, utc=True, errors="raise")
    joined["old_time"] = old_time
    joined["exact_time"] = pd.to_datetime(joined.exact_time, utc=True, errors="raise")
    delta_seconds = (joined.exact_time - joined.old_time).dt.total_seconds()
    if not delta_seconds.isin([-34.0, -35.0, -36.0, -37.0]).all():
        raise ValueError("Unexpected timestamp correction magnitude")

    old_parts, old_b1, old_b2 = chronology_partitions(joined, "old_time")
    exact_parts, exact_b1, exact_b2 = chronology_partitions(joined, "exact_time")
    if old_parts != exact_parts:
        changes = [key for key in sorted(old_parts) if old_parts.get(key) != exact_parts.get(key)]
        raise RuntimeError(f"Exact TAI conversion changes frozen group partitions: {changes[:10]}")
    eligible = joined[quality_zero(joined.quality) & pd.to_numeric(joined.cmd_deg, errors="coerce").abs().le(30.0) & joined.region_group_id.notna()]
    current_parts = eligible.groupby("region_group_id").partition.agg(lambda values: set(values.astype(str)))
    inconsistent = {str(key): sorted(value) for key, value in current_parts.items() if len(value) != 1 or next(iter(value)) != exact_parts[str(key)]}
    if inconsistent:
        raise RuntimeError(f"Existing partition labels disagree with exact chronology: {dict(list(inconsistent.items())[:10])}")

    original_ids = joined.sample_id.astype(str).copy()
    joined["legacy_sample_id"] = original_ids
    joined["t_rec_tai"] = joined.T_REC.astype(str)
    joined["t_rec"] = joined.exact_time.map(lambda value: value.isoformat())
    joined["sample_id"] = canonical_sample_id(joined.harpnum, joined.exact_time)
    if not joined.sample_id.is_unique:
        raise ValueError("Canonical sample_id is not unique")
    joined["exact_time"] = pd.to_datetime(joined.exact_time, utc=True)
    new_labels, new_status = recompute_primary_labels(joined, events)
    label_changes = int((~same_label(joined.label_m1plus_24h, new_labels)).sum())
    joined["label_m1plus_24h"] = new_labels
    joined["label_integrity_status"] = new_status
    hour_key_changes = int(
        (joined.exact_time.dt.floor("h") != joined.old_time.dt.floor("h")).sum()
    )
    joined = joined.drop(
        columns=["metadata_legacy_sample_id", "_merge", "old_time", "exact_time", "T_REC"],
        errors="ignore",
    )
    # ``legacy_sample_id`` and ``t_rec_tai`` remain as row-aligned provenance
    # fields while the model-facing columns retain their original names.
    joined = joined.sort_values(["t_rec", "region_group_id", "sample_id"]).reset_index(drop=True)
    joined.to_csv(manifest_path, index=False, compression="gzip")

    primary_all = joined[joined.partition.isin(PRIMARY)].copy()
    primary = primary_all[primary_all.label_m1plus_24h.notna()].copy()
    url_columns = ["sample_id", "magnetogram_url", "label_m1plus_24h", "partition", "region_group_id", "harpnum", "t_rec", "noaa_ars", "cmd_deg"]
    for name, frame in [("image_urls_all_splits.csv.gz", primary), ("training_image_urls.csv.gz", primary[primary.partition.eq("train")]), ("validation_image_urls.csv.gz", primary[primary.partition.eq("validation")]), ("test_image_urls.csv.gz", primary[primary.partition.eq("test")])]:
        frame[url_columns].to_csv(derived / name, index=False, compression="gzip")

    counts = []
    for partition in ["train", "validation", "test"]:
        frame = primary[primary.partition.eq(partition)]
        positive = frame[frame.label_m1plus_24h.eq(1)]
        counts.append({
            "partition": partition,
            "rows": len(frame),
            "positive_rows": len(positive),
            "independent_groups": int(frame.region_group_id.nunique()),
            "independent_positive_groups": int(positive.region_group_id.nunique()),
            "independent_harps": int(frame.harpnum.nunique()),
            "independent_positive_harps": int(positive.harpnum.nunique()),
            "image_urls": int(frame.magnetogram_url.notna().sum()),
            "censored_negative_rows": int(primary_all.loc[primary_all.partition.eq(partition), "label_integrity_status"].eq("CENSORED_UNRESOLVED_GLOBAL").sum()),
        })
    pd.DataFrame(counts).to_csv(derived / "independent_positive_region_counts.csv", index=False)

    split = {
        "boundary_1": exact_b1.isoformat(),
        "boundary_2": exact_b2.isoformat(),
        "method": "connected HARP-NOAA groups, exact JSOC TAI converted to UTC, chronology 60/20/20, 36h buffer",
        "parts": exact_parts,
    }
    split["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (derived / "frozen_split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")

    audit = json.loads((derived / "manifest_audit.json").read_text()) if (derived / "manifest_audit.json").exists() else {}
    audit.update({
        "time_scale": "T_REC parsed as TAI and converted to UTC with Astropy",
        "primary_rows_after_censoring": len(primary),
        "partitions": counts,
        "split_sha256": split["sha256"],
    })
    (derived / "manifest_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    label_audit_path = derived / "label_integrity_audit.json"
    label_audit = json.loads(label_audit_path.read_text()) if label_audit_path.exists() else {}
    label_audit.update({
        "rule": "global censor of otherwise-negative samples with an unresolved >=M1 flare onset in (t,t+24h]",
        "unresolved_events": int(events.canonical_noaa_ar.isna().sum()),
        "rows_censored_total": int(primary_all.label_integrity_status.eq("CENSORED_UNRESOLVED_GLOBAL").sum()),
        "primary_rows_after_censoring": len(primary),
        "partitions_after_censoring": counts,
        "split_sha256": split["sha256"],
        "reason": "prevents unattributed major flares from being silently converted into false-negative training/evaluation labels",
    })
    label_audit_path.write_text(json.dumps(label_audit, indent=2, default=str) + "\n")

    manifest_sha_after = sha256_file(manifest_path)
    receipt = {
        "status": "PASS",
        "method": "canonical JSOC TAI-to-UTC repair from sharp_metadata.T_REC",
        "input_manifest_sha256": manifest_sha_before,
        "output_manifest_sha256": manifest_sha_after,
        "metadata_sha256": sha256_file(metadata_path),
        "event_sha256": sha256_file(events_path),
        "rows": len(joined),
        "timestamp_delta_seconds": {str(key): int(value) for key, value in delta_seconds.value_counts().sort_index().items()},
        "hour_key_changes": hour_key_changes,
        "label_changes_primary": label_changes,
        "chronology_groups": len(exact_parts),
        "group_partition_changes": 0,
        "old_boundary_1": old_b1.isoformat(),
        "old_boundary_2": old_b2.isoformat(),
        "exact_boundary_1": exact_b1.isoformat(),
        "exact_boundary_2": exact_b2.isoformat(),
        "split_sha256": split["sha256"],
        "counts": counts,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
