#!/usr/bin/env python3
"""Fast unit tests for the historical TAI-manifest repair primitives."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("repair_historical_tai_manifest.py")
SPEC = importlib.util.spec_from_file_location("tai_repair", MODULE_PATH)
assert SPEC and SPEC.loader
tai_repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tai_repair)


def main() -> None:
    tai = pd.Series(["2016.12.31_23:59:36_TAI"])
    legacy = tai_repair.legacy_utc_from_tai(tai).iloc[0]
    assert str(legacy) == "2016-12-31 23:59:36+00:00"

    times = pd.Series(pd.to_datetime(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00.120000Z"],
        utc=True,
        format="mixed",
    ))
    ids = tai_repair.canonical_sample_id(pd.Series([42, 42]), times)
    assert ids.tolist() == ["H42_20200101T000000Z", "H42_20200101T000000.12Z"]

    start = pd.Timestamp("2020-01-01T00:00:00Z").value
    end = start + int(pd.Timedelta(hours=24).value)
    assert not tai_repair.event_in_window(np.asarray([start], dtype="int64"), start)
    assert tai_repair.event_in_window(np.asarray([end], dtype="int64"), start)

    manifest = pd.DataFrame([
        {
            "partition": "train",
            "exact_time": pd.Timestamp("2020-01-01T00:00:00Z"),
            "noaa_ars": "1234",
            "label_m1plus_24h": 1.0,
            "label_integrity_status": "RESOLVED_OR_CLEAN",
        },
        {
            "partition": "validation",
            "exact_time": pd.Timestamp("2020-01-01T00:00:00Z"),
            "noaa_ars": "2345",
            "label_m1plus_24h": 0.0,
            "label_integrity_status": "RESOLVED_OR_CLEAN",
        },
        {
            "partition": "test",
            "exact_time": pd.Timestamp("2020-01-03T00:00:00Z"),
            "noaa_ars": "9999",
            "label_m1plus_24h": 0.0,
            "label_integrity_status": "RESOLVED_OR_CLEAN",
        },
        {
            "partition": "excluded",
            "exact_time": pd.Timestamp("2020-01-01T00:00:00Z"),
            "noaa_ars": "1234",
            "label_m1plus_24h": np.nan,
            "label_integrity_status": "EXCLUDED_BUFFER",
        },
    ])
    events = pd.DataFrame([
        {"event_start": "2020-01-01T00:00:00Z", "canonical_noaa_ar": 1234},
        {"event_start": "2020-01-02T00:00:00Z", "canonical_noaa_ar": 2345},
        {"event_start": "2020-01-03T12:00:00Z", "canonical_noaa_ar": np.nan},
    ])
    labels, statuses = tai_repair.recompute_primary_labels(manifest, events)
    assert labels.iloc[0] == 0.0 and labels.iloc[1] == 1.0
    assert statuses.tolist()[:3] == ["RESOLVED_OR_CLEAN", "RESOLVED_OR_CLEAN", "CENSORED_UNRESOLVED_GLOBAL"]
    assert pd.isna(labels.iloc[2])
    assert pd.isna(labels.iloc[3]) and statuses.iloc[3] == "EXCLUDED_BUFFER"
    print("TAI repair primitive self-test PASS")


if __name__ == "__main__":
    main()
