from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jsoc_time import parse_jsoc_trec_to_utc


def main() -> None:
    values = pd.Series([
        "2012.06.30_23:59:25_TAI",
        "2015.07.01_00:00:00_TAI",
        "2017.01.01_00:00:00_TAI",
    ])
    got = parse_jsoc_trec_to_utc(values)
    expected = pd.to_datetime([
        "2012-06-30T23:58:51Z",
        "2015-06-30T23:59:25Z",
        "2016-12-31T23:59:24Z",
    ], utc=True)
    expected_series = pd.Series(expected)
    # Pandas may retain nanosecond or microsecond datetime storage depending on
    # the installed version. Compare instants after normalizing precision.
    if not got.astype("datetime64[us, UTC]").equals(
        expected_series.astype("datetime64[us, UTC]")
    ):
        raise AssertionError(f"TAI conversion mismatch:\n{got}\n!=\n{expected_series}")

    try:
        parse_jsoc_trec_to_utc(pd.Series(["2017-01-01T00:00:00Z"]))
    except ValueError:
        pass
    else:
        raise AssertionError("non-TAI values must fail in strict mode")

    mixed = parse_jsoc_trec_to_utc(
        pd.Series(["2017-01-01T00:00:00Z", "not-a-time"]), require_tai=False
    )
    if mixed.iloc[0] != pd.Timestamp("2017-01-01T00:00:00Z") or not pd.isna(mixed.iloc[1]):
        raise AssertionError(f"non-strict UTC fallback mismatch: {mixed}")
    print("JSOC TAI-to-UTC self-test PASS")


if __name__ == "__main__":
    main()
