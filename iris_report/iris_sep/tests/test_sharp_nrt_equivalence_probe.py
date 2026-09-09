from __future__ import annotations

import pandas as pd

from iris_report.iris_sep.tools.run_sharp_nrt_equivalence_probe import (
    KEYWORDS,
    MIN_MATCHED_FINITE_ROWS,
    _comparison,
    _finite_counts,
    _recordset,
)


def _frame(*, offset=0.0, rows=24):
    data = {
        "HARPNUM": [100 + (i // 12) for i in range(rows)],
        "T_REC": [f"2024.05.10_{i:02d}:00:00_TAI" for i in range(rows)],
        "CMASKL": [1000 + i for i in range(rows)],
        "MEANGBL": [10.0 + i * 0.01 + offset for i in range(rows)],
        "USFLUXL": [1.0e21 + i * 1.0e18 + offset for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_recordset_is_fixed_series_plus_bounded_time_selector():
    assert _recordset("hmi.sharp_720s", "2024.05.10_00:00:00_TAI", "1d") == (
        "hmi.sharp_720s[][2024.05.10_00:00:00_TAI/1d]"
    )


def test_identical_definitive_keyword_rows_pass_equivalence_gate():
    frame = _frame(rows=max(24, MIN_MATCHED_FINITE_ROWS))
    result = _comparison(frame, frame.copy())
    assert result["passed"] is True
    for keyword in KEYWORDS:
        assert result["keyword_results"][keyword]["passed"] is True
        assert result["keyword_results"][keyword]["max_absolute_difference"] == 0.0


def test_numeric_semantic_mismatch_fails_instead_of_being_rounded_away():
    ccd = _frame(rows=max(24, MIN_MATCHED_FINITE_ROWS))
    cea = ccd.copy()
    cea.loc[0, "MEANGBL"] += 0.1
    result = _comparison(ccd, cea)
    assert result["passed"] is False
    assert result["keyword_results"]["MEANGBL"]["passed"] is False


def test_cmaskl_must_match_exactly():
    ccd = _frame(rows=max(24, MIN_MATCHED_FINITE_ROWS))
    cea = ccd.copy()
    cea.loc[0, "CMASKL"] += 1
    result = _comparison(ccd, cea)
    assert result["passed"] is False
    assert result["keyword_results"]["CMASKL"]["passed"] is False


def test_too_few_matched_rows_fails_even_when_values_are_identical():
    frame = _frame(rows=MIN_MATCHED_FINITE_ROWS - 1)
    result = _comparison(frame, frame.copy())
    assert result["passed"] is False


def test_nonfinite_recent_values_are_counted_not_silently_filled():
    frame = _frame()
    frame.loc[0, "USFLUXL"] = float("nan")
    counts = _finite_counts(frame)
    assert counts["USFLUXL"] == len(frame) - 1
    assert counts["CMASKL"] == len(frame)
