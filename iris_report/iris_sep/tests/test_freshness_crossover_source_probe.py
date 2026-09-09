from __future__ import annotations

from pathlib import Path

import pytest

from iris_report.iris_sep.tools.run_freshness_crossover_source_probe import (
    SourceProbeError,
    inspect_omni_5min_ascii,
    parse_xrs_candidates,
)


def _omni_row(year: int, doy: int, hour: int, minute: int, gt10: float, gt30: float, gt60: float) -> str:
    # The parser only relies on the first four timestamp fields, a stable row
    # width, and the final three documented GOES integral-flux fields.
    middle = ["0"] * 42
    return " ".join([str(year), str(doy), str(hour), str(minute), *middle, f"{gt10:.2f}", f"{gt30:.2f}", f"{gt60:.2f}"])


def test_omni_parser_requires_consistent_year_and_integral_columns() -> None:
    body = (\
        _omni_row(2014, 1, 0, 0, 0.12, 0.03, 0.01) + "\n" +
        _omni_row(2014, 365, 23, 55, 5.00, 0.80, 0.20) + "\n"
    ).encode("ascii")
    summary = inspect_omni_5min_ascii(body, expected_year=2014, fill_value=99999.99)
    assert summary["rows"] == 2
    assert summary["nonfill_gt10_rows"] == 2
    assert summary["negative_gt10_rows"] == 0
    assert summary["appended_columns"] == ["gt10", "gt30", "gt60"]


def test_omni_parser_rejects_negative_nonfill_flux() -> None:
    body = (_omni_row(2014, 1, 0, 0, -1.0, 0.0, 0.0) + "\n").encode("ascii")
    with pytest.raises(SourceProbeError):
        inspect_omni_5min_ascii(body, expected_year=2014, fill_value=99999.99)


def test_omni_fill_value_is_not_counted_as_observed() -> None:
    body = (_omni_row(2014, 1, 0, 0, 99999.99, 99999.99, 99999.99) + "\n").encode("ascii")
    with pytest.raises(SourceProbeError):
        inspect_omni_5min_ascii(body, expected_year=2014, fill_value=99999.99)


def test_xrs_directory_parser_selects_only_goes15_xrs_1m_netcdf() -> None:
    html = """
    <a href="g15_xrs_1m_20140101_20140131.nc">x</a>
    <a href="g15_xrs_1m_20140101_20140131.csv">x</a>
    <a href="g14_xrs_1m_20140101_20140131.nc">x</a>
    <a href="g15_epead_cpflux_5m_20140101_20140131.nc">x</a>
    """
    assert parse_xrs_candidates(html, satellite=15, preferred_extension=".nc") == [
        "g15_xrs_1m_20140101_20140131.nc"
    ]


def test_xrs_directory_parser_does_not_accept_science_or_other_satellite_by_token_accident() -> None:
    html = """
    <a href="g14_xrs_1m_20170101_20170131.nc">x</a>
    <a href="g15_xrs_1m_20170101_20170131_science_v2.nc">x</a>
    """
    candidates = parse_xrs_candidates(html, satellite=15, preferred_extension=".nc")
    # The actual online probe still checks the frozen directory/product metadata;
    # this unit test documents that filename token matching alone is not a skill gate.
    assert "g14_xrs_1m_20170101_20170131.nc" not in candidates
