from __future__ import annotations

from iris_report.iris_sep.tools.run_sgps_l1b_live_binding_v1 import (
    bands_match,
    derive_archive_energy_bands,
    identify_archive_semantics,
    parse_live_energy_band,
    parse_month_netcdf_links,
    summarize_live,
)


def test_month_links_are_sorted_netcdf_only():
    html = '''
    <a href="b.nc">b</a><a href="a.nc">a</a><a href="note.txt">note</a>
    '''
    assert parse_month_netcdf_links(html) == ["a.nc", "b.nc"]


def test_parse_live_energy_band_units():
    assert parse_live_energy_band("1020-1860 keV") == [1.02, 1.86]
    assert parse_live_energy_band("1.02-1.86 MeV") == [1.02, 1.86]


def test_derive_bounds_from_13x2_keV_array():
    vals = [[1000 + i * 100, 1100 + i * 100] for i in range(13)]
    inv = {
        "dimensions": {"energy": 13, "bound": 2},
        "variables": {
            "DiffProtonEnergyBounds": {
                "dimensions": ["energy", "bound"],
                "shape": [13, 2],
                "units": "keV",
                "long_name": "Differential proton channel energy bounds",
                "standard_name": None,
                "description": None,
                "values": vals,
            }
        },
    }
    bands = derive_archive_energy_bands(inv, ["DiffProtonEnergyBounds"])
    assert len(bands) == 13
    assert bands[0] == [1.0, 1.1]


def test_archive_semantics_reject_extra_direction_dimension():
    inv = {
        "dimensions": {"time": 100, "energy": 13, "sensor": 2},
        "variables": {
            "DiffProtonFlux": {
                "dimensions": ["time", "sensor", "energy"],
                "shape": [100, 2, 13],
                "units": "protons/(cm2 sr keV s)",
                "long_name": "Differential proton flux",
                "standard_name": None,
                "description": None,
            },
            "DiffProtonEnergyBounds": {
                "dimensions": ["energy", "bound"],
                "shape": [13, 2],
                "units": "MeV",
                "long_name": "Differential proton channel energy bounds",
                "standard_name": None,
                "description": None,
                "values": [[1 + i, 2 + i] for i in range(13)],
            },
            "time": {
                "dimensions": ["time"], "shape": [100], "units": "seconds since 2000-01-01",
                "long_name": "time", "standard_name": "time", "description": None,
            },
            "quality_flag": {
                "dimensions": ["time"], "shape": [100], "units": None,
                "long_name": "proton data quality flag", "standard_name": None, "description": None,
            },
        },
    }
    result = identify_archive_semantics(inv)
    assert result["has_unambiguous_13_channel_flux"] is False
    assert result["non_time_non_channel_dimensions"] == [{"dimension": "sensor", "size": 2}]


def test_live_summary_requires_thirteen_numeric_bands():
    rows = []
    for i in range(13):
        rows.append({
            "time_tag": "2026-09-12T00:00:00Z",
            "flux": 1.0,
            "energy": f"{1+i}-{2+i} MeV",
            "channel": f"P{i+1}",
        })
    result = summarize_live(rows)
    assert result["passed"] is True
    assert len(result["parsed_energy_bands"]) == 13


def test_band_match_is_fail_closed():
    a = [[float(i + 1), float(i + 2)] for i in range(13)]
    b = [[float(i + 1), float(i + 2)] for i in range(13)]
    assert bands_match(a, b, 0.02)["passed"] is True
    b[4][1] *= 1.2
    assert bands_match(a, b, 0.02)["passed"] is False
