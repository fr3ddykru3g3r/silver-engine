from __future__ import annotations

from iris_report.iris_sep.tools.run_noaa_reduced_input_source_probe_v2 import (
    event_catalog_semantics,
    parse_operational_daily_links,
    post_event_report_semantics,
    sgps_differential_semantics,
    xrs_semantics,
)


def test_month_parser_uses_operational_files_only():
    html = '''
    <a href="dn_sgps-l2-avg1m_g16_d20240501_v2-0-0.nc">one</a>
    <a href="dn_sgps-l2-avg1m_g16_d20240503_v2-0-0.nc">two</a>
    <a href="sci_sgps-l2-avg1m_g16_d20240502_v2-0-0.nc">science</a>
    <a href="dn_sgps-l2-avg1m_g18_d20240501_v2-0-0.nc">other-satellite</a>
    <a href="dn_sgps-l2-avg1m_g16_d20240601_v2-0-0.nc">other-month</a>
    '''
    assert parse_operational_daily_links(html, "sgps-l2-avg1m", 16, "2024-05") == [
        "dn_sgps-l2-avg1m_g16_d20240501_v2-0-0.nc",
        "dn_sgps-l2-avg1m_g16_d20240503_v2-0-0.nc",
    ]


def test_sgps_v2_accepts_documented_differential_channels_without_gt10_integral():
    inventory = {
        "variables": {
            "AvgDiffProtonFlux": {
                "long_name": "Time-averaged proton fluxes in several differential channels between 1 and 500 MeV.",
                "standard_name": None,
                "units": "protons/(cm^2 sr keV s)",
            },
            "DiffProtonLowerEnergy": {
                "long_name": "Lower band energies for the thirteen SGPS proton differential channels",
                "standard_name": None,
                "units": "keV",
            },
            "DiffProtonUpperEnergy": {
                "long_name": "Upper band energies for the thirteen SGPS proton differential channels",
                "standard_name": None,
                "units": "keV",
            },
            "DiffValidL1bSamplesInAvg": {
                "long_name": "Number of valid L1b samples in each differential flux average",
                "standard_name": None,
                "units": None,
            },
            "L2_SciData_TimeStamp": {
                "long_name": "Time stamp at the start of the averaging period",
                "standard_name": None,
                "units": "seconds since 2000-01-01 12:00:00",
            },
            "AvgIntProtonFlux": {
                "long_name": "Time-averaged proton fluxes in the P11 >500 MeV integral channel",
                "standard_name": None,
                "units": "protons/(cm^2 sr s)",
            },
        }
    }
    result = sgps_differential_semantics(inventory)
    assert result["passed"] is True
    assert result["direct_integral_gt10_candidates"] == []
    assert "AvgDiffProtonFlux" in result["differential_flux"]


def test_sgps_v2_rejects_a_direct_gt10_integral_candidate_in_the_bound_interface():
    inventory = {
        "variables": {
            "AvgDiffProtonFlux": {
                "long_name": "Differential proton flux 1 to 500 MeV",
                "standard_name": None,
                "units": "protons/(cm^2 sr keV s)",
            },
            "DiffProtonEffectiveEnergy": {
                "long_name": "Effective energy for differential proton channels",
                "standard_name": None,
                "units": "keV",
            },
            "time": {
                "long_name": "time",
                "standard_name": "time",
                "units": "seconds since 2000-01-01 00:00:00",
            },
            "quality_flag": {
                "long_name": "data quality flag",
                "standard_name": None,
                "units": None,
            },
            "integral_gt10": {
                "long_name": "Integral proton flux >10 MeV",
                "standard_name": None,
                "units": "pfu",
            },
        }
    }
    result = sgps_differential_semantics(inventory)
    assert result["passed"] is False
    assert result["direct_integral_gt10_candidates"] == ["integral_gt10"]


def test_xrs_requires_both_bands_time_and_quality():
    inventory = {
        "variables": {
            "xrsa_flux": {"long_name": "XRS-A irradiance flux", "standard_name": None, "units": "W/m2"},
            "xrsb_flux": {"long_name": "XRS-B irradiance flux", "standard_name": None, "units": "W/m2"},
            "time": {"long_name": "time", "standard_name": "time", "units": "seconds since 2000-01-01"},
            "quality_flag": {"long_name": "XRS quality flag", "standard_name": None, "units": None},
        }
    }
    assert xrs_semantics(inventory)["passed"] is True


def test_catalog_semantics_require_start_and_end_definition_language():
    html = '''
    <h1>Solar Proton Events Affecting the Earth Environment</h1>
    Begin Time | Maximum Time | &gt;10 MeV Maximum
    Please Note: SESC defines the start of a proton event to be the first of 3 consecutive
    data points with fluxes greater than or equal to 10 pfu. The end of an event is the last
    time the flux was greater than or equal to 10 pfu.
    '''
    assert event_catalog_semantics(html)["passed"] is True


def test_post_event_report_requires_proton_start_and_end_language():
    text = "A 10 MeV proton event began on 10 May at 1335 and ended at 12/1235."
    assert post_event_report_semantics(text)["passed"] is True
    assert post_event_report_semantics("No proton events were observed at geosynchronous orbit.")["passed"] is False
