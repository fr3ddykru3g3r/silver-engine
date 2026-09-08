from __future__ import annotations

from iris_report.iris_sep.tools.run_noaa_reduced_input_source_probe import (
    identify_proton_integral_candidates,
    identify_xrs_candidates,
    parse_daily_file_links,
    quality_variables,
    time_variables,
)


def meta(*, long_name=None, standard_name=None, description=None, units=None):
    return {
        "long_name": long_name,
        "standard_name": standard_name,
        "description": description,
        "units": units,
    }


def test_directory_parser_accepts_only_exact_operational_daily_file():
    html = '''
    <a href="dn_xrsf-l2-avg1m_g16_d20240511_v2-2-1.nc">good</a>
    <a href="sci_xrsf-l2-avg1m_g16_d20240511_v2-2-1.nc">science</a>
    <a href="dn_xrsf-l2-avg1m_g16_d20240512_v2-2-1.nc">other-day</a>
    <a href="dn_xrsf-l2-avg1m_g18_d20240511_v2-2-1.nc">other-sat</a>
    '''
    assert parse_daily_file_links(html, "xrsf-l2-avg1m", 16, "20240511") == [
        "dn_xrsf-l2-avg1m_g16_d20240511_v2-2-1.nc"
    ]


def test_xrs_requires_separate_a_and_b_semantics():
    variables = {
        "xrsa_flux": meta(long_name="XRS-A irradiance flux", units="W/m2"),
        "xrsb_flux": meta(long_name="XRS-B irradiance flux", units="W/m2"),
        "random_flux": meta(long_name="other flux", units="W/m2"),
    }
    assert identify_xrs_candidates(variables) == {
        "A": ["xrsa_flux"],
        "B": ["xrsb_flux"],
    }


def test_proton_candidate_must_explicitly_be_integral_and_10mev():
    variables = {
        "differential_proton": meta(
            long_name="Differential proton flux 9.5 to 12 MeV",
            units="cm-2 s-1 sr-1 MeV-1",
        ),
        "integral_gt10": meta(
            long_name="Integral proton flux >10 MeV",
            units="pfu",
        ),
    }
    found = identify_proton_integral_candidates(variables)
    assert [row["variable"] for row in found] == ["integral_gt10"]


def test_differential_channel_near_10mev_is_not_silently_relabelled_integral():
    variables = {
        "proton_10mev": meta(
            long_name="Differential proton flux 10 MeV",
            units="cm-2 s-1 sr-1 MeV-1",
        )
    }
    assert identify_proton_integral_candidates(variables) == []


def test_time_and_quality_metadata_are_identified_without_values():
    variables = {
        "time": meta(standard_name="time", units="seconds since 2000-01-01 12:00:00"),
        "xrs_quality_flag": meta(long_name="XRS data quality flag"),
        "flux": meta(units="W/m2"),
    }
    assert time_variables(variables) == ["time"]
    assert quality_variables(variables) == ["xrs_quality_flag"]
