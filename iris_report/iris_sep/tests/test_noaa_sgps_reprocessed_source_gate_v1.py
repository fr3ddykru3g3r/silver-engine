from datetime import date

from iris_report.iris_sep.tools.run_noaa_sgps_reprocessed_source_gate_v1 import (
    bands_match, expected_dates, live_bands, parse_science_daily_links,
)


def test_parser_accepts_only_science_goes16_daily_files():
    html = '<a href="sci_sgps-l2-avg1m_g16_d20240501_v3-0-2.nc">a</a><a href="dn_sgps-l2-avg1m_g16_d20240502_v3-0-2.nc">b</a><a href="sci_sgps-l2-avg1m_g18_d20240501_v3-0-2.nc">c</a>'
    assert parse_science_daily_links(html, "2024-05") == ["sci_sgps-l2-avg1m_g16_d20240501_v3-0-2.nc"]


def test_interval_edges_are_the_only_calendar_exceptions():
    values = expected_dates("2020-03", date(2020, 3, 11), date(2020, 3, 13))
    assert values == [date(2020, 3, 11), date(2020, 3, 12), date(2020, 3, 13)]


def test_live_energy_parser_requires_every_named_channel():
    channels = ["P1", "P2A"]
    result = live_bands([{"channel": "P1", "energy": "1020-1860 keV"}, {"channel": "P2A", "energy": "1.9-2.3 MeV"}], channels)
    assert result["passed"] is True
    assert result["mapping"]["P1"] == [1.02, 1.86]


def test_numeric_binding_is_ordered_and_fail_closed():
    channels = ["P1", "P2A"]
    live = {"P1": [1.02, 1.86], "P2A": [1.9, 2.3]}
    assert bands_match([1.02, 1.9], [1.86, 2.3], live, channels, 0.02)["passed"] is True
    assert bands_match([1.02, 2.4], [1.86, 3.0], live, channels, 0.02)["passed"] is False


def test_contract_requires_both_yaw_states_before_training():
    import json
    from iris_report.iris_sep.tools.run_noaa_sgps_reprocessed_source_gate_v1 import CONTRACT
    contract = json.loads(CONTRACT.read_text())
    assert contract["go_no_go"]["live_sensor_reduction_verified_for_both_yaw_states"] is True
    assert "upright" in contract["sensor_binding_gate"]
    assert "yaw-flipped" in contract["sensor_binding_gate"]
