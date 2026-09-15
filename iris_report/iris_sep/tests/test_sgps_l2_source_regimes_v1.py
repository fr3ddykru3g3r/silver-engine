from iris_report.iris_sep.tools.scan_sgps_l2_source_regimes_v1 import contiguous_regimes, parse_listing


def test_parse_listing_extracts_prefix_date_version():
    html='''
    <a href="dn_sgps-l2-avg1m_g16_d20220101_v2-0-0.nc">a</a>
    <a href="sci_sgps-l2-avg1m_g16_d20240101_v3-0-2.nc">b</a>
    <a href="junk.nc">c</a>
    '''
    rows=parse_listing(html)
    assert rows == [
        {"name":"dn_sgps-l2-avg1m_g16_d20220101_v2-0-0.nc","prefix":"dn","day":"20220101","version":"2-0-0"},
        {"name":"sci_sgps-l2-avg1m_g16_d20240101_v3-0-2.nc","prefix":"sci","day":"20240101","version":"3-0-2"},
    ]


def test_contiguous_regimes_do_not_merge_version_changes():
    months=[
        {"year_month":"2022-01","prefixes":["dn"],"versions":["2-0-0"],"complete_daily_coverage":True},
        {"year_month":"2022-02","prefixes":["dn"],"versions":["2-0-0"],"complete_daily_coverage":True},
        {"year_month":"2022-03","prefixes":["sci"],"versions":["3-0-2"],"complete_daily_coverage":True},
    ]
    regimes=contiguous_regimes(months)
    assert len(regimes)==2
    assert regimes[0]["key"]=="dn|2-0-0"
    assert regimes[1]["key"]=="sci|3-0-2"


def test_partial_endpoint_preserves_complete_subrun_and_gaps_split():
    months = [dict(year_month=f"2024-{m:02d}", prefixes=["sci"],
                   versions=["3"], complete_daily_coverage=m < 12)
              for m in range(1, 13)]
    regimes = contiguous_regimes(months)
    assert [(r["months"], r["complete_months"]) for r in regimes] == [(11, 11), (1, 0)]
    assert len(contiguous_regimes([months[0], months[2]])) == 2
