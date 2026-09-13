from iris_report.iris_sep.tools.probe_sgps_l2_filename_transition_v1 import classify, href_names


def test_href_names_extracts_only_netcdf_files():
    html = '''
    <a href="dn_sgps-l2-avg1m_g16_d20240501_v3-0-0.nc">a</a>
    <a href="sci_sgps-l2-avg1m_g16_d20240502_v3-0-0.nc">b</a>
    <a href="notes.txt">c</a>
    '''
    assert href_names(html) == [
        "dn_sgps-l2-avg1m_g16_d20240501_v3-0-0.nc",
        "sci_sgps-l2-avg1m_g16_d20240502_v3-0-0.nc",
    ]


def test_prefix_classification_is_explicit():
    assert classify("dn_sgps-l2-avg1m_g16_d20240501_v3-0-0.nc") == "DN_OPERATIONAL"
    assert classify("sci_sgps-l2-avg1m_g16_d20240501_v3-0-0.nc") == "SCIENCE_PREFIX"
    assert classify("foo_sgps-l2-avg1m_g16_d20240501.nc") == "OTHER_SGPS_AVG1M"
