from __future__ import annotations

import pandas as pd
import pytest

from tools.run_episode_normalized_development_benchmark_v1_compat import family_features_compat


def test_timezone_aware_proton_features_do_not_use_object_view() -> None:
    df = pd.DataFrame({
        "time": pd.to_datetime(["2016-01-01T00:00:00Z", "2016-01-01T00:05:00Z"], utc=True),
        "proton": [1.0, 2.0],
    })
    out = family_features_compat(df, pd.Timestamp("2016-01-01T00:10:00Z"), 0, "proton")
    assert out["p_n"] == 2.0
    assert out["p_last"] == 2.0
    assert out["p_age"] == pytest.approx(5.0)


def test_timezone_aware_xrs_features_preserve_a_b_streams() -> None:
    df = pd.DataFrame({
        "time": pd.to_datetime(["2016-01-01T00:00:00Z", "2016-01-01T00:01:00Z"], utc=True),
        "A": [1e-7, 2e-7],
        "B": [1e-6, 2e-6],
    })
    out = family_features_compat(df, pd.Timestamp("2016-01-01T00:02:00Z"), 0, "xrs")
    assert out["a_n"] == 2.0
    assert out["b_n"] == 2.0
    assert out["a_last"] == pytest.approx(2e-7)
    assert out["b_last"] == pytest.approx(2e-6)
    assert out["a_age"] == pytest.approx(1.0)
    assert out["b_age"] == pytest.approx(1.0)
