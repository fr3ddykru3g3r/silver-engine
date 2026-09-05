import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.two_state_features import build_two_state_features


class TwoStateFeatureTests(unittest.TestCase):
    def test_exact_24h_state_and_delta(self):
        frame = pd.DataFrame({
            "window_end": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-02T00:00Z", "2026-01-03T00:00Z"]),
            "a": [1.0, 4.0, 10.0],
            "b": [2.0, 3.0, 5.0],
        })
        out, receipt = build_two_state_features(frame, ["a", "b"])
        self.assertEqual(receipt["lag_available_rows"], 2)
        self.assertTrue(np.isnan(out.loc[0, "lag24__a"]))
        self.assertEqual(out.loc[1, "lag24__a"], 1.0)
        self.assertEqual(out.loc[2, "delta24__a"], 6.0)
        self.assertEqual(out.loc[2, "delta24__b"], 2.0)

    def test_nearest_within_tolerance_is_causal(self):
        frame = pd.DataFrame({
            "window_end": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T23:00Z", "2026-01-02T01:30Z"]),
            "x": [1.0, 99.0, 4.0],
        })
        out, _ = build_two_state_features(frame, ["x"])
        # Target for final row is 2026-01-01 01:30; 01:00 is closer than 23:00.
        self.assertEqual(out.loc[2, "lag24__x"], 1.0)

    def test_missing_pair_keeps_delta_missing(self):
        frame = pd.DataFrame({
            "window_end": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-02T00:00Z"]),
            "x": [np.nan, 3.0],
        })
        out, _ = build_two_state_features(frame, ["x"])
        self.assertTrue(np.isnan(out.loc[1, "delta24__x"]))

    def test_unsorted_or_duplicate_time_fails_closed(self):
        for times in (
            ["2026-01-02T00:00Z", "2026-01-01T00:00Z"],
            ["2026-01-01T00:00Z", "2026-01-01T00:00Z"],
        ):
            with self.subTest(times=times), self.assertRaises(ValueError):
                build_two_state_features(pd.DataFrame({"window_end": pd.to_datetime(times), "x": [1.0, 2.0]}), ["x"])


if __name__ == "__main__":
    unittest.main()
