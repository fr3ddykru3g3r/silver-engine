import inspect
import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import run_daily_modality_outage as outage


class DailyModalityOutageTests(unittest.TestCase):
    @staticmethod
    def source_clock(unit="ns"):
        return pd.Series(pd.date_range("2019-01-01T00:00:00Z", periods=240, freq="24h").as_unit(unit))

    def test_block_selection_has_no_label_or_probability_argument(self):
        parameters = inspect.signature(outage.deterministic_daily_blocks).parameters
        lowered = {name.lower() for name in parameters}
        self.assertFalse(any("label" in name for name in lowered))
        self.assertFalse(any("target" in name for name in lowered))
        self.assertFalse(any("prob" in name for name in lowered))

    def test_24_72_168_hours_map_to_1_3_7_nonoverlapping_daily_cycles(self):
        times = self.source_clock()
        lower, upper = times.iloc[0], times.iloc[-1]
        for hours, cycles in ((24, 1), (72, 3), (168, 7)):
            with self.subTest(hours=hours):
                blocks = outage.deterministic_daily_blocks(times, hours, lower, upper)
                self.assertEqual(len(blocks), 5)
                self.assertTrue(all(b["duration_daily_cycles"] == cycles for b in blocks))
                starts = sorted(b["issue_start"] for b in blocks)
                width = pd.Timedelta(days=cycles)
                self.assertTrue(all(b >= a + width for a, b in zip(starts, starts[1:])))
                for block in blocks:
                    self.assertEqual(block["issue_end_exclusive"] - block["issue_start"], width)

    def test_nanosecond_and_microsecond_clocks_select_same_blocks(self):
        ns = self.source_clock("ns")
        us = self.source_clock("us")
        for hours in (24, 72, 168):
            left = outage.deterministic_daily_blocks(ns, hours, ns.iloc[0], ns.iloc[-1])
            right = outage.deterministic_daily_blocks(us, hours, us.iloc[0], us.iloc[-1])
            left_view = [(b["issue_start"].isoformat(), b["issue_end_exclusive"].isoformat()) for b in left]
            right_view = [(b["issue_start"].isoformat(), b["issue_end_exclusive"].isoformat()) for b in right]
            self.assertEqual(left_view, right_view)

    def test_projection_hides_only_score_rows_and_requested_modality(self):
        times = self.source_clock()
        n = len(times)
        finite = np.ones((n, 5), dtype=bool)
        score = np.zeros(n, dtype=bool)
        score[20:220] = True
        feature_indices = np.asarray([1, 3], dtype=int)

        holdout, outage_rows, blocks = outage.scenario_holdout(
            finite, score, times, times, feature_indices, 72
        )
        self.assertTrue(outage_rows.any())
        self.assertFalse(holdout[~score].any())
        self.assertFalse(holdout[:, [0, 2, 4]].any())
        self.assertTrue(holdout[:, feature_indices].any())
        self.assertEqual(len(blocks), 5)

    def test_interval_rows_remain_visible_when_requested_cells_have_zero_exposure(self):
        times = self.source_clock()
        n = len(times)
        finite = np.ones((n, 4), dtype=bool)
        finite[:, 2] = False
        score = np.zeros(n, dtype=bool)
        score[20:220] = True

        holdout, outage_rows, blocks = outage.scenario_holdout(
            finite, score, times, times, np.asarray([2], dtype=int), 168
        )
        self.assertTrue(outage_rows.any())
        self.assertFalse(holdout.any())
        self.assertTrue(all(b["eligible_score_rows_affected"] > 0 for b in blocks))
        self.assertTrue(all(b["finite_interface_cells_hidden"] == 0 for b in blocks))
        self.assertTrue(all(b["evaluable"] is False for b in blocks))

    def test_sparse_eligible_rows_do_not_change_source_clock_duration(self):
        times = self.source_clock()
        n = len(times)
        finite = np.ones((n, 2), dtype=bool)
        score = np.zeros(n, dtype=bool)
        score[20:220:3] = True
        holdout, outage_rows, blocks = outage.scenario_holdout(
            finite, score, times, times, np.asarray([0], dtype=int), 168
        )
        self.assertTrue(outage_rows.any())
        self.assertTrue(holdout.any())
        for block in blocks:
            start = pd.Timestamp(block["issue_start"])
            end = pd.Timestamp(block["issue_end_exclusive"])
            self.assertEqual(end - start, pd.Timedelta(days=7))
            self.assertLessEqual(block["eligible_score_rows_affected"], 3)

    def test_one_class_affected_metrics_are_reported_without_ranking_claim(self):
        y = np.zeros(6, dtype=int)
        p = np.linspace(0.05, 0.30, 6)
        mask = np.ones(6, dtype=bool)
        result = outage.metrics_on_mask(y, p, mask, threshold=0.2, prevalence=0.1)
        self.assertEqual(result["rows"], 6)
        self.assertEqual(result["positives"], 0)
        self.assertIsNone(result["AUPRC"])
        self.assertIsNone(result["AUROC"])
        self.assertIsNone(result["matched_detection"])


if __name__ == "__main__":
    unittest.main()
