from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools.run_operator_review_budget_diagnostic import budget_metrics


class OperatorReviewBudgetTests(unittest.TestCase):
    def test_top_budget_capture_and_enrichment(self):
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        t = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        out = budget_metrics(y, p, t, 0.2)
        self.assertEqual(out["review_rows"], 2)
        self.assertEqual(out["captured_positive_rows"], 1)
        self.assertAlmostEqual(out["event_capture_fraction"], 0.5)
        self.assertAlmostEqual(out["precision_within_review_budget"], 0.5)
        self.assertAlmostEqual(out["enrichment_vs_random_review"], 2.5)

    def test_cutoff_tie_reports_robust_capture_bounds(self):
        y = np.array([1, 1, 0, 0], dtype=int)
        p = np.array([0.9, 0.5, 0.5, 0.1])
        t = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
        out = budget_metrics(y, p, t, 0.5)
        self.assertEqual(out["review_rows"], 2)
        self.assertEqual(out["cutoff_tie_rows"], 2)
        self.assertEqual(out["cutoff_slots_selected"], 1)
        self.assertAlmostEqual(out["tie_robust_event_capture_min"], 0.5)
        self.assertAlmostEqual(out["tie_robust_event_capture_max"], 1.0)

    def test_invalid_fraction_fails(self):
        with self.assertRaises(ValueError):
            budget_metrics([0, 1], [0.1, 0.9], pd.date_range("2020-01-01", periods=2, tz="UTC"), 0.0)


if __name__ == "__main__":
    unittest.main()
