import unittest

import numpy as np

from iris_report.iris_sep.tools.run_selective_outage_policy import (
    exposure_mask,
    full_cohort_accounting,
)


class SelectiveOutagePolicyTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0], dtype=int)
        self.p = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
        self.score = np.ones(4, dtype=bool)
        self.affected = np.array([False, True, True, False])
        self.threshold = 0.5

    def test_abstention_removes_only_declared_outage_rows(self):
        exposed = exposure_mask(self.score, self.affected, "ABSTAIN_ON_DECLARED_OUTAGE")
        np.testing.assert_array_equal(exposed, np.array([True, False, False, True]))

    def test_abstained_positive_is_counted_as_missed_not_erased(self):
        row = full_cohort_accounting(
            self.y,
            self.p,
            self.score,
            self.affected,
            self.threshold,
            "ABSTAIN_ON_DECLARED_OUTAGE",
        )
        self.assertEqual(row["covered_rows"], 2)
        self.assertEqual(row["abstained_rows"], 2)
        self.assertEqual(row["total_positives"], 2)
        self.assertEqual(row["abstained_positives"], 1)
        self.assertEqual(row["true_alerts"], 1)
        self.assertEqual(row["coverage_adjusted_missed_positives"], 1)
        self.assertAlmostEqual(row["coverage_adjusted_detection_fraction"], 0.5)

    def test_degraded_policy_keeps_probability_coverage_visible(self):
        row = full_cohort_accounting(
            self.y,
            self.p,
            self.score,
            self.affected,
            self.threshold,
            "EXPOSE_DEGRADED_ON_DECLARED_OUTAGE",
        )
        self.assertEqual(row["covered_rows"], 4)
        self.assertEqual(row["abstained_rows"], 0)
        self.assertEqual(row["degraded_rows"], 2)
        self.assertEqual(row["degraded_positives"], 1)
        self.assertAlmostEqual(row["coverage_fraction"], 1.0)

    def test_always_expose_has_full_coverage_without_degraded_status(self):
        row = full_cohort_accounting(
            self.y,
            self.p,
            self.score,
            self.affected,
            self.threshold,
            "ALWAYS_EXPOSE_NORMAL",
        )
        self.assertEqual(row["covered_rows"], 4)
        self.assertEqual(row["degraded_rows"], 0)
        self.assertEqual(row["false_alerts"], 1)


if __name__ == "__main__":
    unittest.main()
