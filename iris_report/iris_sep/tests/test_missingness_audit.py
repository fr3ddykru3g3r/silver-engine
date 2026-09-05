import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from iris_report.iris_sep.src.iris_sep.missingness_audit import summarize_missingness


UTC = timezone.utc


class MissingnessAuditTests(unittest.TestCase):
    def test_manifest_counts_features_eras_quarters_and_outages(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        times = [start + timedelta(days=i) for i in range(6)]
        labels = [0, 1, 0, 1, 0, 0]
        features = ["mag", "xrs"]
        observed = np.array([
            [True, True],
            [False, True],
            [False, False],
            [False, True],
            [True, True],
            [True, False],
        ])
        eras = ["A", "A", "A", "B", "B", "B"]
        result = summarize_missingness(
            issue_times=times,
            labels=labels,
            feature_names=features,
            observed_mask=observed,
            eras=eras,
            expected_cadence_minutes=1440,
        )
        self.assertEqual(result["scope"], "TRAIN_ONLY_DESCRIPTIVE_MISSINGNESS_AUDIT")
        self.assertEqual(result["issue_count"], 6)
        self.assertEqual(result["positive_issue_count"], 2)
        self.assertEqual(result["complete_issue_count"], 2)
        self.assertEqual(result["all_features_missing_issue_count"], 1)

        mag = next(item for item in result["features"] if item["feature"] == "mag")
        self.assertEqual(mag["missing_count"], 3)
        self.assertAlmostEqual(mag["missing_fraction"], 0.5)
        self.assertEqual(mag["longest_missing_run_rows"], 3)
        self.assertAlmostEqual(mag["longest_missing_run_minutes"], 3 * 1440)
        self.assertAlmostEqual(mag["positive_missing_fraction"], 1.0)
        self.assertAlmostEqual(mag["negative_missing_fraction"], 0.25)
        self.assertAlmostEqual(mag["missing_fraction_difference_positive_minus_negative"], 0.75)

        self.assertEqual(len(result["by_era"]), 4)
        self.assertEqual(len(result["by_quarter"]), 2)
        self.assertIn("no event-independence", result["claim_boundary"])

    def test_large_timestamp_break_splits_missing_run(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        times = [start, start + timedelta(days=1), start + timedelta(days=10)]
        result = summarize_missingness(
            issue_times=times,
            labels=[0, 0, 1],
            feature_names=["mag"],
            observed_mask=[[False], [False], [False]],
            eras=["A", "A", "A"],
            expected_cadence_minutes=1440,
            continuity_tolerance=1.5,
        )
        feature = result["features"][0]
        self.assertEqual(feature["longest_missing_run_rows"], 2)
        self.assertAlmostEqual(feature["longest_missing_run_minutes"], 2 * 1440)

    def test_invalid_contracts_fail_closed(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        with self.assertRaises(ValueError):
            summarize_missingness(
                issue_times=[start, start], labels=[0, 1], feature_names=["x"],
                observed_mask=[[True], [False]], eras=["A", "A"],
                expected_cadence_minutes=1440,
            )
        with self.assertRaises(ValueError):
            summarize_missingness(
                issue_times=[start], labels=[2], feature_names=["x"],
                observed_mask=[[True]], eras=["A"], expected_cadence_minutes=1440,
            )
        with self.assertRaises(ValueError):
            summarize_missingness(
                issue_times=[start], labels=[0], feature_names=["x", "x"],
                observed_mask=[[True, True]], eras=["A"], expected_cadence_minutes=1440,
            )

    def test_zero_class_support_uses_null_fraction_not_fake_zero(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        result = summarize_missingness(
            issue_times=[start, start + timedelta(days=1)],
            labels=[0, 0], feature_names=["x"], observed_mask=[[True], [False]],
            eras=["A", "A"], expected_cadence_minutes=1440,
        )
        feature = result["features"][0]
        self.assertIsNone(feature["positive_missing_fraction"])
        self.assertIsNone(feature["missing_fraction_difference_positive_minus_negative"])


if __name__ == "__main__":
    unittest.main()
