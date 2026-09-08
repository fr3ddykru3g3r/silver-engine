from __future__ import annotations

import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.alert_decision_filter import (
    binary_metrics,
    improvement_summary,
    meta_features,
    select_decision_threshold,
)


class AlertDecisionFilterTests(unittest.TestCase):
    def test_meta_features_are_position_stable_and_finite(self):
        kwargs = {
            "v3_probability": [0.02, 0.03, 0.04],
            "v1_probability": [0.018, 0.028, 0.038],
            "solar_only_probability": [0.004, 0.005, 0.006],
            "raw_solar_probability": [0.01, 0.02, 0.03],
            "raw_context_probability": [0.2, 0.4, 0.6],
            "solar_seed_range": [0.01, 0.02, 0.03],
            "solar_seed_std": [0.003, 0.006, 0.009],
            "context_seed_range": [0.04, 0.05, 0.06],
            "context_seed_std": [0.01, 0.02, 0.03],
        }
        x = meta_features(**kwargs)
        self.assertEqual(x.shape, (3, 17))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_absent_family_cannot_be_smuggled_as_extra_feature(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            meta_features(
                v3_probability=[0.1, 0.2],
                v1_probability=[0.1, 0.2],
                solar_only_probability=[0.1, 0.2],
                raw_solar_probability=[0.1, 0.2],
                raw_context_probability=[0.1],
                solar_seed_range=[0.01, 0.01],
                solar_seed_std=[0.01, 0.01],
                context_seed_range=[0.01, 0.01],
                context_seed_std=[0.01, 0.01],
            )

    def test_threshold_selection_enforces_tp_retention(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        score = np.array([0.9, 0.8, 0.2, 0.7, 0.6, 0.5, 0.1, 0.05])
        result = select_decision_threshold(
            labels=y,
            decision_score=score,
            baseline_true_positives=3,
            max_true_positive_loss=1,
        )
        self.assertGreaterEqual(result["true_positives"], 2)
        self.assertEqual(result["minimum_true_positives"], 2)

    def test_large_improvement_requires_no_detection_loss_and_20pct_fp_cut(self):
        reference = binary_metrics([1, 1] + [0] * 10, [True, True] + [True] * 10)
        candidate = binary_metrics([1, 1] + [0] * 10, [True, True] + [True] * 7 + [False] * 3)
        result = improvement_summary(reference, candidate)
        self.assertTrue(result["large_development_improvement_gate"])
        lost = binary_metrics([1, 1] + [0] * 10, [True, False] + [True] * 6 + [False] * 4)
        self.assertFalse(improvement_summary(reference, lost)["large_development_improvement_gate"])


if __name__ == "__main__":
    unittest.main()
