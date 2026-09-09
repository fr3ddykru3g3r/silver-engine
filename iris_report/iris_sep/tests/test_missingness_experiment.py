import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.missingness_experiment import (
    deterministic_transient_random_holdout,
    eligible_transient_cells,
    evaluate_forecast_preservation,
    recover_causal_forward_fill,
    recover_train_median,
    transient_block_holdout,
)


class MissingnessExperimentTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(20, dtype=float).reshape(5, 4)
        self.observed = np.ones((5, 4), dtype=bool)
        self.structural = np.zeros((5, 4), dtype=bool)
        self.structural[0, 3] = True
        self.observed[0, 3] = False

    def test_random_holdout_is_deterministic_and_never_structural(self):
        first = deterministic_transient_random_holdout(
            self.observed,
            self.structural,
            missing_fraction=0.25,
            seed=7,
        )
        second = deterministic_transient_random_holdout(
            self.observed,
            self.structural,
            missing_fraction=0.25,
            seed=7,
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse((first & self.structural).any())
        self.assertFalse((first & ~self.observed).any())

    def test_block_holdout_clips_out_structural_cells(self):
        mask = transient_block_holdout(
            self.observed,
            self.structural,
            start_row=0,
            length_rows=2,
            feature_indices=[2, 3],
        )
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[0, 3])
        self.assertTrue(mask[1, 3])

    def test_train_median_recovers_hidden_truth_but_not_structural_history(self):
        holdout = np.zeros_like(self.observed)
        holdout[3, 1] = True
        result = recover_train_median(
            self.values,
            self.observed,
            self.structural,
            holdout,
            fit_rows=[True, True, True, False, False],
        )
        self.assertTrue(result.reconstructed_mask[3, 1])
        self.assertFalse(result.available_mask[0, 3])
        self.assertTrue(np.isnan(result.values[0, 3]))
        self.assertEqual(result.values[3, 1], np.median([1.0, 5.0, 9.0]))

    def test_forward_fill_uses_only_past_and_does_not_cross_structural_gap(self):
        holdout = np.zeros_like(self.observed)
        holdout[2, 0] = True
        result = recover_causal_forward_fill(
            self.values,
            self.observed,
            self.structural,
            holdout,
        )
        self.assertEqual(result.values[2, 0], self.values[1, 0])
        self.assertTrue(np.isnan(result.values[0, 3]))
        self.assertFalse(result.available_mask[0, 3])

    def test_identical_forecasts_have_zero_degradation(self):
        labels = [0, 1, 0, 1, 0, 1]
        probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9])
        result = evaluate_forecast_preservation(
            labels=labels,
            reference_probabilities=probabilities,
            candidate_probabilities=probabilities,
            reference_threshold=0.5,
            candidate_threshold=0.5,
            role="train_only_inner_score",
        )
        self.assertEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(
            result["delta_candidate_minus_reference"]["TSS"],
            0.0,
        )
        self.assertAlmostEqual(
            result["delta_candidate_minus_reference"]["BRIER"],
            0.0,
        )

    def test_abstention_is_visible_as_lost_coverage_and_reference_is_matched(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        reference = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9])
        candidate = reference.copy()
        candidate[[0, 5]] = np.nan
        result = evaluate_forecast_preservation(
            labels=labels,
            reference_probabilities=reference,
            candidate_probabilities=candidate,
            reference_threshold=0.5,
            candidate_threshold=0.5,
            role="train_only_inner_score",
        )
        self.assertAlmostEqual(result["coverage"], 4 / 6)
        self.assertEqual(result["retained_rows"], 4)
        self.assertAlmostEqual(
            result["delta_candidate_minus_reference"]["TSS"],
            0.0,
        )

    def test_locked_or_other_roles_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            evaluate_forecast_preservation(
                labels=[0, 1],
                reference_probabilities=[0.1, 0.9],
                candidate_probabilities=[0.1, 0.9],
                reference_threshold=0.5,
                candidate_threshold=0.5,
                role="locked_test",
            )

    def test_structural_cell_cannot_also_be_observed(self):
        with self.assertRaises(ValueError):
            eligible_transient_cells([[True]], [[True]])


if __name__ == "__main__":
    unittest.main()
