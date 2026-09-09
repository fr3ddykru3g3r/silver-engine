import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.fixed_forecast_benchmark import (
    build_design_matrix,
    fit_fixed_reference_forecaster,
    predict_with_frozen_reference,
    score_recovery_arm,
)
from iris_report.iris_sep.src.iris_sep.missingness_experiment import (
    recover_causal_forward_fill,
)


class FixedForecastBenchmarkTests(unittest.TestCase):
    def _fixture(self):
        rng = np.random.default_rng(20260905)
        rows = 96
        labels = np.tile([0, 1], rows // 2)
        values = rng.normal(size=(rows, 4))
        values[:, 0] = labels * 1.5 + rng.normal(scale=0.7, size=rows)
        values[:, 1] = np.sin(np.arange(rows) / 5.0) + rng.normal(
            scale=0.2,
            size=rows,
        )
        observed = np.ones_like(values, dtype=bool)
        structural = np.zeros_like(values, dtype=bool)
        # One feature is structurally unavailable during the fit block. It must
        # be excluded rather than silently learning from a value that did not exist.
        structural[:24, 3] = True
        observed[:24, 3] = False
        values[:24, 3] = np.nan
        roles = np.array(
            ["fit"] * 24
            + ["calibration"] * 24
            + ["threshold"] * 24
            + ["score"] * 24
        )
        return values, observed, structural, labels, roles

    def test_reference_fit_freezes_calibration_and_threshold(self):
        values, observed, structural, labels, roles = self._fixture()
        forecaster = fit_fixed_reference_forecaster(
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
        )
        self.assertTrue(0.0 <= forecaster.threshold <= 1.0)
        self.assertEqual(forecaster.calibration.fit_role, "validation_calibration")
        self.assertFalse(forecaster.supported_features[3])

    def test_identical_frozen_prediction_has_zero_forecast_degradation(self):
        values, observed, structural, labels, roles = self._fixture()
        forecaster = fit_fixed_reference_forecaster(
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
        )
        no_reconstruction = np.zeros_like(observed)
        reference = predict_with_frozen_reference(
            forecaster,
            values=values,
            observed_mask=observed,
            reconstructed_mask=no_reconstruction,
        )
        result = score_recovery_arm(
            forecaster,
            labels=labels,
            roles=roles,
            reference_probabilities=reference,
            candidate_probabilities=reference,
        )
        self.assertAlmostEqual(
            result["delta_candidate_minus_reference"]["TSS"],
            0.0,
        )
        self.assertAlmostEqual(
            result["delta_candidate_minus_reference"]["BRIER"],
            0.0,
        )

    def test_score_time_outage_uses_frozen_model_without_retraining(self):
        values, observed, structural, labels, roles = self._fixture()
        forecaster = fit_fixed_reference_forecaster(
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
        )
        reference = predict_with_frozen_reference(
            forecaster,
            values=values,
            observed_mask=observed,
            reconstructed_mask=np.zeros_like(observed),
        )
        holdout = np.zeros_like(observed)
        score_rows = np.flatnonzero(roles == "score")
        holdout[score_rows[::2], 0] = True
        experimental_observed = observed & ~holdout
        forward = recover_causal_forward_fill(
            values,
            observed,
            structural,
            holdout,
        )
        candidate = predict_with_frozen_reference(
            forecaster,
            values=forward.values,
            observed_mask=experimental_observed,
            reconstructed_mask=forward.reconstructed_mask,
        )
        result = score_recovery_arm(
            forecaster,
            labels=labels,
            roles=roles,
            reference_probabilities=reference,
            candidate_probabilities=candidate,
        )
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["score_rows"], 24)
        self.assertTrue(np.isfinite(candidate).all())

    def test_hidden_truth_is_not_read_when_observed_mask_is_false(self):
        values, observed, structural, labels, roles = self._fixture()
        forecaster = fit_fixed_reference_forecaster(
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
        )
        score_row = int(np.flatnonzero(roles == "score")[0])
        hidden_observed = observed.copy()
        hidden_observed[score_row, 0] = False
        reconstruction = np.zeros_like(observed)
        first = predict_with_frozen_reference(
            forecaster,
            values=values,
            observed_mask=hidden_observed,
            reconstructed_mask=reconstruction,
        )
        mutated_hidden_truth = values.copy()
        mutated_hidden_truth[score_row, 0] = 1e12
        second = predict_with_frozen_reference(
            forecaster,
            values=mutated_hidden_truth,
            observed_mask=hidden_observed,
            reconstructed_mask=reconstruction,
        )
        self.assertAlmostEqual(first[score_row], second[score_row], places=14)

    def test_design_rejects_cell_declared_both_observed_and_reconstructed(self):
        values, observed, structural, labels, roles = self._fixture()
        forecaster = fit_fixed_reference_forecaster(
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
        )
        reconstructed = np.zeros_like(observed)
        reconstructed[30, 0] = True
        with self.assertRaises(ValueError):
            build_design_matrix(
                values,
                observed,
                reconstructed,
                fit_medians=forecaster.fit_medians,
                supported_features=forecaster.supported_features,
            )

    def test_locked_or_extra_role_is_rejected_before_fit(self):
        values, observed, structural, labels, roles = self._fixture()
        roles = roles.copy()
        roles[-1] = "locked_test"
        with self.assertRaises(ValueError):
            fit_fixed_reference_forecaster(
                values=values,
                observed_mask=observed,
                structural_unavailable_mask=structural,
                labels=labels,
                roles=roles,
            )


if __name__ == "__main__":
    unittest.main()
