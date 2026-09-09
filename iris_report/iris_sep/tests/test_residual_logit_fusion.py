import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.residual_logit_fusion import (
    ResidualFusionConfig,
    ResidualLogitFusion,
)


class ResidualLogitFusionTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
        self.solar = np.array([.03, .06, .10, .12, .45, .55, .09, .60, .08, .65])
        self.xrs = np.array([.02, .05, .07, .10, .60, .70, .06, .75, .04, .80])
        self.proton = np.array([.03, .04, .05, .08, .55, .65, .05, .70, .04, .75])

    def test_fit_is_deterministic_and_weights_nonnegative(self):
        cfg = ResidualFusionConfig(l2_weight=.01)
        a = ResidualLogitFusion(cfg).fit(self.solar, self.xrs, self.proton, self.y)
        b = ResidualLogitFusion(cfg).fit(self.solar, self.xrs, self.proton, self.y)
        self.assertGreaterEqual(a.fit_.xrs_weight, 0.0)
        self.assertGreaterEqual(a.fit_.proton_weight, 0.0)
        np.testing.assert_allclose(a.predict_proba(self.solar, self.xrs, self.proton),
                                   b.predict_proba(self.solar, self.xrs, self.proton), atol=1e-12)

    def test_zero_reliability_removes_context_effect(self):
        model = ResidualLogitFusion(ResidualFusionConfig(l2_weight=.01)).fit(
            self.solar, self.xrs, self.proton, self.y,
            xrs_reliability=np.zeros(len(self.y)),
            proton_reliability=np.zeros(len(self.y)),
        )
        z = model.decision_function(
            self.solar, np.full(len(self.y), .99), np.full(len(self.y), .99),
            xrs_reliability=np.zeros(len(self.y)),
            proton_reliability=np.zeros(len(self.y)),
        )
        expected = np.log(self.solar / (1 - self.solar)) + model.fit_.bias
        np.testing.assert_allclose(z, expected, atol=1e-12)

    def test_higher_context_risk_cannot_decrease_logit(self):
        model = ResidualLogitFusion(ResidualFusionConfig(l2_weight=.01)).fit(
            self.solar, self.xrs, self.proton, self.y
        )
        solar = np.array([.20, .20])
        low = np.array([.05, .05])
        high = np.array([.80, .80])
        z_low = model.decision_function(solar, low, low)
        z_high = model.decision_function(solar, high, high)
        self.assertTrue(np.all(z_high >= z_low))

    def test_residuals_are_bounded_before_weighting(self):
        cfg = ResidualFusionConfig(residual_logit_limit=1.25, l2_weight=.01)
        model = ResidualLogitFusion(cfg).fit(self.solar, self.xrs, self.proton, self.y)
        solar_z, xrs_delta, proton_delta = model._features(
            np.array([.5, .5]), np.array([1e-12, 1 - 1e-12]), np.array([1e-12, 1 - 1e-12]),
            prevalence=model.fit_.prevalence,
        )
        self.assertLessEqual(np.max(np.abs(xrs_delta)), 1.25 + 1e-12)
        self.assertLessEqual(np.max(np.abs(proton_delta)), 1.25 + 1e-12)
        self.assertTrue(np.isfinite(solar_z).all())

    def test_invalid_probability_fails_closed(self):
        model = ResidualLogitFusion()
        with self.assertRaises(ValueError):
            model.fit(self.solar, np.full(len(self.y), 1.1), self.proton, self.y)


if __name__ == "__main__":
    unittest.main()
