from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.modeling.availability_direct_specialist import (
    DirectSpecialistConfig,
    feature_schema_sha256,
    fit_predict_direct_specialist,
    state_feature_names,
)


class AvailabilityDirectSpecialistTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 80
        solar = rng.normal(size=n)
        xrs = rng.normal(size=n)
        proton = rng.normal(size=n)
        score = 0.9 * solar + 1.0 * xrs + 0.8 * proton + 0.7 * solar * proton
        y = (score > np.median(score)).astype(np.int8)
        self.frame = pd.DataFrame({
            "solar_a": solar,
            "xrs_a": xrs,
            "proton_a": proton,
        })
        self.y = y
        self.fit = np.zeros(n, dtype=bool)
        self.fit[:60] = True
        self.cfg = DirectSpecialistConfig(seeds=(7, 13), n_estimators=20, n_jobs=1)

    def test_state_schema_drops_only_unavailable_family(self):
        no_xrs = state_feature_names(
            "NO_XRS", solar=("solar_a",), xrs=("xrs_a",), proton=("proton_a",)
        )
        no_proton = state_feature_names(
            "NO_PROTON", solar=("solar_a",), xrs=("xrs_a",), proton=("proton_a",)
        )
        self.assertEqual(no_xrs, ("solar_a", "proton_a"))
        self.assertEqual(no_proton, ("solar_a", "xrs_a"))

    def test_future_column_is_rejected(self):
        with self.assertRaises(ValueError):
            state_feature_names(
                "NO_XRS",
                solar=("Future_cheat",),
                xrs=("xrs_a",),
                proton=("proton_a",),
            )

    def test_direct_specialist_emits_finite_probabilities_without_imputation(self):
        names = state_feature_names(
            "NO_XRS", solar=("solar_a",), xrs=("xrs_a",), proton=("proton_a",)
        )
        result = fit_predict_direct_specialist(
            frame=self.frame,
            labels=self.y,
            fit_mask=self.fit,
            feature_names=names,
            config=self.cfg,
        )
        p = result["probability"]
        self.assertEqual(p.shape, (len(self.y),))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertFalse(result["imputation_used"])
        self.assertFalse(result["reconstruction_used"])
        self.assertFalse(result["runtime_retraining"])
        self.assertEqual(result["fit_rows"], 60)

    def test_native_nan_is_allowed_but_infinity_is_not(self):
        names = ("solar_a", "proton_a")
        nan_frame = self.frame.copy()
        nan_frame.loc[3, "proton_a"] = np.nan
        fit_predict_direct_specialist(
            frame=nan_frame,
            labels=self.y,
            fit_mask=self.fit,
            feature_names=names,
            config=self.cfg,
        )
        inf_frame = self.frame.copy()
        inf_frame.loc[3, "proton_a"] = np.inf
        with self.assertRaises(ValueError):
            fit_predict_direct_specialist(
                frame=inf_frame,
                labels=self.y,
                fit_mask=self.fit,
                feature_names=names,
                config=self.cfg,
            )

    def test_schema_hash_is_order_sensitive(self):
        self.assertNotEqual(
            feature_schema_sha256(("solar_a", "proton_a")),
            feature_schema_sha256(("proton_a", "solar_a")),
        )


if __name__ == "__main__":
    unittest.main()
