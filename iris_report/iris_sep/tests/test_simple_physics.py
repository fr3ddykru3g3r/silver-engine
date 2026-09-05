import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.simple_physics import (
    RotateSpreadConfig,
    reconstruct_rotate_and_spread,
)


class SimplePhysicsTests(unittest.TestCase):
    def setUp(self):
        self.field = np.arange(24, dtype=float).reshape(4, 6) - 5.0

    def test_zero_gap_returns_exact_observation(self):
        config = RotateSpreadConfig(60.0, 13.0, 0.0)
        result = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=0,
            config=config,
        )
        np.testing.assert_array_equal(result.field, self.field)
        self.assertEqual(result.substeps, 0)
        self.assertEqual(result.uncertainty_proxy, 0.0)

    def test_one_pixel_rotation_without_diffusion(self):
        config = RotateSpreadConfig(
            longitude_degrees_per_pixel=10.0,
            rotation_degrees_per_day=10.0,
            diffusion_pixels2_per_day=0.0,
            max_substep_hours=24.0,
            validated_horizon_hours=48.0,
        )
        result = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=24,
            config=config,
        )
        np.testing.assert_allclose(
            result.field,
            np.roll(self.field, 1, axis=1),
            atol=1e-12,
        )
        self.assertAlmostEqual(result.mean_before, result.mean_after, places=12)
        self.assertAlmostEqual(result.uncertainty_proxy, 0.5)

    def test_diffusion_spreads_a_local_peak_and_preserves_grid_mean(self):
        field = np.zeros((7, 9), dtype=float)
        field[3, 4] = 10.0
        config = RotateSpreadConfig(
            longitude_degrees_per_pixel=40.0,
            rotation_degrees_per_day=0.0,
            diffusion_pixels2_per_day=0.1,
            max_substep_hours=6.0,
        )
        result = reconstruct_rotate_and_spread(
            field,
            gap_hours=24,
            config=config,
        )
        self.assertLess(result.field.max(), 10.0)
        self.assertGreater(result.field[3, 3], 0.0)
        self.assertAlmostEqual(result.field.mean(), field.mean(), places=12)

    def test_invalid_geometry_nonfinite_field_and_negative_gap_fail_closed(self):
        with self.assertRaises(ValueError):
            RotateSpreadConfig(0.0, 1.0)
        config = RotateSpreadConfig(10.0, 1.0)
        with self.assertRaises(ValueError):
            reconstruct_rotate_and_spread(
                [[1.0, np.nan], [2.0, 3.0]],
                gap_hours=1.0,
                config=config,
            )
        with self.assertRaises(ValueError):
            reconstruct_rotate_and_spread(
                self.field,
                gap_hours=-1.0,
                config=config,
            )

    def test_uncertainty_proxy_is_monotone_and_bounded(self):
        config = RotateSpreadConfig(
            10.0,
            0.0,
            0.0,
            validated_horizon_hours=12.0,
        )
        short = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=3.0,
            config=config,
        )
        longer = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=9.0,
            config=config,
        )
        beyond = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=30.0,
            config=config,
        )
        self.assertLess(short.uncertainty_proxy, longer.uncertainty_proxy)
        self.assertEqual(beyond.uncertainty_proxy, 1.0)

    def test_method_identifies_itself_as_reduced_physics_not_mhd(self):
        config = RotateSpreadConfig(10.0, 0.0)
        result = reconstruct_rotate_and_spread(
            self.field,
            gap_hours=1.0,
            config=config,
        )
        self.assertEqual(result.method_id, "ROTATE_SPREAD_2D_V1")
        self.assertEqual(result.method_class, "PHYSICS_CONSTRAINED")
        self.assertNotIn("MHD", result.method_id)
        self.assertIn("last_observation_only", result.constraints)


if __name__ == "__main__":
    unittest.main()
