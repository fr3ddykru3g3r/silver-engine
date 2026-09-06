import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.physics_gap_experiment import (
    benchmark_hidden_maps,
)
from iris_report.iris_sep.src.iris_sep.simple_physics import RotateSpreadConfig


class PhysicsGapExperimentTests(unittest.TestCase):
    def _fixture(self):
        base = np.arange(24, dtype=float).reshape(4, 6)
        maps = np.stack([np.roll(base, index, axis=1) for index in range(8)])
        observed = np.ones(8, dtype=bool)
        structural = np.zeros(8, dtype=bool)
        times = np.arange(8, dtype=float) * 24.0 * 3600.0
        roles = np.array(
            [
                "fit",
                "fit",
                "calibration",
                "threshold",
                "score",
                "score",
                "score",
                "score",
            ]
        )
        config = RotateSpreadConfig(
            longitude_degrees_per_pixel=60.0,
            rotation_degrees_per_day=60.0,
            diffusion_pixels2_per_day=0.0,
            max_substep_hours=24.0,
            validated_horizon_hours=72.0,
        )
        return maps, observed, structural, times, roles, config

    def test_consecutive_hidden_maps_use_only_last_earlier_real_map(self):
        maps, observed, structural, times, roles, config = self._fixture()
        holdout = np.zeros(8, dtype=bool)
        holdout[[5, 6]] = True
        result = benchmark_hidden_maps(
            maps=maps,
            map_observed=observed,
            structural_unavailable=structural,
            issue_time_unix_seconds=times,
            roles=roles,
            holdout_rows=holdout,
            config=config,
        )
        self.assertEqual(result["scored_maps"], 2)
        self.assertEqual(result["rows"][0]["source_index"], 4)
        self.assertEqual(result["rows"][1]["source_index"], 4)
        self.assertAlmostEqual(result["rows"][0]["gap_hours"], 24.0)
        self.assertAlmostEqual(result["rows"][1]["gap_hours"], 48.0)
        self.assertAlmostEqual(result["mean_physics_mae"], 0.0, places=12)
        self.assertGreater(result["mean_persistence_mae"], 0.0)

    def test_future_map_mutation_cannot_change_earlier_reconstruction(self):
        maps, observed, structural, times, roles, config = self._fixture()
        holdout = np.zeros(8, dtype=bool)
        holdout[5] = True
        first = benchmark_hidden_maps(
            maps=maps,
            map_observed=observed,
            structural_unavailable=structural,
            issue_time_unix_seconds=times,
            roles=roles,
            holdout_rows=holdout,
            config=config,
        )
        mutated = maps.copy()
        mutated[7] += 1e9
        second = benchmark_hidden_maps(
            maps=mutated,
            map_observed=observed,
            structural_unavailable=structural,
            issue_time_unix_seconds=times,
            roles=roles,
            holdout_rows=holdout,
            config=config,
        )
        self.assertEqual(first["rows"][0], second["rows"][0])

    def test_structurally_unavailable_map_cannot_be_hidden_as_known_truth(self):
        maps, observed, structural, times, roles, config = self._fixture()
        observed = observed.copy()
        structural = structural.copy()
        observed[5] = False
        structural[5] = True
        holdout = np.zeros(8, dtype=bool)
        holdout[5] = True
        with self.assertRaises(ValueError):
            benchmark_hidden_maps(
                maps=maps,
                map_observed=observed,
                structural_unavailable=structural,
                issue_time_unix_seconds=times,
                roles=roles,
                holdout_rows=holdout,
                config=config,
            )

    def test_holdout_is_restricted_to_score_role(self):
        maps, observed, structural, times, roles, config = self._fixture()
        holdout = np.zeros(8, dtype=bool)
        holdout[2] = True
        with self.assertRaises(ValueError):
            benchmark_hidden_maps(
                maps=maps,
                map_observed=observed,
                structural_unavailable=structural,
                issue_time_unix_seconds=times,
                roles=roles,
                holdout_rows=holdout,
                config=config,
            )

    def test_missing_prior_map_is_an_abstention_not_future_fill(self):
        base = np.arange(12, dtype=float).reshape(3, 4)
        maps = np.stack([base, np.roll(base, 1, axis=1), np.roll(base, 2, axis=1)])
        observed = np.ones(3, dtype=bool)
        structural = np.zeros(3, dtype=bool)
        times = np.arange(3, dtype=float) * 24.0 * 3600.0
        roles = np.array(["score", "score", "score"])
        holdout = np.array([True, False, True])
        config = RotateSpreadConfig(90.0, 90.0, max_substep_hours=24.0)
        result = benchmark_hidden_maps(
            maps=maps,
            map_observed=observed,
            structural_unavailable=structural,
            issue_time_unix_seconds=times,
            roles=roles,
            holdout_rows=holdout,
            config=config,
        )
        self.assertEqual(result["abstained_indices"], [0])
        self.assertEqual(result["scored_maps"], 1)
        self.assertAlmostEqual(result["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
