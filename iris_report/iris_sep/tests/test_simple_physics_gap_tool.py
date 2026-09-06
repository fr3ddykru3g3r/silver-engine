import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools.run_simple_physics_gap_benchmark import (
    load_package,
    run,
)


class SimplePhysicsGapToolTests(unittest.TestCase):
    def _package(self, root: Path, *, locked=False):
        base = np.arange(24, dtype=float).reshape(4, 6)
        maps = np.stack([np.roll(base, index, axis=1) for index in range(12)])
        observed = np.ones(12, dtype=bool)
        structural = np.zeros(12, dtype=bool)
        roles = np.array(
            ["fit"] * 3
            + ["calibration"] * 2
            + ["threshold"] * 2
            + ["score"] * 5
        )
        issue_ids = np.array([f"map-{index:02d}" for index in range(12)])
        times = np.arange(12, dtype=float) * 24.0 * 3600.0
        package = root / "maps.npz"
        np.savez(
            package,
            maps=maps,
            map_observed=observed,
            structural_unavailable=structural,
            roles=roles,
            issue_ids=issue_ids,
            issue_time_unix_seconds=times,
        )
        metadata = root / "maps.json"
        metadata.write_text(
            json.dumps(
                {
                    "format": "IRIS_SEP_TRAIN_ONLY_MAGNETIC_MAP_PACKAGE_V1",
                    "scope": "TRAIN_ONLY_TRANSIENT_MAGNETIC_MAP_GAPS",
                    "locked_test_included": locked,
                    "source_manifest_sha256": "1" * 64,
                    "geometry": {
                        "longitude_degrees_per_pixel": 60.0,
                        "rotation_degrees_per_day": 60.0,
                        "diffusion_pixels2_per_day": 0.0,
                        "max_substep_hours": 24.0,
                        "validated_horizon_hours": 72.0,
                    },
                }
            )
        )
        return package, metadata

    def test_fixture_runner_completes_and_remains_non_sep_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            output = root / "run"
            receipt = run(
                package,
                metadata,
                output,
                missing_fraction=0.40,
                seed=7,
            )
            self.assertEqual(
                receipt["status"],
                "COMPLETED_TRAIN_ONLY_HIDDEN_MAP_DIAGNOSTIC",
            )
            self.assertFalse(receipt["locked_test_accessed"])
            self.assertFalse(receipt["downstream_sep_scored"])
            self.assertFalse(receipt["physics_advantage_established"])
            self.assertGreater(receipt["result"]["scored_maps"], 0)
            self.assertTrue((output / "preregistration.json").exists())
            self.assertTrue((output / "holdout.npz").exists())
            self.assertTrue((output / "receipt.json").exists())

    def test_locked_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root, locked=True)
            with self.assertRaises(ValueError):
                load_package(package, metadata)

    def test_geometry_is_explicit_not_silently_defaulted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            value = json.loads(metadata.read_text())
            del value["geometry"]["rotation_degrees_per_day"]
            metadata.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_package(package, metadata)

    def test_output_directory_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            output = root / "run"
            run(package, metadata, output, missing_fraction=0.20, seed=7)
            with self.assertRaises(ValueError):
                run(package, metadata, output, missing_fraction=0.20, seed=7)


if __name__ == "__main__":
    unittest.main()
