import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools.run_missingness_forecast_benchmark import (
    load_package,
    run,
)


class MissingnessForecastToolTests(unittest.TestCase):
    def _package(self, root: Path, *, locked=False):
        rng = np.random.default_rng(20260905)
        rows = 96
        labels = np.tile([0, 1], rows // 2).astype(np.int8)
        values = rng.normal(size=(rows, 4))
        values[:, 0] = labels * 1.5 + rng.normal(scale=0.8, size=rows)
        observed = np.ones_like(values, dtype=bool)
        structural = np.zeros_like(values, dtype=bool)
        structural[:24, 3] = True
        observed[:24, 3] = False
        values[:24, 3] = np.nan
        roles = np.array(
            ["fit"] * 24
            + ["calibration"] * 24
            + ["threshold"] * 24
            + ["score"] * 24
        )
        issue_ids = np.array([f"issue-{index:03d}" for index in range(rows)])
        unit_ids = np.array([f"unit-{index:03d}" for index in range(rows)])
        # 48-hour spacing makes every role boundary strictly greater than the
        # frozen 24-hour purge without relying on metadata alone.
        times = np.arange(rows, dtype=np.float64) * 48.0 * 3600.0
        package = root / "package.npz"
        np.savez(
            package,
            values=values,
            observed_mask=observed,
            structural_unavailable_mask=structural,
            labels=labels,
            roles=roles,
            issue_ids=issue_ids,
            unit_ids=unit_ids,
            issue_time_unix_seconds=times,
        )
        metadata = root / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "format": "IRIS_SEP_TRAIN_ONLY_MISSINGNESS_PACKAGE_V1",
                    "scope": "TRAIN_ONLY_NEW_CROSSING_MISSINGNESS",
                    "target": "new_sep_10mev_10pfu_within_24h",
                    "locked_test_included": locked,
                    "chronological_roles_verified": True,
                    "episode_disjoint_roles_verified": True,
                    "purge_hours": 24,
                    "source_manifest_sha256": "0" * 64,
                    "feature_names": ["a", "b", "c", "old-era-only"],
                }
            )
        )
        return package, metadata

    def test_fixture_package_runs_end_to_end_without_physics_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            output = root / "run"
            receipt = run(
                package,
                metadata,
                output,
                missing_fraction=0.20,
                seed=7,
            )
            self.assertEqual(
                receipt["status"],
                "COMPLETED_TRAIN_ONLY_MISSINGNESS_DIAGNOSTIC",
            )
            self.assertFalse(receipt["locked_test_accessed"])
            self.assertFalse(receipt["final_new_crossing_result"])
            self.assertFalse(receipt["superiority_established"])
            self.assertEqual(receipt["supported_features"], 3)
            self.assertGreater(receipt["held_out_cells"], 0)
            self.assertEqual(
                set(receipt["arms"]),
                {
                    "MASK_AWARE_NO_FILL",
                    "TRAIN_FIT_MEDIAN",
                    "CAUSAL_FORWARD_FILL",
                },
            )
            self.assertTrue((output / "preregistration.json").exists())
            self.assertTrue((output / "holdout.npz").exists())
            self.assertTrue((output / "predictions.npz").exists())
            self.assertTrue((output / "receipt.json").exists())
            prereg = json.loads((output / "preregistration.json").read_text())
            self.assertFalse(prereg["physics_arm_included"])

    def test_output_directory_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            output = root / "run"
            run(package, metadata, output, missing_fraction=0.10, seed=7)
            with self.assertRaises(ValueError):
                run(package, metadata, output, missing_fraction=0.10, seed=7)

    def test_locked_test_package_fails_before_model_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root, locked=True)
            with self.assertRaises(ValueError):
                load_package(package, metadata)

    def test_unit_crossing_roles_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, metadata = self._package(root)
            with np.load(package, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["unit_ids"] = arrays["unit_ids"].copy()
            arrays["unit_ids"][23] = "crossing-unit"
            arrays["unit_ids"][24] = "crossing-unit"
            np.savez(package, **arrays)
            with self.assertRaises(ValueError):
                load_package(package, metadata)


if __name__ == "__main__":
    unittest.main()
