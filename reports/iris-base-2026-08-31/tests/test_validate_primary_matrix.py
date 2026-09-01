from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import validate_primary_matrix as validator  # noqa: E402


class PrimaryMatrixValidatorTests(unittest.TestCase):
    def test_exploratory_morphology_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (root / "fidelity_utility_points.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["arm", "tss_delta_vs_duplicate"])
                writer.writeheader()
                for arm in ("base", "pil", "pil_blur", "geometry_flip", "block_shuffle"):
                    writer.writerow({"arm": arm, "tss_delta_vs_duplicate": "0"})
            errors = validator.validate_primary_artifact(root)
            self.assertTrue(any("exploratory morphology" in error for error in errors))
            self.assertTrue(
                any(
                    "missing required" in error or "no primary_metrics" in error
                    for error in errors
                )
            )

    def test_exact_primary_matrix_and_shared_test_ids_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metrics = root / "primary_metrics.csv"
            with metrics.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["science_arm"])
                writer.writeheader()
                for arm in validator.PRIMARY_SCIENCE_ARMS:
                    writer.writerow({"science_arm": arm})
            ids = ["test-a", "test-b"]
            implementation = {
                "R": "real",
                "Rw": "real_weighted",
                "D": "duplicate",
                "L0": "base",
                "L2": "hj",
                "L3": "hj_pil",
            }
            for science_arm, name in implementation.items():
                directory = root / "outputs" / name
                directory.mkdir(parents=True)
                payload = {
                    "arm": name,
                    "added_positive_rows": 0 if science_arm in ("R", "Rw") else 250,
                    "threshold_source": "validation",
                }
                (directory / "metrics.json").write_text(json.dumps(payload))
                with (directory / "test_predictions.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["sample_id", "y", "p"])
                    writer.writeheader()
                    for sample_id in ids:
                        writer.writerow({"sample_id": sample_id, "y": 0, "p": 0.1})
            self.assertEqual(validator.validate_primary_artifact(root), [])

    def test_continuation_inputs_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = validator.validate(
                evidence_dir=root / "evidence",
                fits_dir=root / "fits",
                checkpoint=root / "base.pt",
                expected_fits=5273,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertGreaterEqual(len(result["errors"]), 3)


if __name__ == "__main__":
    unittest.main()
