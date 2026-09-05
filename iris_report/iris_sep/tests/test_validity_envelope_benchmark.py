import tempfile
import unittest
from pathlib import Path

from iris_report.iris_sep.tools.run_validity_envelope_benchmark import run


class ValidityBenchmarkTests(unittest.TestCase):
    def test_every_fault_and_status_path_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory) / "run", trials=900, seed=20260905)
        self.assertEqual(result["iris_status_accuracy"], 1.0)
        self.assertEqual(result["iris_unsafe_valid_outputs"], 0)
        self.assertTrue(all(row["trials"] > 0 for row in result["counts"].values()))
        self.assertFalse(result["scientific_superiority_established"])


if __name__ == "__main__":
    unittest.main()
