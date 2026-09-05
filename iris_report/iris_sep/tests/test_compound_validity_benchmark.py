import tempfile
import unittest
from pathlib import Path
from iris_report.iris_sep.tools.run_compound_validity_benchmark import run


class CompoundValidityTests(unittest.TestCase):
    def test_compound_faults_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run(Path(directory)/"run",1200,20260905)
        self.assertEqual(result["status_errors"],0)
        self.assertEqual(result["unsafe_valid_outputs"],0)
        self.assertEqual(result["recovery_failures"],0)
        self.assertGreater(result["unique_fault_combinations"],100)


if __name__ == "__main__": unittest.main()
