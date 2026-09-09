import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from iris_report.iris_sep.tools.run_compact_nonfinite_replay import (
    audit_preregistered_dependencies,
    run,
)


class CompactReplayToolTests(unittest.TestCase):
    def test_dependency_audit_detects_mutation_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.py"
            first.write_text("alpha\n")
            preregistration = {
                "dependency_sha256": {
                    "a.py": hashlib.sha256(first.read_bytes()).hexdigest(),
                    "missing.py": "0" * 64,
                }
            }
            audit = audit_preregistered_dependencies(root, preregistration)
            self.assertFalse(audit["all_match"])
            self.assertEqual(audit["missing"], ["missing.py"])
            self.assertEqual(audit["mismatched"], [])

            preregistration["dependency_sha256"].pop("missing.py")
            self.assertTrue(audit_preregistered_dependencies(root, preregistration)["all_match"])
            first.write_text("mutated\n")
            mutated = audit_preregistered_dependencies(root, preregistration)
            self.assertFalse(mutated["all_match"])
            self.assertEqual(mutated["mismatched"], ["a.py"])

    def test_missing_artifact_receipt_includes_frozen_preregistration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "iris_sep"
            diagnostic = root / "artifacts" / "train_inner_diagnostic_v4"
            output = Path(directory) / "receipt.json"
            result = run(root, diagnostic, output)
            self.assertEqual(result["status"], "NOT_RUN_MISSING_LOCAL_ARTIFACTS")
            self.assertIsNone(result["causal_conclusion"])
            self.assertIn(str(diagnostic / "preregistration.json"), result["missing_paths"])
            persisted = json.loads(output.read_text())
            self.assertEqual(persisted["status"], result["status"])


if __name__ == "__main__":
    unittest.main()
