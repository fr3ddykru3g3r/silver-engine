"""Regression: legacy tools imports must work in the actual child process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).resolve().parents[1] / "tools/run_source_only_tests.py"
spec = importlib.util.spec_from_file_location("source_only_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SourceOnlyEnvironmentTests(unittest.TestCase):
    def test_real_child_resolves_tools_with_existing_path_preserved(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/tmp/iris-existing-path"}):
            env = runner.test_environment()
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[-1], "/tmp/iris-existing-path")
        child = subprocess.run([sys.executable, "-c",
            "import importlib.util; s=importlib.util.find_spec('tools.episode_semantics'); assert s is not None; print(s.origin)"],
            cwd=runner.ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertIn(str(runner.TEST_ROOT.parent / "tools/episode_semantics.py"), child.stdout)

    def test_runner_passes_environment_to_child(self):
        with patch.object(runner, "load_registry", return_value={"ignore_files": [], "deselect_nodeids": []}), \
             patch.object(runner, "validate_registry"), \
             patch.object(runner.subprocess, "run") as child:
            child.return_value.returncode = 0
            self.assertEqual(runner.run(), 0)
            self.assertIn(str(runner.TEST_ROOT.parent), child.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep))

if __name__ == "__main__":
    unittest.main()
