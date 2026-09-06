"""Run every IRIS-SEP source-only test while explicitly separating data-bound cases.

The registry is part of the source contract: entries must point to existing test
files/nodeids, and the excluded inventory is printed before pytest runs. This
prevents a missing gitignored dataset from masquerading as a software failure
without silently dropping source-only tests that share the same module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "iris_report/iris_sep/config/data_dependent_test_registry_v1.json"
TEST_ROOT = ROOT / "iris_report/iris_sep/tests"


def load_registry(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ignore_files = data.get("ignore_files")
    deselect = data.get("deselect_nodeids")
    if not isinstance(ignore_files, list) or not isinstance(deselect, list):
        raise ValueError("registry must contain ignore_files and deselect_nodeids lists")
    if len(ignore_files) != len(set(ignore_files)) or len(deselect) != len(set(deselect)):
        raise ValueError("registry contains duplicate exclusions")
    return data


def validate_registry(data: dict) -> None:
    discovered = {str(p.relative_to(ROOT)) for p in TEST_ROOT.glob("test_*.py")}
    for rel in data["ignore_files"]:
        if rel not in discovered:
            raise ValueError(f"ignored test file does not exist: {rel}")
    for nodeid in data["deselect_nodeids"]:
        rel = nodeid.split("::", 1)[0]
        if rel not in discovered:
            raise ValueError(f"deselected nodeid file does not exist: {nodeid}")
        if "::" not in nodeid:
            raise ValueError(f"deselected entry must be an exact pytest nodeid: {nodeid}")


def command(data: dict) -> list[str]:
    args = [sys.executable, "-m", "pytest", str(TEST_ROOT), "-v", "--maxfail=1"]
    for rel in data["ignore_files"]:
        args.extend(["--ignore", str(ROOT / rel)])
    for nodeid in data["deselect_nodeids"]:
        args.extend(["--deselect", nodeid])
    return args


def run(registry: Path = DEFAULT_REGISTRY) -> int:
    data = load_registry(registry)
    validate_registry(data)
    print("DATA_DEPENDENT_TEST_FILES_EXCLUDED_FROM_SOURCE_ONLY_CI")
    for rel in data["ignore_files"]:
        print("  file:", rel)
    for nodeid in data["deselect_nodeids"]:
        print("  nodeid:", nodeid)
    print("These tests remain required when their hash-pinned local inputs are materialized.")
    return subprocess.run(command(data), cwd=ROOT, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    raise SystemExit(run(args.registry))


if __name__ == "__main__":
    main()
