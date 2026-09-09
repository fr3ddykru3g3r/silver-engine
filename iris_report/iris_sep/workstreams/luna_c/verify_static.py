"""Dependency-free static verification for Luna C's isolated workstream.

This script intentionally parses source with the standard library only.  It
does not import PyTorch, open a dataset, inspect benchmark partitions, or read
locked-test outcomes.  Colab should run the PyTorch unit/smoke tests separately
after installing its declared runtime.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCE_NAMES = ("model.py", "checkpoint.py", "smoke_test.py", "test_model.py", "test_unittest.py")
FORBIDDEN_CALLS = {
    "open",
    "read_csv",
    "read_parquet",
    "read_json",
    "read_pickle",
    "read_feather",
    "load_dataset",
    "glob",
    "iglob",
    "listdir",
    "walk",
    "urlopen",
}
FORBIDDEN_IMPORT_ROOTS = {
    "pandas",
    "polars",
    "datasets",
    "requests",
    "urllib",
    "sklearn",
}


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _import_root(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]
    return (node.module or "").split(".", 1)[0]


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _assert_source_is_data_agnostic(path: Path, tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            root = _import_root(node)
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise AssertionError(f"{path.name}: forbidden data/network import {root}")
        if isinstance(node, ast.Call) and _call_name(node) in FORBIDDEN_CALLS:
            raise AssertionError(f"{path.name}:{node.lineno}: forbidden data/network call {_call_name(node)}")


def run() -> dict[str, object]:
    parsed: dict[str, ast.Module] = {}
    for name in SOURCE_NAMES:
        path = HERE / name
        if not path.exists():
            raise AssertionError(f"missing workstream source: {name}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _assert_source_is_data_agnostic(path, tree)
        parsed[name] = tree

    model_names = _top_level_names(parsed["model.py"])
    required_model_names = {
        "CausalConv1d",
        "CausalTemporalExpert",
        "IRISSEPConfig",
        "IRISSEPInputs",
        "IRISSEPModel",
        "ModalityInput",
        "ForecastOutput",
        "sample_modality_keep_mask",
        "compute_task_losses",
    }
    missing_model = required_model_names - model_names
    if missing_model:
        raise AssertionError(f"model.py missing required interfaces: {sorted(missing_model)}")

    modality_assignment = next(
        (
            node
            for node in parsed["model.py"].body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "MODALITY_NAMES" for target in node.targets)
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "MODALITY_NAMES"
            )
        ),
        None,
    )
    if modality_assignment is None:
        raise AssertionError("model.py must declare MODALITY_NAMES")
    value = ast.literal_eval(modality_assignment.value)
    if value != ("magnetic", "eruption", "particle"):
        raise AssertionError(f"unexpected modality names: {value!r}")

    model_text = (HERE / "model.py").read_text(encoding="utf-8")
    required_model_tokens = (
        "masked_fill",
        "gate_weights",
        "time_since_observation_hours",
        "onset_hazard_head",
        "flare_activity_head",
        "cme_activity_head",
    )
    for token in required_model_tokens:
        if token not in model_text:
            raise AssertionError(f"model.py is missing required architecture token: {token}")

    checkpoint_names = _top_level_names(parsed["checkpoint.py"])
    required_checkpoint_names = {
        "capture_rng_state",
        "restore_rng_state",
        "build_checkpoint_payload",
        "atomic_torch_save",
        "save_checkpoint",
        "load_checkpoint",
    }
    missing_checkpoint = required_checkpoint_names - checkpoint_names
    if missing_checkpoint:
        raise AssertionError(f"checkpoint.py missing required interfaces: {sorted(missing_checkpoint)}")
    checkpoint_text = (HERE / "checkpoint.py").read_text(encoding="utf-8")
    for token in ('"model"', '"optimizer"', '"scheduler"', '"scaler"', '"rng_state"'):
        if token not in checkpoint_text:
            raise AssertionError(f"checkpoint.py missing state component: {token}")

    smoke_text = (HERE / "smoke_test.py").read_text(encoding="utf-8")
    for token in ("synthetic_input_only", "scientific_claims", "run_smoke_test"):
        if token not in smoke_text:
            raise AssertionError(f"smoke_test.py missing safety marker: {token}")

    return {
        "status": "PASS",
        "imported_torch": False,
        "dataset_accessed": False,
        "locked_test_accessed": False,
        "files_checked": list(SOURCE_NAMES),
        "note": "AST/interface/data-agnostic checks only; PyTorch execution remains a Colab gate.",
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True))
    except (AssertionError, OSError, SyntaxError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        sys.exit(1)
