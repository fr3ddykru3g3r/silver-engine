"""Build the credential-free, development-only IRIS-SEP Colab notebook."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
IRIS_SEP = ROOT / "iris_sep"
OUTPUT = Path(__file__).with_name("IRIS_SEP_Development_Training_2026-09-04.ipynb")

BUNDLE_FILES = (
    "iris_sep/src/iris_sep/__init__.py",
    "iris_sep/src/iris_sep/modeling/__init__.py",
    "iris_sep/src/iris_sep/modeling/tabular_multibranch.py",
    "iris_sep/tools/__init__.py",
    "iris_sep/tools/prepare_sepnet_v1_development.py",
    "iris_sep/tools/train_tabular_multibranch.py",
    "iris_sep/tests/test_tabular_model_runtime.py",
    "iris_sep/workstreams/luna_g_data_pipeline/__init__.py",
    "iris_sep/workstreams/luna_g_data_pipeline/iris_sep_pipeline/__init__.py",
    "iris_sep/workstreams/luna_g_data_pipeline/iris_sep_pipeline/cohort.py",
    "iris_sep/workstreams/luna_g_data_pipeline/iris_sep_pipeline/errors.py",
    "iris_sep/workstreams/luna_g_data_pipeline/iris_sep_pipeline/schemas.py",
    "iris_sep/workstreams/luna_g_data_pipeline/test_pipeline_unittest.py",
    "iris_sep/workstreams/luna_i_eval_ops/__init__.py",
    "iris_sep/workstreams/luna_i_eval_ops/evaluation.py",
    "iris_sep/workstreams/luna_i_eval_ops/operator.py",
    "iris_sep/workstreams/luna_i_eval_ops/test_eval_ops_unittest.py",
)


def _bundle() -> str:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE_FILES:
            source = ROOT / relative
            archive.write(source, f"iris_report/{relative}")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _cell(cell_type: str, source: str) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build() -> Path:
    encoded = _bundle()
    cells = [
        _cell("markdown", """# IRIS-SEP development training — locked-test safe

This notebook trains only on the publisher-separated SEPNET V1 **training** file. It cannot produce a final SEPVAL score, a new-crossing score, an operational certification, or a breakthrough claim. It contains no credential and no locked/test URL. The particle-context branch is intentionally unavailable in this legacy table; this run is a pipeline and architecture development experiment, not the final three-modality benchmark."""),
        _cell("code", """%pip install -q pandas==3.0.1

import hashlib, importlib.metadata, json, os, platform, subprocess, sys
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Select a GPU runtime before training."
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "pandas": importlib.metadata.version("pandas"),
})"""),
        _cell("code", """from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/IRIS_SEP_DEVELOPMENT_2026-09-04")
WORKSPACE = Path("/content/iris_sep_workspace")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
print({"drive_root": str(DRIVE_ROOT), "workspace": str(WORKSPACE)})"""),
        _cell("code", f"""import base64, io, zipfile

BUNDLE_B64 = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
print("Embedded source extracted; files:", len(list(WORKSPACE.rglob("*.py"))))"""),
        _cell("code", """from urllib.request import urlopen

TRAIN_URL = "https://raw.githubusercontent.com/yuyian/SEP-Prediction/f9cff73adfa41c4fbffc73a8693c529d39e80995/data/rolling_combinded_training.csv"
TRAIN_SHA256 = "59e9e659798798047728cf85a59f2e182dbbff87c5becdae068b27a5b9ed2454"
raw_path = DRIVE_ROOT / "rolling_combinded_training.csv"
if not raw_path.exists():
    raw_path.write_bytes(urlopen(TRAIN_URL, timeout=120).read())
actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
assert actual == TRAIN_SHA256, (actual, TRAIN_SHA256)
print({"training_bytes": raw_path.stat().st_size, "training_sha256": actual, "locked_test_downloaded": False})"""),
        _cell("code", """processed_path = DRIVE_ROOT / "sepnet_v1_development_v3.csv"
manifest_path = DRIVE_ROOT / "sepnet_v1_development_v3_manifest.json"
expected_processed = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
expected_manifest = "18c10d4fc76a2ce5e03b9a271951003f274435aa00180fcb90e4f2947eedaebb"
if not processed_path.exists() and not manifest_path.exists():
    subprocess.run([
        sys.executable, "-m", "iris_report.iris_sep.tools.prepare_sepnet_v1_development",
        "--source", str(raw_path), "--output", str(processed_path), "--manifest", str(manifest_path),
    ], cwd=WORKSPACE, check=True)
assert hashlib.sha256(processed_path.read_bytes()).hexdigest() == expected_processed
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_manifest
print(json.loads(manifest_path.read_text()))"""),
        _cell("code", """subprocess.run([
    sys.executable, "-m", "unittest",
    "iris_report.iris_sep.tests.test_tabular_model_runtime",
    "iris_report.iris_sep.workstreams.luna_g_data_pipeline.test_pipeline_unittest",
    "iris_report.iris_sep.workstreams.luna_i_eval_ops.test_eval_ops_unittest",
    "-v",
], cwd=WORKSPACE, check=True)
print("GPU_RUNTIME_AND_CONTRACT_TESTS_PASS")"""),
        _cell("code", """run_dir = DRIVE_ROOT / "tabular_multibranch_v1"
command = [
    sys.executable, "-m", "iris_report.iris_sep.tools.train_tabular_multibranch",
    "--source", str(processed_path),
    "--source-manifest", str(manifest_path),
    "--output-dir", str(run_dir),
    "--max-epochs", "200", "--patience", "20", "--batch-size", "256",
]
if run_dir.exists():
    command.append("--resume")
subprocess.run(command, cwd=WORKSPACE, check=True)
print("TRAINING_COMPLETE")"""),
        _cell("code", """receipt_path = run_dir / "receipt.json"
receipt = json.loads(receipt_path.read_text())
assert receipt["locked_test_accessed"] is False
assert receipt["claims_forbidden"] == ["SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION"]
print({
    "status": receipt["status"],
    "device": receipt["device"],
    "particle_context_absent_from_v1": receipt["particle_context_absent_from_v1"],
    "monitor_selection_side_metrics": receipt["metrics"]["validation_monitor"],
    "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "claim_scope": "DEVELOPMENT_ONLY; NOT SEPVAL; NOT OPERATIONAL",
})"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
