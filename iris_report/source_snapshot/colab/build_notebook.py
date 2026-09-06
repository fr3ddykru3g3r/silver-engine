#!/usr/bin/env python3
"""Build the self-contained Colab notebook from pinned project artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def b64(path: Path) -> tuple[str, str, int]:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest(), len(data)


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--evidence-archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--embed-evidence", action="store_true",
        help="Embed the metadata/evidence archive; omit for a faster light notebook.",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    source_path = Path(args.source_archive).resolve()
    evidence_path = Path(args.evidence_archive).resolve()
    source_b64, source_sha, source_bytes = b64(source_path)
    evidence_b64, evidence_sha, evidence_bytes = b64(evidence_path)

    controls = f'''from pathlib import Path
import base64
import hashlib
import os
import subprocess
import sys

# Each student must use a different Drive root. Kyros uses the first value;
# Lokesh changes only the suffix to _lokesh before running this cell.
USE_GOOGLE_DRIVE = True
DRIVE_ROOT = '/content/drive/MyDrive/iris_silver_engine_kyros'

# Stage controls. Run BASE first, then PHYSICS, then DOWNSTREAM.
RUN_BASE = '1'
RUN_PHYSICS = '0'
RUN_DOWNSTREAM = '0'
FITS_SCOPE = 'base'       # base -> physics -> all before downstream
ACQUIRE_FITS = '1'
JSOC_EXPORT_METHOD = 'url'
JSOC_EXPORT_PROTOCOL = 'fits'
SEED = '2026'

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK_ROOT = Path(DRIVE_ROOT)
else:
    WORK_ROOT = Path('/content/iris_silver_engine')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_DIR = WORK_ROOT / 'silver-engine-{source_sha[:12]}'
EVIDENCE_DIR = WORK_ROOT / 'evidence-{evidence_sha[:12]}'
FITS_SOURCE = WORK_ROOT / 'fits_cache'

os.environ['IRIS_WORK_ROOT'] = str(WORK_ROOT)
os.environ['IRIS_RUN_BASE'] = RUN_BASE
os.environ['IRIS_RUN_PHYSICS'] = RUN_PHYSICS
os.environ['IRIS_RUN_DOWNSTREAM'] = RUN_DOWNSTREAM
os.environ['IRIS_SEED'] = SEED
os.environ['IRIS_EVIDENCE_DIR'] = str(EVIDENCE_DIR)
os.environ['IRIS_FITS_SOURCE'] = str(FITS_SOURCE)
os.environ['IRIS_REQUIRE_LOCAL_FITS'] = '1'
os.environ['JSOC_EXPORT_METHOD'] = JSOC_EXPORT_METHOD
os.environ['JSOC_EXPORT_PROTOCOL'] = JSOC_EXPORT_PROTOCOL
print('WORK_ROOT =', WORK_ROOT)
print('Drive identity =', DRIVE_ROOT)
print('BASE/PHYSICS/DOWNSTREAM =', RUN_BASE, RUN_PHYSICS, RUN_DOWNSTREAM)
print('FITS_SCOPE =', FITS_SCOPE, '| FITS_SOURCE =', FITS_SOURCE)
'''

    source_cell = f'''# The matching source is embedded in this notebook.
SOURCE_ARCHIVE = WORK_ROOT / 'silver-engine-colab-source-{source_sha[:12]}.zip'
SOURCE_SHA256 = '{source_sha}'
SOURCE_B64 = {source_b64!r}
if not SOURCE_ARCHIVE.exists():
    SOURCE_ARCHIVE.write_bytes(base64.b64decode(SOURCE_B64))
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
if sha256_file(SOURCE_ARCHIVE) != SOURCE_SHA256:
    raise RuntimeError('Embedded source checksum mismatch')
if not (SOURCE_DIR / 'iris-model').is_dir():
    import zipfile
    staging = WORK_ROOT / 'source_embedded'
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        archive.extractall(staging)
    candidates = [staging] + [p.parent for p in staging.rglob('iris-model') if p.is_dir()]
    SOURCE_DIR = next((p for p in candidates if (p / 'iris-model').is_dir()), None)
    if SOURCE_DIR is None:
        raise RuntimeError('Embedded source archive does not contain iris-model/')
required = ['colab/iris_colab_runner.py', 'colab/acquire_sharp_fits.py', 'iris-model/train_generator_v2.py', 'iris-model/fit_cache.py']
missing = [name for name in required if not (SOURCE_DIR / name).is_file()]
if missing:
    raise RuntimeError(f'Missing embedded source files: {{missing}}')
print('SOURCE_DIR =', SOURCE_DIR, '| source SHA256 =', sha256_file(SOURCE_ARCHIVE))
'''

    if args.embed_evidence:
        evidence_cell = f'''# The evidence archive is embedded too. It contains metadata/labels, not binary FITS images.
EVIDENCE_ARCHIVE = WORK_ROOT / 'iris-historical-evidence-integrity-{evidence_sha[:12]}.tar.gz'
EVIDENCE_SHA256 = '{evidence_sha}'
EVIDENCE_B64 = {evidence_b64!r}
if not EVIDENCE_ARCHIVE.exists():
    EVIDENCE_ARCHIVE.write_bytes(base64.b64decode(EVIDENCE_B64))
if sha256_file(EVIDENCE_ARCHIVE) != EVIDENCE_SHA256:
    raise RuntimeError('Embedded evidence checksum mismatch')
if not (EVIDENCE_DIR / 'data/derived/training_manifest.csv.gz').is_file():
    import tarfile
    staging = WORK_ROOT / 'evidence_embedded'
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(EVIDENCE_ARCHIVE) as archive:
        archive.extractall(staging)
    roots = [p.parent.parent.parent for p in staging.rglob('training_manifest.csv.gz') if p.parent.name == 'derived']
    if not roots:
        raise RuntimeError('Embedded evidence archive has no training manifest')
    EVIDENCE_DIR = roots[0]
os.environ['IRIS_EVIDENCE_DIR'] = str(EVIDENCE_DIR)
print('EVIDENCE_DIR =', EVIDENCE_DIR, '| evidence SHA256 =', sha256_file(EVIDENCE_ARCHIVE))
'''
    else:
        evidence_cell = f'''# The evidence archive is supplied once from the project bundle or as a separate upload.
# It contains metadata/labels, not binary FITS images. Expected SHA256 is pinned below.
import shutil
import zipfile
EVIDENCE_ARCHIVE = WORK_ROOT / 'iris-historical-evidence-integrity-{evidence_sha[:12]}.tar.gz'
EVIDENCE_SHA256 = '{evidence_sha}'
if not EVIDENCE_ARCHIVE.exists():
    candidates = [p for p in list(Path('/content').glob('*.tar.gz')) + list(Path('/content').glob('*.zip')) if p.is_file()]
    if not candidates:
        from google.colab import files
        print('Upload the project bundle .zip or evidence .tar.gz once.')
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError('No evidence/project archive uploaded')
        candidates = [Path('/content') / next(iter(uploaded))]
    copied = False
    for candidate in candidates:
        if candidate.name.endswith(('.tar.gz', '.tgz')):
            shutil.copyfile(candidate, EVIDENCE_ARCHIVE)
            copied = True
            break
        if zipfile.is_zipfile(candidate):
            with zipfile.ZipFile(candidate) as bundle:
                names = [n for n in bundle.namelist() if 'evidence' in n.lower() and n.endswith(('.tar.gz', '.tgz'))]
                if names:
                    EVIDENCE_ARCHIVE.write_bytes(bundle.read(names[0]))
                    copied = True
                    break
    if not copied:
        raise RuntimeError('No evidence .tar.gz found in the uploaded files')
if sha256_file(EVIDENCE_ARCHIVE) != EVIDENCE_SHA256:
    raise RuntimeError('Evidence checksum mismatch; use the evidence file from the matching project bundle')
if not (EVIDENCE_DIR / 'data/derived/training_manifest.csv.gz').is_file():
    import tarfile
    staging = WORK_ROOT / 'evidence_embedded'
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(EVIDENCE_ARCHIVE) as archive:
        archive.extractall(staging)
    roots = [p.parent.parent.parent for p in staging.rglob('training_manifest.csv.gz') if p.parent.name == 'derived']
    if not roots:
        raise RuntimeError('Evidence archive has no training manifest')
    EVIDENCE_DIR = roots[0]
os.environ['IRIS_EVIDENCE_DIR'] = str(EVIDENCE_DIR)
print('EVIDENCE_DIR =', EVIDENCE_DIR, '| evidence SHA256 =', sha256_file(EVIDENCE_ARCHIVE))
'''

    install_cell = '''packages = [
    'numpy>=2.0', 'pandas>=2.2', 'requests>=2.32', 'astropy>=6.1',
    'scipy>=1.14', 'scikit-learn>=1.5', 'matplotlib>=3.9',
    'tqdm>=4.66', 'PyYAML>=6.0', 'drms>=0.9.1'
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *packages])
import torch
print('torch =', torch.__version__, '| cuda =', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('WARNING: CPU mode is only for smoke testing; select Runtime > Change runtime type > T4 GPU.')
'''

    acquisition_cell = '''# This is the data gate. It must finish before the runner cell.
# JSOC_EMAIL is read only in memory and is never written to the notebook or Drive.
import getpass
import json

report_path = FITS_SOURCE / 'acquisition_report.json'
if ACQUIRE_FITS == '1':
    if (not report_path.exists()
            or json.loads(report_path.read_text()).get('status') != 'PASS'
            or json.loads(report_path.read_text()).get('scope') != FITS_SCOPE):
        os.environ['JSOC_EMAIL'] = getpass.getpass('Registered JSOC export email (not saved): ')
    subprocess.check_call([
        sys.executable, SOURCE_DIR / 'colab/acquire_sharp_fits.py',
        '--evidence-dir', str(EVIDENCE_DIR), '--output-dir', str(FITS_SOURCE),
        '--scope', FITS_SCOPE, '--seed', SEED,
    ], cwd=str(SOURCE_DIR))
else:
    print('ACQUIRE_FITS=0; expecting an already complete verified cache')
'''

    # The runner exposes the same preflight function used by its CLI entrypoint;
    # calling it directly keeps this cell from starting a second subprocess.
    preflight_cell = '''# Fail closed before constructing a model.
import importlib.util
spec = importlib.util.spec_from_file_location('iris_runner', SOURCE_DIR / 'colab/iris_colab_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.preflight(EVIDENCE_DIR)
print('DATA_GATE_READY_FOR_CONFIGURED_STAGE')
'''

    runner_cell = '''# Start only the configured stage after the preflight cell passes.
subprocess.check_call([
    sys.executable, SOURCE_DIR / 'colab/iris_colab_runner.py'
], cwd=str(SOURCE_DIR))
'''

    results_cell = '''# Optional: package only the results/provenance from this Drive root.
import shutil
results_root = WORK_ROOT / 'runs'
if results_root.is_dir():
    archive = shutil.make_archive(str(WORK_ROOT / 'iris_colab_results'), 'zip', root_dir=str(results_root))
    print('results archive:', archive)
else:
    print('No runs directory yet; no results archive created.')
'''

    notebook = {
        "cells": [
            markdown("""# IRIS/ISEF solar magnetogram experiment — self-contained Colab runner\n\nThis notebook contains the pinned source and metadata/evidence archive. The evidence archive does **not** contain HMI/SHARP FITS images, so the data-gate cell uses a registered JSOC export route to materialize only the deterministic samples required by the selected stage.\n\nNo forecast result is valid unless `REAL_FITS_CACHE_PREFLIGHT_PASS` appears before the runner starts. Use separate Drive roots for Kyros and Lokesh.\n"""),
            markdown("""## Staged protocol\n\nRun BASE first with `FITS_SCOPE = 'base'`. After the BASE fidelity gate passes, set `RUN_BASE = '0'`, `RUN_PHYSICS = '1'`, `FITS_SCOPE = 'physics'` and rerun the controls, acquisition, preflight, and runner cells. Before downstream forecasting, set `RUN_BASE = '0'`, `RUN_PHYSICS = '0'`, `RUN_DOWNSTREAM = '1'`, `FITS_SCOPE = 'all'`.\n\nIf JSOC returns 403 for `url_quick/as-is`, set `JSOC_EXPORT_METHOD = 'url'` and `JSOC_EXPORT_PROTOCOL = 'fits'`, then rerun acquisition. Do not disable TLS verification and do not substitute PNGs.\n"""),
            code(controls), code(install_cell), code(source_cell), code(evidence_cell),
            code(acquisition_cell), code(preflight_cell), code(runner_cell),
            markdown("""## After BASE passes\n\nChange only the stage controls described above and rerun the relevant cells. Keep the same Drive root, seed, source checksum, evidence checksum, and frozen evidence archive.\n"""),
            code(results_cell),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "iris_artifacts": {
                "source_sha256": source_sha, "source_bytes": source_bytes,
                "evidence_sha256": evidence_sha, "evidence_bytes": evidence_bytes,
                "generator_training_authorized": False,
                "evidence_embedded": bool(args.embed_evidence),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(json.dumps({
        "output": str(output), "source_sha256": source_sha, "source_bytes": source_bytes,
        "evidence_sha256": evidence_sha, "evidence_bytes": evidence_bytes,
        "notebook_bytes": output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
