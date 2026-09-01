# IRIS full-bundle Colab GPU run

Use a Colab runtime with **Runtime -> Change runtime type -> T4 GPU** (or another CUDA GPU). For the 2.0 GB combined archive, upload it to Google Drive first; do not use `files.upload()` because that loads the large file through the notebook session.

The fastest no-redownload path is the combined archive:
`IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip`. It contains the notebooks,
source/docs, evidence, all 5,273 verified BASE FITS files, and the local partial
run artifacts.

## 1. Confirm the GPU

```python
import torch
assert torch.cuda.is_available(), "Select a Colab GPU runtime before continuing"
print(torch.cuda.get_device_name(0))
```

## 2. Mount Drive and set the required controls

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/iris_silver_engine_kyros'
RUN_BASE = '1'
RUN_PHYSICS = '0'
RUN_DOWNSTREAM = '0'
FITS_SCOPE = 'base'
SEED = 2026
```

## 3. Put the complete bundle in Drive and extract only what is needed

Upload `IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip` to My Drive using the Drive web interface first. This cell copies only the two notebook files into the temporary Colab filesystem; it does not unpack the 1.5 GB FITS cache into RAM.

```python
from pathlib import Path
import shutil, zipfile

combined_zip = Path('/content/drive/MyDrive/IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip')
assert combined_zip.is_file(), 'Upload the combined archive to My Drive first'
bundle_dir = Path('/content/iris_bundle')
shutil.rmtree(bundle_dir, ignore_errors=True)
bundle_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(combined_zip) as archive:
    for name in ('IRIS_Colab_Training_2026-08-31-FITS-GATED.ipynb', 'IRIS_Colab_Training_2026-08-31-FITS-GATED-LIGHT.ipynb'):
        target = bundle_dir / name
        with archive.open(name) as source, target.open('wb') as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
print('Notebook files ready:', sorted(p.name for p in bundle_dir.glob('*.ipynb')))
```

### Reuse the already-acquired FITS cache (recommended)

The following streams only fits/ and evidence/ from the archive into the exact
Drive paths expected by the notebook. It skips files whose sizes already match,
so rerunning it is resumable. Set ACQUIRE_FITS = '0' in the notebook config
after this cell; the verified cache is then reused without another JSOC download.

```python
from pathlib import Path
import shutil, zipfile

combined_zip = Path('/content/drive/MyDrive/IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip')
work_root = Path(DRIVE_ROOT)
fits_target = work_root / 'fits_cache'
evidence_target = work_root / 'evidence-3e83a50d08f9'

def stream_prefix(prefix, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(combined_zip) as archive:
        entries = [info for info in archive.infolist()
                   if info.filename.startswith(prefix) and not info.is_dir()]
        for info in entries:
            relative = Path(info.filename[len(prefix):])
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                continue
            with archive.open(info) as source, destination.open('wb') as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)

stream_prefix('fits/', fits_target)
stream_prefix('evidence/', evidence_target)
print('Reused verified cache at:', fits_target, evidence_target)
print('FITS files:', len(list(fits_target.glob('*.fits'))))
```

## 4. Open the gated notebook inside the full bundle

In the Colab file browser, open:

`IRIS_Colab_Training_2026-08-31-FITS-GATED.ipynb`

The extracted notebook is at:

```python
from pathlib import Path
print(Path('/content/iris_bundle/IRIS_Colab_Training_2026-08-31-FITS-GATED.ipynb'))
```

Run the notebook cells in order. In its configuration cell, ensure these values are exactly:

```python
DRIVE_ROOT = '/content/drive/MyDrive/iris_silver_engine_kyros'
RUN_BASE = '1'
RUN_PHYSICS = '0'
RUN_DOWNSTREAM = '0'
FITS_SCOPE = 'base'
ACQUIRE_FITS = '0'  # use this when the verified cache was streamed above
SEED = 2026
```

When starting from an empty cache instead, leave ACQUIRE_FITS = '1'; the notebook will prompt for the currently registered JSOC email only when it needs to acquire files. The address is intentionally not written into notebook code, source, GitHub, or Drive.

## 5. Required execution order

Run all cells in this order:

1. Install/import dependencies.
2. Mount Drive and apply configuration.
3. Extract the complete bundle/source.
4. Verify the reused BASE FITS cache, or acquire the BASE FITS files from JSOC when ACQUIRE_FITS = '1'.
5. Run FITS verification and preflight.
6. Run the BASE generator/training cell.
7. Run the BASE evaluation/gate cells.

Do not start physics or downstream cells during this run. The notebook must report the real-FITS cache gate as passed before BASE training starts.

## 6. Verify the BASE result

After the notebook completes, inspect the Drive run directory:

```python
import glob, json, os

reports = glob.glob(
    DRIVE_ROOT + '/runs/base/audit/v2_manipulation_metrics.json',
    recursive=True,
)
print(reports)
if reports:
    print(json.dumps(json.load(open(reports[0])), indent=2))
```

The successful run must contain a BASE audit report and a passing BASE fidelity gate. A training checkpoint or loss log alone is not the final simulation result.

## 7. Run the confirmatory physics arms on a GPU

Do this only after the BASE report and its train-only gate have been reviewed.
The full runner uses 1,200 optimizer steps, writes checkpoints every 200 steps,
and runs the required `hj` and `hj_pil` generator arms. The `pil`-only arm is an
auxiliary factorial diagnostic, not a primary downstream arm.

Change only the stage controls in the notebook and rerun the controls,
preflight, and runner cells in order:

```python
RUN_BASE = '0'
RUN_PHYSICS = '1'
RUN_DOWNSTREAM = '0'
FITS_SCOPE = 'physics'
ACQUIRE_FITS = '1'  # use '0' only if the physics preflight already passes
```

The full bundle contains the BASE cache, but it is not evidence that every
physics-scope FITS payload is present. If `REAL_FITS_CACHE_PREFLIGHT_PASS` does
not appear, leave acquisition enabled and enter the registered JSOC email only
at the prompt. Never substitute PNGs or disable TLS verification.

The runner must produce, for both arms:

```text
runs/l2/outputs/hj/generator.pt
runs/l2/samples/hj/synthetic_manifest.csv
runs/l2/audit/v2_manipulation_metrics.json
runs/l3/outputs/hj_pil/generator.pt
runs/l3/samples/hj_pil/synthetic_manifest.csv
runs/l3/audit/v2_manipulation_metrics.json
```

Afterward, inspect the independent generic audit and the targeted geometry/PIL
manipulation checks. A lower training loss alone is not a physics pass. Select
coefficients using training-only evidence, and do not inspect any downstream
test metric while choosing a physics arm.

## 8. Run the frozen downstream matrix

Only after the physics arms satisfy the predeclared gates and Gate 0
administrative checks, acquire the complete `all` scope and run:

```python
RUN_BASE = '0'
RUN_PHYSICS = '0'
RUN_DOWNSTREAM = '1'
FITS_SCOPE = 'all'
ACQUIRE_FITS = '1'  # use '0' only after the all-scope preflight passes
```

The authoritative arm definitions are in
`V2_DOWNSTREAM_MATRIX_FREEZE_2026-08-27.md`. The runner enforces matched
synthetic exposure and trains the six frozen primary arms: `R`, `Rw`, `D`,
`L0`, `L2`, and `L3`. Here `R` is real-only/unweighted, `Rw` is the same
real-only set with `N_negative/N_positive` balanced positive weighting, `D` is
duplicated real positives, and `L0/L2/L3` map to BASE/HJ/HJ+PIL synthetic
positives. The `pil`-only arm is auxiliary and cannot replace `L3`. The runner
writes per-arm metrics and test predictions plus:

```text
runs/downstream/primary_metrics.csv
runs/downstream/primary_paired_tss_bootstrap.json
```

Thresholds are selected from validation TSS and the test partition is evaluated
once. Do not change thresholds, seeds, preprocessing, arm definitions, or
exclusions after test results are visible. A null or negative `L2/L3 - D`
result is a valid scientific result and must be reported as such.

Before accepting a downstream artifact, run the dependency-free structural
check from the bundle:

```bash
python reports/iris-base-2026-08-31/scripts/validate_primary_matrix.py \
  --artifact-dir runs/downstream
```

It rejects morphology-only summaries, incomplete arm sets, mismatched test
identities, and unequal positive exposure.

## Important

The previous local CPU run was intentionally stopped because the laptop was overheating. The local temporary cache/checkpoint is not automatically visible to Colab; Colab should use the Drive-backed cache produced by the gated notebook and reacquire any files not already in Drive.

The combined archive also includes `iris_local_resume_generator.py`, a laptop
continuation helper. It resumes from the saved step-400 checkpoint with two
PyTorch CPU threads and a preprocessed tensor cache; it is a backup/diagnostic
run, not a substitute for the CUDA Colab run.

`IRIS_LOCAL_POSTPROCESS.sh` can watch for the completed checkpoint and then
produce the matched samples and independent BASE audit automatically.

To run that helper locally after extracting the bundle/cache, use:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  /private/tmp/iris-venv/bin/python iris_local_resume_generator.py \
  --source-dir /private/tmp/iris_gated_run/source \
  --evidence-dir /private/tmp/iris_gated_run/evidence \
  --fits-source /private/tmp/iris_gated_run/fits \
  --checkpoint /private/tmp/iris_gated_run/work/runs/base/outputs/base/generator_step_400.pt \
  --out-dir /private/tmp/iris_gated_run/work/runs/base_local_resume/outputs \
  --prepared-cache /private/tmp/iris_gated_run/work/preprocessed_base \
  --target-steps 1200 --threads 2 --checkpoint-every 200
```
