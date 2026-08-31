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

## 6. Verify the final result

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
