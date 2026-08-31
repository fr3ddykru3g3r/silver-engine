# IRIS BASE evidence report

`report.pdf` is the compiled 77-page report for the completed IRIS BASE run and its 100-step HJ/PIL screening follow-up. It covers the verified FITS acquisition, chronology and label audits, the local step-400 → step-1200 resume, 128 generated arrays, independent generic/geometry/PIL audits, the corrected metric negative control, an alternate sampler seed, selective-destruction controls, the v2 physics self-test, three cached physics screens, cluster-bootstrap intervals, the independent thirteen-feature line-of-sight magnetic/spectral/topology proxy audit, 2D/3D figures, OBJ visualization meshes, limitations, and the exact future gates that remain open.

The primary scientific result is deliberately bounded: the BASE broad calibration gate passes for seed 2026, but the corrected generic distance is 7.681 times the real split-half p90 reference. A second sampler seed fails that gate at 11.055. The 100-step HJ, PIL, and HJ+PIL screens activate their intended losses but fail the broad generic gate at 20.855, 21.521, and 20.081. No forecasting result or causal physics claim is supported.

## Rebuild the report

The report-generation script reads the existing evidence/cache/run roots and writes only inside this directory:

```bash
export IRIS_RUN_ROOT=/private/tmp/iris_gated_run
export IRIS_REPORT_DIR="$PWD/iris_report"
MPLCONFIGDIR=/private/tmp/mplconfig \
XDG_CACHE_HOME=/private/tmp/cache \
/private/tmp/iris-venv/bin/python \
  iris_report/scripts/generate_analysis.py
```

The default temporary roots expected by the script are:

- `/private/tmp/iris_gated_run/source`
- `/private/tmp/iris_gated_run/evidence`
- `/private/tmp/iris_gated_run/fits`
- `/private/tmp/iris_gated_run/work/runs/base_local_resume`

Compile with the bundled LaTeX helper:

```bash
COMPILE_LATEX \
  iris_report/report.tex
```

The raw FITS payloads are not duplicated inside this small report directory. They are included in the parent archive `IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip`, along with the source bundle, evidence, checkpoints, generated arrays, alternate seed, and destruction-control outputs.

`artifacts/public_report_metadata.json` and `artifacts/public_inventory_summary.json` are sanitized publication manifests with workstation-specific paths replaced by symbolic roots. The unsanitized receipts are retained locally for audit purposes.

The `scripts/` directory includes the exact patched physics-screen trainer, sampler, fixed evaluator, and v2 self-test entry points used for the follow-up.

The OBJ files are scalar-field visualization meshes: x/y are in Mm and z is a magnetogram value or ensemble standard deviation. They are not coronal field-line or volume reconstructions.
