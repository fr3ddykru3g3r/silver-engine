# IRIS BASE evidence report — 2026-08-31

This directory is the compact publication package for the completed IRIS BASE experiment and its separately labeled physics screening follow-up. The main artifact is the 73-page `report.pdf`; `report.tex`, figures, tables, scalar-field OBJ meshes, audit scripts, sanitized metadata, and the lightweight notebook are included alongside it.

## Result boundary

- 648,926 manifest rows were audited; 5,273 FITS files were acquired and verified for the BASE stage.
- The final local resume reached step 1,200 and produced 128 normalized synthetic arrays.
- The predeclared corrected generic calibration gate passed for sampler seed 2026 at 7.681× the real split-half p90 reference.
- An inference-only sampler-seed check with seed 2027 failed the same gate at 11.055×.
- The v2 physics self-test passed; 100-step HJ, PIL, and HJ+PIL screens activated their intended losses but failed the broad generic gate at 20.855×, 21.521×, and 20.081×.
- Cluster-bootstrap descriptor intervals over 125 real and 64 synthetic connected-region groups quantify which median shifts are stable under group resampling.
- An incomplete seed-2028 CPU sampling attempt is recorded and explicitly excluded; it produced no manifest or audit result.
- Physics-constrained training and downstream flare forecasting were intentionally not executed. No forecast or causal physics claim is supported.

## Data and privacy boundary

The 1.8 GB raw-data archive is intentionally not committed here because standard Git hosting is not an appropriate transport for that archive. The complete local bundle is named `IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip` and contains the source, evidence, verified FITS cache, checkpoints, generated arrays, alternate-seed outputs, and destruction-control outputs. The GitHub package contains derived outputs and provenance receipts, not a replacement for that local archive. The machine-readable completion ledger is `artifacts/completion_audit.json`; it separates verified deliverables from the still-open long physics arms, forecasting, remote preservation, and cleanup gates.

No JSOC email address or credential is stored in this publication package.

## Rebuild notes

The scripts are portable and accept `IRIS_RUN_ROOT` and `IRIS_REPORT_DIR`. They require the matching IRIS source/evidence/cache/run tree and the Python dependencies used for the local audit. The sanitized JSON manifests replace workstation-specific paths with symbolic roots.
