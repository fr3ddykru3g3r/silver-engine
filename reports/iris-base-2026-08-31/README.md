# IRIS BASE evidence report

`report.pdf` is the compiled 88-page report for the completed IRIS BASE run and its separately labeled 100-step HJ/PIL screening follow-up. It covers the verified FITS acquisition, chronology and label audits, the local step-400 → step-1200 resume, 128 generated arrays, independent generic/geometry/PIL audits, the corrected metric negative control, an alternate sampler seed, selective-destruction controls, the v2 physics self-test, three cached physics screens, cluster-bootstrap uncertainty intervals, an independent thirteen-feature line-of-sight magnetic/spectral/topology proxy audit, a group-held-out real-versus-generated classifier audit, a latitude-conditioned proxy audit, connected-group gate-stability bootstrap analysis, exact-hash and nearest-neighbor memorization checks, the mathematical HJ/PIL proxy definitions, five offline protocol self-tests, 2D/3D figures, OBJ visualization meshes, limitations, and the exact future gates that remain open.

The primary scientific result is deliberately bounded: the BASE broad calibration gate passes for seed 2026, but the corrected generic distance is 7.681 times the real split-half p90 reference and the joint gate passes in only 0.508 of connected-group bootstrap replicates. A second sampler seed fails that gate at 11.055. The 100-step HJ, PIL, and HJ+PIL screens activate their intended losses but fail the broad generic gate at 20.855, 21.521, and 20.081 times the real split-half p90 reference. The independent secondary proxy audit finds a robust-standardized distance ratio of 60.52 against its own real split-half p90. No forecasting result or causal physics claim is supported.

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
/Users/Kyrosah/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py \
  iris_report/report.tex
```

The raw FITS payloads are not duplicated inside this small report directory. They are included in the parent archive `IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip`, along with the source bundle, evidence, checkpoints, generated arrays, alternate seed, and destruction-control outputs.

`artifacts/public_report_metadata.json` and `artifacts/public_inventory_summary.json` are sanitized publication manifests with workstation-specific paths replaced by symbolic roots. The unsanitized receipts are retained locally for audit purposes. The `scripts/` directory also includes the exact physics-screen trainer, sampler, fixed evaluator, and v2 self-test entry points used for the follow-up. `artifacts/completion_audit.json` is the machine-readable requirement ledger; it records which requested deliverables are verified and which remain open because of the unrun long physics arms. The report branch and the 2.13 GB full archive release asset are now preserved in `https://github.com/fr3ddykru3g3r/silver-engine` and were checked against the local SHA-256 receipt.

The OBJ files are scalar-field visualization meshes: x/y are in Mm and z is a magnetogram value or ensemble standard deviation. They are not coronal field-line or volume reconstructions.

The auxiliary `figures/two_sample_classifier.svg` is a dependency-free vector plot of the classifier fold AUCs and leading scalar-proxy coefficients. Its high scalar AUC is a distinguishability result, not a forecast or physical-realism claim.

`artifacts/offline_protocol_selftests.json` records five post-update offline self-tests; all passed. They validate infrastructure and policy primitives only and do not add a training or forecasting result.

`artifacts/conditional_proxy_diagnostics.json` records the train-only latitude-conditioned audit. The well-populated bands retain materially different proxy ratios, while the sparse `(30,90]` band is explicitly caveated and is not used as a standalone conclusion.

`artifacts/gate_stability_diagnostics.json` records the connected-group bootstrap stability audit. The joint pass fraction is 0.5075 because the corrected multivariate distance is the limiting criterion; this is finite-sample uncertainty, not a scientific-validity probability.

`artifacts/memorization_diagnostics.json` records the low-dimensional duplicate/collapse check. It found zero exact normalized-array duplicates; the generated nearest-neighbor distances were not suspiciously close to real-to-real distances.
