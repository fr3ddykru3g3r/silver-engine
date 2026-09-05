# Start here: the corrected Colab workflow

The failure you saw was a real data failure, not a model failure. The
metadata/evidence archive has no binary magnetogram images. The corrected
workflow therefore has three explicit stages:

1. Open `IRIS_Colab_Training_2026-08-31-FITS-GATED.ipynb`.
2. In the controls cell, choose one Drive root:
   - Kyros: `iris_silver_engine_kyros`
   - Lokesh: `iris_silver_engine_lokesh`
3. Run the cells through the acquisition cell. When prompted, enter that
   student's registered JSOC export email. It is used in memory only.
4. Wait for `acquisition_report.json` to say `PASS` and for the preflight cell
   to print `REAL_FITS_CACHE_PREFLIGHT_PASS`.
5. Run the runner cell with BASE enabled.

The notebook contains the matching source bundle and the metadata/evidence
archive, but it cannot contain the historical FITS images because those are
downloaded only through the approved JSOC export route. Acquisition is
resumable and stores the real FITS cache in Drive.

If `url`/`fits` export is rejected, keep the registered email and change the
controls to `JSOC_EXPORT_METHOD = 'url_quick'` and
`JSOC_EXPORT_PROTOCOL = 'as-is'`, then rerun acquisition. If both routes fail,
save the exact `ACQUISITION_FAILED` message; do not disable TLS checks or use
PNG screenshots as a replacement.

Do not enable physics or downstream until BASE has passed its independent
train-only fidelity gate. Before downstream, change `FITS_SCOPE` to `all` so
validation and test FITS files are acquired too.
