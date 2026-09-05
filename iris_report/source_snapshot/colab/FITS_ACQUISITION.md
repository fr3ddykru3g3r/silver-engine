# Colab FITS acquisition gate

The evidence archive contains the historical metadata and labels, not the
binary HMI/SHARP magnetograms.  The training runner therefore requires a
separate local FITS cache and fails before model construction if that cache is
missing or incomplete.

## One-time setup for each student

1. Register the student's email for JSOC data export using the official JSOC
   registration page.
2. In the notebook, keep the student's Drive folder separate:
   `iris_silver_engine_kyros` or `iris_silver_engine_lokesh`.
3. Enter the registered email with `getpass`; do not write it into the
   notebook, GitHub, source bundle, or results archive.
4. Run the acquisition cell with `FITS_SCOPE = 'base'`. It is resumable and
   writes validated files under the Drive folder's `fits_cache/` directory.
5. Run the cache preflight cell. Only after it prints
   `REAL_FITS_CACHE_PREFLIGHT_PASS` should `RUN_BASE = '1'` be used.

The acquisition uses the `drms` export client rather than assuming that a raw
`http://jsoc.stanford.edu/SUM...` URL is downloadable. It requests exact
manifest records in small batches. The default notebook transport is the
registered-email `url`/`fits` export, which avoids relying on direct SUM URL
access. The lower-load `url_quick`/`as-is` mode is also supported; if it returns
403, switch back to `url`/`fits`. Never disable TLS verification or silently
substitute PNGs for FITS data.

`FITS_SCOPE = 'base'` covers the deterministic generator training and
train-only real-image fidelity audit. After the base gate passes, `physics`
can reuse that cache. Before `RUN_DOWNSTREAM = '1'`, acquire with
`FITS_SCOPE = 'all'` so the real validation and test samples are present.

The produced `acquisition_report.json`, `fits_cache_manifest.csv.gz`, and
`acquisition_log.jsonl` are provenance artifacts. A successful acquisition is
not a forecasting result; it only unlocks the next reproducible experiment.
