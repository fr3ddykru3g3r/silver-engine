# Luna D — exact AARP/HMI identity bridge

This workstream is an isolated, standard-library-only contract for the
optional AIA feature expert. It does not read the AARP NetCDF, FITS files,
SEPVAL/CLEAR outcomes, or any locked-test artifact.

## Decision rule

An AARP row may enter HMI late fusion only when an explicitly supplied
`authoritative_crosswalk` proves all of the following:

* the canonical AARP filename identity and AARP region number;
* the exact HMI `(harpnum, t_rec_tai)` key;
* an authoritative exact-observation time proof;
* an authoritative region crosswalk proof; and
* source URI/version, artifact SHA-256, row identifier, row SHA-256, and
  retrieval-time provenance for both source manifests and the crosswalk.

Date-only, nearest-time, rounded-time, row-order, filename-substring, and
numeric AARP/HARPNUM matches are rejected. Duplicate, ambiguous, unmatched,
or malformed rows remain external-only; no neighboring value is copied.

`identity_bridge.py` exposes `evaluate_bridge()` and receipt builders.
`aia_pretraining.py` defines a small, label-free AIA masked-reconstruction
interface. It permits only external AIA pretraining partitions and explicitly
prohibits downloading the approximately 9.5 TB raw AIA archive. Its encoder
weights/normalization statistics may be transferred later, but SEP heads,
calibration, thresholds, and labels may not.

Run the synthetic tests from the repository root:

```text
python3 -m unittest iris_report.iris_sep.workstreams.luna_d.test_identity_bridge \
  iris_report.iris_sep.workstreams.luna_d.test_aia_pretraining
```

The tests use only in-memory toy metadata and arrays. The current recorded
state has no authoritative crosswalk, so `current_bridge_decision.json`
intentionally records `REJECT` and disables AIA/HMI fusion.
