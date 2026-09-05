# SOL bounded adversarial review — immutable inference bundle

## Scope

This workstream is independent of model selection and locked evaluation. It is a
source-only adversarial review of the immutable admission-V2 inference envelope.
It must not train a model, inspect locked identities/outcomes, fetch a mixed
training/test table, send external messages, or change the frozen scientific
gate.

## Threats to test

1. Replace the outer bundle bytes while retaining the old trusted bundle digest.
2. Mutate a bound transformed-feature or model-output array and recompute only
   outer JSON hashes.
3. Change source-revision metadata independently from forecast-time freshness.
4. Change admission support boundaries or allowed source revisions.
5. Change calibration method/parameter or calibration identifier.
6. Change an operator threshold or threshold policy identifier.
7. Change model version or input schema binding.
8. Remove or mutate the embedded evidence receipt.
9. Inject NaN/Inf or an empty bound array.
10. Recompute derived probability without matching the bound raw model output.

Every mutation must either be rejected before serialization or fail closed. A
successful synthetic test establishes only software-integrity behavior for the
tested cases; it is not SEP forecast skill, operational certification, novelty,
or competitor superiority.

## Review finding and correction

The first envelope design had a real integrity gap: the external bundle SHA-256
made the entire envelope immutable for a trusted caller, but the embedded
scientific evidence receipt did not independently bind the admission policy,
calibration parameter, and threshold snapshot. A caller able to establish a new
outer trust anchor could therefore create a different internally consistent
inference envelope around the same evidence receipt.

The corrected design adds `static_inference_binding_sha256`. The evidence receipt
must contain that digest, computed over the frozen admission policy, runtime
policy, logit-intercept calibration parameter, operator thresholds, model
version, and input schema. Bundle construction and replay reject a missing or
mismatched static binding. This does not establish publisher authenticity: the
trusted receipt and bundle digests still need an external provenance chain.

## Executed source-only checks

The integration test module `tests/test_inference_bundle.py` covers valid replay,
outer tamper, inner-array tamper with a recomputed outer anchor, source-revision
mismatch, threshold/policy mutation, calibration mutation, unbound evidence, and
nonfinite arrays. The separate compact-layer replay tests do not share this
workstream's integrity assumptions.

Disposition: the source-only adversarial cases pass in the continuation runtime.
The envelope remains an offline research artifact. Real pilot admission still
requires a receipt from the eventual frozen model/calibration/threshold run and
verified source/latency provenance.
