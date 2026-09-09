# SOL bounded adversarial review — immutable inference bundle

## Scope and independence boundary

This workstream is isolated from model selection, score inspection, and locked
evaluation. It is a source-only adversarial review of the immutable admission-V2
inference envelope. It does not train a model, inspect locked identities/outcomes,
fetch a mixed training/test table, send external messages, or change the frozen
scientific gate.

"Independent" here means an isolated red-team write/test scope with assumptions
separate from the implementation path; it is not a claim of an external human or
third-party security audit.

## Threats to test

1. Replace the outer bundle bytes while retaining the old trusted bundle digest.
2. Mutate a bound transformed-feature or model-output array while leaving an
   inconsistent inner array digest, even if the outer JSON anchor is recomputed.
3. Change source-revision metadata independently from forecast-time freshness.
4. Change admission support boundaries or allowed source revisions.
5. Change calibration method/parameter or calibration identifier.
6. Change an operator threshold or threshold policy identifier.
7. Change model version or input schema binding.
8. Remove or mutate the embedded evidence receipt.
9. Inject NaN/Inf or an empty bound array.
10. Recompute or replace a derived probability without matching the bound raw
    model output and calibration.
11. Reuse an otherwise valid scientific receipt after changing a static policy,
    calibration, threshold, model, or schema contract.

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
mismatched static binding.

The external bundle SHA-256 remains the trust anchor for dynamic inference
content such as exact feature-array bytes, model-output bytes, source-revision
snapshot, and request metadata. If a caller deliberately replaces the bundle and
also establishes a new trusted external SHA, that is a new bundle, not tampering
that this file format can distinguish by itself. Therefore the eventual trusted
bundle digest and scientific evidence digest still require an external provenance
chain. This is an explicit boundary, not operational certification.

## Executed/source-only test matrix

`tests/test_inference_bundle.py` now covers:

- valid replay from bundle bytes only;
- outer-byte tamper against the retained trust anchor;
- inner model-output byte tamper with a recomputed outer anchor but stale inner
  array digest;
- source-revision snapshot/freshness disagreement;
- runtime threshold/policy mutation;
- calibration mutation;
- admission support-boundary mutation;
- model-version and input-schema mutation;
- embedded evidence-receipt mutation;
- derived calibrated-probability mutation;
- unbound scientific evidence;
- nonfinite arrays; and
- empty feature/output arrays.

The separate compact replay tests do not share this workstream's integrity
assumptions. The compact replay also now verifies the original diagnostic
preregistration and source hashes before interpreting a failure stage.

Disposition: these are bounded source-level adversarial cases. Real pilot
admission still requires a receipt from the eventual frozen
model/calibration/threshold run, verified source/latency provenance, and an
externally retained trust anchor for each immutable inference bundle.
