# Historical model-development freeze

**Effective baseline:** `7bd7f0659b8ce2f4310d5c83f7d81b6db439c40b`  
**Effective date:** 2026-09-11

For the IRIS submission claim, retrospective model development is closed.

The following may not be changed in response to the inspected historical scores and then substituted back into the same historical claim:

- model architecture or learned parameters;
- feature families or feature-selection policy;
- thresholds or alert policy;
- fit/calibration/threshold/score boundaries;
- episode mapping or onset/persistence definitions;
- matched inferential cohort rules;
- bootstrap unit, seed or decision rule;
- negative-block construction;
- chosen primary contrasts.

Corrections for demonstrable implementation/evidence bugs remain permissible only if they are transparently logged, preserve old files, document whether predictions or model fitting changed, and do not use protected outcomes. They are corrections, not opportunities to optimize the result.

Any scientifically interesting follow-up must receive a **new study name/preregistration before its result is inspected** and must remain clearly separated from the frozen historical submission result.

This freeze does not prevent student rewriting, citation verification, figure typography/layout changes that do not alter data, forms completion, oral-defense preparation, fail-closed prospective infrastructure work that does not inspect protected outcomes, or preparation of unsent stakeholder-review materials.

The protected post-2025 outcome pool must not be used to rescue statistical power, select a model, select a threshold, tune missing-data limits or adjust the narrative.
