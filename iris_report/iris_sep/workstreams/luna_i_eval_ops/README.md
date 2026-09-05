# Luna I evaluation and operator contracts

This workstream contains dependency-light, synthetic-only evaluation code. It
enforces dedicated calibration and threshold roles, computes the declared
classification and probability metrics, implements the predeclared
matched-detection diagnostic, and resamples complete episode/quiet-block units
for paired TSS uncertainty.

The operator response is advisory only. It exposes calibration/policy/model
identifiers, freshness, missing inputs, uncertainty, an evidence hash, and an
explicit abstention path. Secondary forecasts remain null until independently
validated. It never emits a spacecraft command.

Run locally:

```bash
python3 -m unittest iris_report.iris_sep.workstreams.luna_i_eval_ops.test_eval_ops_unittest -v
```
