# Luna G data-pipeline contract

This dependency-light workstream implements the integrity behavior required
before a real SEP table may enter tuning. It accepts synthetic records only and
cannot open protected data.

It tests canonical issue identities, observation/publication timestamps,
source-latency limits, new crossings versus already-enhanced intervals,
physical episode and quiet-block grouping, chronological whole-unit roles,
24-hour purging, train-only imputation/scaling, feature masks, and canonical
immutable manifests.

The production adapter must translate an approved, hash-pinned source into
these concepts and produce cohort, partition, feature, exclusion, and transform
receipts. SEPVAL identities and outcomes remain outside tuning.

Run locally:

```bash
python3 -m unittest iris_report.iris_sep.workstreams.luna_g_data_pipeline.test_pipeline_unittest -v
```
