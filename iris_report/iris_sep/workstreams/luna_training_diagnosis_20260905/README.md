# IRIS-SEP training diagnosis — 2026-09-05

This is a bounded structural audit of the pinned V3 development table. The script verifies the V3 CSV and manifest SHA-256 values, binds the 98-feature schema, and filters to `role=train` before calculating diagnostics. It does not fit a model, tune a parameter, inspect outer validation values or metrics, or access a locked test file.

The training slice contains 7,812 windows across 1,382 isolated units from 1986-02-03 through 2013-05-21. There are 1,318 positive windows (16.87%), 6,494 quiet windows, 135 event units, and 1,247 quiet units. Quarterly prevalence is highly uneven: for example, 1986 Q1 is 14/34 positive windows (41.18%), while 1986 Q2 and Q3 contain no positive windows; late quarters include 2012 Q4 at 2/92 (2.17%), 2013 Q1 at 6/90 (6.67%), and 2013 Q2 at 17/50 (34.00%). This is support for chronological coverage imbalance, not a performance result.

The clearest structural limitation is era-dependent SHARP feature availability. In the fixed training-era split, the early half (1986-02-03 through 1999-09-27; 4,002 rows) has 100% missingness for the ten most-missing SHARP aggregates, including `sharp_MEANPOT_*`, `sharp_MEANJZH_min`, `sharp_TOTPOT_*`, `sharp_LAT_MIN_min`, and `sharp_SAVNCPP_*`. In the late half (1999-09-28 through 2013-05-21; 3,810 rows), those same fields remain 71.89% missing. The coarse modality availability flag is 100% present because label-like columns such as `sharp_label`, `flare_label`, and `CME_label` are populated; it should not be read as evidence that the underlying magnetic measurements are complete. The table contains no particle-context branch.

The audit found no exact duplicate predictor groups and no numeric feature that is constant among its observed values. The label-like feature counts are `sharp_label`: 1,071 ones / 6,741 zeros; `flare_label`: 7,690 ones / 122 zeros; `CME_label`: 466 ones / 7,346 zeros. These are predictor-structure observations, not claims that these fields are valid causal proxies.

The principal failure hypotheses to test in a future, predeclared training protocol are: a single model may learn calendar/era or data-availability regimes instead of stable physical signal; imputation or missingness masks may dominate the magnetic branch because large SHARP blocks are unavailable; and the sparse, uneven quarterly event support may make chronological learning unstable. These are hypotheses suggested by the training table and require targeted train-only ablations or a fresh, properly audited cohort; they are not causal findings. The label remains the publisher’s legacy future operational-window label, explicitly not the audited NEW-crossing target.

Re-run:

```bash
python3 iris_report/iris_sep/workstreams/luna_training_diagnosis_20260905/run_training_diagnosis.py
```

Machine-readable output: [training_diagnosis.json](./training_diagnosis.json).
