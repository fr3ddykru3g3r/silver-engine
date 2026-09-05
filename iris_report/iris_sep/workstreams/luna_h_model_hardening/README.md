# Luna H model hardening

This workstream fixes model semantics before real-data training. It contains no
dataset paths, identities, labels, or locked-test outcomes.

The default scientific objective is primary SEP occurrence only. Optional peak
and onset tasks require explicit activation, validity masks, and later
validation-side ablations. Peak loss is conditional on a valid positive event;
onset uses a discrete-time right-censored hazard likelihood. Feature-level
masks are preserved, missing-modality fallback is sampled without fixed expert
priority, the five-seed summary uses the frozen median policy, and an all-inputs
missing row produces an abstention signal.

Local verification:

```bash
python3 -m unittest iris_report.iris_sep.workstreams.luna_h_model_hardening.test_contract_unittest -v
```

`colab_runtime_test.py` is a generated-tensor-only PyTorch check for the later
GPU runtime. Passing it is not a forecasting result.
