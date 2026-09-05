# Luna C — compact IRIS-SEP model prototype

This isolated workstream provides a small PyTorch model for later integration
with the frozen IRIS-SEP benchmark. It is an implementation prototype, not a
trained model and not evidence of a forecasting result. It does not open data
files or contain any locked-test identities, outcomes, thresholds, or
predictions.

## Architecture

Three independent experts consume causal hourly histories:

1. `magnetic` — SHARP/SMARP feature sequence;
2. `eruption` — flare, GOES XRS, and CME feature sequence;
3. `particle` — pre-event proton-context sequence.

Each expert appends an observation-mask channel and a normalized
`time_since_observation_hours` channel, applies two compact left-padded
residual temporal-convolution blocks, and pools a fixed-size embedding. The
pooling is mask-weighted and returns a neutral contribution for an entirely
absent history. A gated late-fusion layer receives all three embeddings plus
feed-availability bits; unavailable experts receive exactly zero gate weight.
The fused representation feeds task-specific heads.

The default heads are:

- `primary_logit`: new `>10 MeV, ≥10 pfu` occurrence logit;
- `high_energy_logit`: new `>100 MeV, ≥1 pfu` occurrence logit;
- `peak_flux_log_quantiles`: ordered `(p10, p50, p90)` estimates in log-pfu
  space;
- `onset_hazard_logits`: one logit per future hour/bin;
- `flare_activity_logit` and `cme_activity_logit`: auxiliary activity logits.

The model returns logits, not operating thresholds or calibrated probabilities.
Threshold selection and calibration remain trainer/evaluation responsibilities
under the frozen benchmark contract.

## Explicit input contract

```python
from iris_report.iris_sep.workstreams.luna_c import (
    IRISSEPConfig, IRISSEPInputs, IRISSEPModel, ModalityInput,
)

config = IRISSEPConfig(
    magnetic_input_features=16,
    eruption_input_features=32,
    particle_input_features=8,
    lookback_steps=24,
)
model = IRISSEPModel(config)
inputs = IRISSEPInputs(
    magnetic=ModalityInput(values, observed_mask, time_since_hours, available),
    eruption=ModalityInput(values, observed_mask, time_since_hours, available),
    particle=ModalityInput(values, observed_mask, time_since_hours, available),
)
output = model(inputs, apply_missing_modality_dropout=False)
```

For every modality, `values` is `[B, T, F]`; `observed_mask` is `[B, T]` or
feature-level `[B, T, F]`; and `time_since_observation_hours` is `[B, T]` or
`[B, T, 1]`. Values are expected to be finite and already transformed by the
train-only data adapter. The model masks missing values but does not silently
impute them. All modalities must share `B` and `T`; `T` must equal the
configured `lookback_steps`.

`available` is an optional boolean `[B]` feed-availability flag. If omitted,
availability is derived from the mask. For deterministic validation/replay,
pass `apply_missing_modality_dropout=False`. During training, the configured
`missing_modality_dropout` samples a keep mask while ensuring at least one
available modality remains when any feed exists. A trainer can provide an
auditable deterministic `[B, 3]` `modality_keep_mask` instead.

## Restart safety

`checkpoint.py` atomically saves and restores model, optimizer, scheduler, AMP
scaler, Python RNG, NumPy RNG (when installed), and CPU/CUDA PyTorch RNG state:

```python
from iris_report.iris_sep.workstreams.luna_c import load_checkpoint, save_checkpoint

save_checkpoint("latest.pt", model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, step=step, epoch=epoch)
state = load_checkpoint("latest.pt", model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler)
```

Use a checkpoint path in the run's own output directory. The helper does not
write dataset content or test predictions.

## Local/generated smoke checks

The test suite uses generated tensors only:

```bash
python iris_report/iris_sep/workstreams/luna_c/verify_static.py
python -m unittest discover -s iris_report/iris_sep/workstreams/luna_c -p 'test_unittest.py'
python -m pytest iris_report/iris_sep/workstreams/luna_c/test_model.py
python iris_report/iris_sep/workstreams/luna_c/smoke_test.py
```

The static script is dependency-free. The stdlib test entry point skips cleanly
when PyTorch is unavailable. The smoke test's only claim is that shapes, finite
synthetic losses, missing-modality routing, and checkpoint metadata round-trip
successfully in a PyTorch environment. It is not a benchmark and must not be
cited as one.
