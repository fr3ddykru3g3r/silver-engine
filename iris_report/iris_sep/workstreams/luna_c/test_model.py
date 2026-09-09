"""Unit tests for the Luna C prototype.

These tests construct only generated tensors.  They do not read a dataset,
benchmark manifest, partition identity, label, prediction, or locked-test
outcome.  The module is skipped automatically in environments without PyTorch;
Colab's pinned training environment should run it before a benchmark job.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iris_report.iris_sep.workstreams.luna_c.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from iris_report.iris_sep.workstreams.luna_c.model import (  # noqa: E402
    CausalConv1d,
    IRISSEPConfig,
    IRISSEPInputs,
    IRISSEPModel,
    ModalityInput,
    compute_task_losses,
)


def _config(*, missing_modality_dropout: float = 0.0) -> IRISSEPConfig:
    return IRISSEPConfig(
        magnetic_input_features=5,
        eruption_input_features=4,
        particle_input_features=3,
        lookback_steps=12,
        hidden_channels=8,
        embedding_dim=12,
        shared_dim=16,
        temporal_layers=2,
        kernel_size=3,
        dropout=0.0,
        onset_horizon_bins=6,
        missing_modality_dropout=missing_modality_dropout,
    )


def _inputs(batch_size: int = 4, steps: int = 12) -> IRISSEPInputs:
    torch.manual_seed(11)
    masks = {
        "magnetic": torch.ones(batch_size, steps),
        "eruption": torch.ones(batch_size, steps, 4),
        "particle": torch.ones(batch_size, steps),
    }
    # Exercise both a delayed prefix and feature-level missingness.
    masks["magnetic"][1, :3] = 0.0
    masks["eruption"][3, :, 0] = 0.0
    masks["particle"][2] = 0.0
    return IRISSEPInputs(
        magnetic=ModalityInput(
            values=torch.randn(batch_size, steps, 5),
            observed_mask=masks["magnetic"],
            time_since_observation_hours=torch.arange(steps).repeat(batch_size, 1).float(),
            available=torch.tensor([True, True, False, True]),
        ),
        eruption=ModalityInput(
            values=torch.randn(batch_size, steps, 4),
            observed_mask=masks["eruption"],
            time_since_observation_hours=torch.zeros(batch_size, steps),
            available=torch.tensor([True, True, True, True]),
        ),
        particle=ModalityInput(
            values=torch.randn(batch_size, steps, 3),
            observed_mask=masks["particle"],
            time_since_observation_hours=torch.full((batch_size, steps), 2.0),
            available=torch.tensor([True, True, False, True]),
        ),
    )


def test_forward_shapes_quantile_order_and_availability() -> None:
    model = IRISSEPModel(_config())
    model.eval()
    output = model(_inputs(), apply_missing_modality_dropout=False)

    assert output.primary_logit.shape == (4,)
    assert output.high_energy_logit.shape == (4,)
    assert output.peak_flux_log_quantiles.shape == (4, 3)
    assert output.onset_hazard_logits.shape == (4, 6)
    assert output.flare_activity_logit.shape == (4,)
    assert output.cme_activity_logit.shape == (4,)
    assert output.shared_embedding.shape == (4, 16)
    assert output.modality_embeddings.shape == (4, 3, 12)
    assert output.gate_weights.shape == (4, 3)
    assert output.modality_available.shape == (4, 3)
    assert torch.isfinite(output.shared_embedding).all()
    assert (output.peak_flux_log_quantiles[:, 1:] >= output.peak_flux_log_quantiles[:, :-1]).all()
    assert torch.allclose(output.gate_weights.sum(dim=1)[:2], torch.ones(2))
    assert torch.equal(output.modality_available[2], torch.tensor([False, True, False]))
    assert torch.allclose(output.gate_weights[2], torch.tensor([0.0, 1.0, 0.0]))


def test_mask_derived_availability_matches_gate_routing() -> None:
    original = _inputs()
    magnetic_mask = original.magnetic.observed_mask.clone()
    particle_mask = original.particle.observed_mask.clone()
    magnetic_mask[2] = 0.0
    particle_mask[2] = 0.0
    no_explicit_flags = IRISSEPInputs(
        magnetic=ModalityInput(
            original.magnetic.values,
            magnetic_mask,
            original.magnetic.time_since_observation_hours,
        ),
        eruption=ModalityInput(
            original.eruption.values,
            original.eruption.observed_mask,
            original.eruption.time_since_observation_hours,
        ),
        particle=ModalityInput(
            original.particle.values,
            particle_mask,
            original.particle.time_since_observation_hours,
        ),
    )
    model = IRISSEPModel(_config())
    model.eval()
    output = model(no_explicit_flags, apply_missing_modality_dropout=False)
    assert torch.equal(output.modality_available[2], torch.tensor([False, True, False]))
    assert torch.allclose(output.gate_weights[2], torch.tensor([0.0, 1.0, 0.0]))


def test_causal_convolution_prefix_is_invariant_to_future_changes() -> None:
    torch.manual_seed(5)
    convolution = CausalConv1d(3, 4, kernel_size=3, dilation=2)
    convolution.eval()
    sequence = torch.randn(2, 3, 12)
    changed_future = sequence.clone()
    changed_future[:, :, 7:] += 100.0
    original = convolution(sequence)
    altered = convolution(changed_future)
    torch.testing.assert_close(original[:, :, :7], altered[:, :, :7])


def test_explicit_missing_modality_keep_mask_zeroes_gates() -> None:
    model = IRISSEPModel(_config())
    model.eval()
    keep = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [True, True, True],
        ]
    )
    output = model(
        _inputs(),
        modality_available=torch.ones(4, 3, dtype=torch.bool),
        modality_keep_mask=keep,
        apply_missing_modality_dropout=False,
    )
    assert torch.equal(output.modality_available, keep)
    assert torch.allclose(output.gate_weights[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(output.gate_weights[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(output.gate_weights[2], torch.zeros(3))
    assert torch.allclose(output.gate_weights[3].sum(), torch.tensor(1.0))


def test_train_time_missing_modality_hook_keeps_an_available_feed() -> None:
    torch.manual_seed(23)
    model = IRISSEPModel(_config(missing_modality_dropout=0.99))
    model.train()
    output = model(_inputs(), generator=torch.Generator().manual_seed(9))
    available = output.modality_available
    assert (available.sum(dim=1) >= 1).all()


def test_task_losses_are_finite_and_differentiable() -> None:
    model = IRISSEPModel(_config())
    model.train()
    output = model(_inputs(), apply_missing_modality_dropout=False)
    targets = {
        "primary": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "high_energy": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "peak_flux_log": torch.tensor([1.0, 2.0, 0.5, 1.5]),
        "onset_hazard": torch.zeros(4, 6),
        "flare_activity": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "cme_activity": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    losses = compute_task_losses(output, targets)
    assert set(losses) == {
        "primary",
        "high_energy",
        "peak_flux_log",
        "onset_hazard",
        "flare_activity",
        "cme_activity",
        "total",
    }
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_checkpoint_round_trip_restores_model_and_metadata(tmp_path: Path) -> None:
    config = _config()
    model = IRISSEPModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    model.train()
    output = model(_inputs(), apply_missing_modality_dropout=False)
    output.primary_logit.mean().backward()
    optimizer.step()
    scheduler.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint_path = save_checkpoint(
        tmp_path / "resume.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=7,
        epoch=2,
        best_metric=0.123,
        history=[{"step": 7, "loss": 0.5}],
        extra={"synthetic_smoke": True},
    )
    assert checkpoint_path.exists()

    restored_model = IRISSEPModel(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
    restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
    state = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        restore_rng=False,
    )
    assert state.step == 7
    assert state.epoch == 2
    assert state.best_metric == pytest.approx(0.123)
    assert state.config == config.to_dict()
    assert state.extra["synthetic_smoke"] is True
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
