"""Small generated-tensor smoke test for the Luna C prototype.

This is an integration sanity check only.  It deliberately reports no
forecast metric and makes no scientific claim.  It never opens a data path or
uses benchmark/locked-test records.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import torch

try:  # Support both package execution and ``python smoke_test.py`` in Colab.
    from .checkpoint import load_checkpoint, save_checkpoint
    from .model import IRISSEPConfig, IRISSEPInputs, IRISSEPModel, ModalityInput, compute_task_losses
except ImportError:  # pragma: no cover - exercised by direct script execution
    from checkpoint import load_checkpoint, save_checkpoint
    from model import IRISSEPConfig, IRISSEPInputs, IRISSEPModel, ModalityInput, compute_task_losses


def _make_inputs(batch_size: int, steps: int) -> IRISSEPInputs:
    values = {
        "magnetic": torch.randn(batch_size, steps, 4),
        "eruption": torch.randn(batch_size, steps, 3),
        "particle": torch.randn(batch_size, steps, 2),
    }
    masks = {
        "magnetic": torch.ones(batch_size, steps),
        "eruption": torch.ones(batch_size, steps, 3),
        "particle": torch.ones(batch_size, steps),
    }
    masks["magnetic"][0, :2] = 0.0
    masks["eruption"][1, :, 1] = 0.0
    masks["particle"][2] = 0.0
    times = torch.arange(steps, dtype=torch.float32).repeat(batch_size, 1)
    availability = {
        "magnetic": torch.tensor([True, True, True, False]),
        "eruption": torch.tensor([True, True, True, True]),
        "particle": torch.tensor([True, True, False, False]),
    }
    return IRISSEPInputs(
        magnetic=ModalityInput(values["magnetic"], masks["magnetic"], times, availability["magnetic"]),
        eruption=ModalityInput(values["eruption"], masks["eruption"], times, availability["eruption"]),
        particle=ModalityInput(values["particle"], masks["particle"], times, availability["particle"]),
    )


def _make_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except (AttributeError, TypeError):  # pragma: no cover - old torch fallback
        return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def run_smoke_test() -> dict[str, object]:
    """Run two optimizer steps and a checkpoint round trip on synthetic tensors."""

    torch.manual_seed(20260904)
    config = IRISSEPConfig(
        magnetic_input_features=4,
        eruption_input_features=3,
        particle_input_features=2,
        lookback_steps=8,
        hidden_channels=8,
        embedding_dim=12,
        shared_dim=16,
        temporal_layers=2,
        kernel_size=3,
        dropout=0.05,
        onset_horizon_bins=8,
        missing_modality_dropout=0.20,
    )
    model = IRISSEPModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = _make_scaler()
    inputs = _make_inputs(batch_size=4, steps=config.lookback_steps)
    targets = {
        "primary": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "high_energy": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "peak_flux_log": torch.tensor([0.5, 1.0, 0.2, 1.3]),
        "onset_hazard": torch.zeros(4, config.onset_horizon_bins),
        "flare_activity": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "cme_activity": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }

    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = compute_task_losses(output, targets)["total"]
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if not torch.isfinite(loss).item():
            raise AssertionError("synthetic smoke loss is not finite")

    with tempfile.TemporaryDirectory(prefix="iris_sep_luna_c_smoke_") as directory:
        checkpoint_path = save_checkpoint(
            Path(directory) / "latest.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=2,
            epoch=1,
            history=[{"step": 1}, {"step": 2}],
            extra={"synthetic_smoke": True},
        )
        restored = IRISSEPModel(config)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
        restored_scaler = _make_scaler()
        state = load_checkpoint(
            checkpoint_path,
            restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            restore_rng=False,
        )
        restored.eval()
        restored_output = restored(inputs)
        if restored_output.primary_logit.shape != (4,):
            raise AssertionError("restored model returned an unexpected primary shape")
        if state.step != 2 or state.epoch != 1:
            raise AssertionError("checkpoint metadata did not round-trip")

    return {
        "status": "PASS",
        "synthetic_input_only": True,
        "scientific_claims": False,
        "optimizer_steps": 2,
        "parameter_count": model.num_parameters,
        "checkpoint_schema": state.checkpoint_schema,
        "note": "Shape, finite-loss, missing-modality, and checkpoint sanity only.",
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), sort_keys=True))
