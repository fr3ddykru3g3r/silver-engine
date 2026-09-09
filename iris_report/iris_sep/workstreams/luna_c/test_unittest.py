"""stdlib-only test entry point for environments where pytest is unavailable."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:  # The test methods skip cleanly on a source-only laptop.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LunaCModelTests(unittest.TestCase):
    def _require_torch(self):
        if torch is None:
            self.skipTest("PyTorch is not installed; run this suite in the Colab runtime")
        from iris_report.iris_sep.workstreams.luna_c.model import (  # noqa: PLC0415
            IRISSEPConfig,
            IRISSEPModel,
            ModalityInput,
        )

        return IRISSEPConfig, IRISSEPModel, ModalityInput

    def _synthetic_inputs(self):
        IRISSEPConfig, _, ModalityInput = self._require_torch()
        config = IRISSEPConfig(
            magnetic_input_features=2,
            eruption_input_features=2,
            particle_input_features=2,
            lookback_steps=5,
            hidden_channels=4,
            embedding_dim=6,
            shared_dim=8,
            temporal_layers=1,
            kernel_size=3,
            dropout=0.0,
            onset_horizon_bins=3,
            missing_modality_dropout=0.0,
        )
        batch, steps = 3, config.lookback_steps
        times = torch.zeros(batch, steps)
        return config, {
            "magnetic": ModalityInput(
                torch.randn(batch, steps, 2),
                torch.ones(batch, steps),
                times,
                torch.tensor([True, True, False]),
            ),
            "eruption": ModalityInput(
                torch.randn(batch, steps, 2),
                torch.ones(batch, steps),
                times,
                torch.tensor([True, True, True]),
            ),
            "particle": ModalityInput(
                torch.randn(batch, steps, 2),
                torch.ones(batch, steps),
                times,
                torch.tensor([True, False, False]),
            ),
        }

    def test_shapes_and_gate_availability(self):
        _, IRISSEPModel, _ = self._require_torch()
        config, inputs = self._synthetic_inputs()
        model = IRISSEPModel(config).eval()
        output = model(inputs, apply_missing_modality_dropout=False)
        self.assertEqual(tuple(output.primary_logit.shape), (3,))
        self.assertEqual(tuple(output.onset_hazard_logits.shape), (3, 3))
        self.assertEqual(tuple(output.peak_flux_log_quantiles.shape), (3, 3))
        self.assertTrue(torch.equal(output.modality_available[2], torch.tensor([False, True, False])))
        self.assertTrue(torch.allclose(output.gate_weights[2], torch.tensor([0.0, 1.0, 0.0])))

    def test_checkpoint_round_trip(self):
        _, IRISSEPModel, _ = self._require_torch()
        from iris_report.iris_sep.workstreams.luna_c.checkpoint import (  # noqa: PLC0415
            load_checkpoint,
            save_checkpoint,
        )

        config, inputs = self._synthetic_inputs()
        model = IRISSEPModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(inputs, apply_missing_modality_dropout=False).primary_logit.mean().backward()
        optimizer.step()
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="iris_sep_luna_c_unittest_") as directory:
            path = save_checkpoint(directory + "/resume.pt", model, optimizer=optimizer, step=3, epoch=1)
            restored = IRISSEPModel(config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            state = load_checkpoint(path, restored, optimizer=restored_optimizer, restore_rng=False)
        self.assertEqual(state.step, 3)
        self.assertEqual(state.epoch, 1)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key])


if __name__ == "__main__":
    unittest.main()
