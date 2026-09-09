"""Generated-tensor runtime tests for a PyTorch environment."""

from __future__ import annotations

import unittest
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from iris_sep.modeling.tabular_multibranch import BranchInput, IRISSEPTabularModel, TabularModelConfig


@unittest.skipIf(torch is None, "PyTorch unavailable locally")
class TabularModelRuntimeTests(unittest.TestCase):
    def test_shapes_masks_and_all_missing(self) -> None:
        config = TabularModelConfig(8, 5, 3, missing_modality_dropout=0.0)
        model = IRISSEPTabularModel(config).eval()
        inputs = {}
        for name, features in config.feature_counts.items():
            values = torch.randn(4, features)
            mask = torch.ones(4, features, dtype=torch.bool)
            mask[-1] = False
            inputs[name] = BranchInput(values, mask)
        output = model(inputs, apply_missing_modality_dropout=False)
        self.assertEqual(tuple(output.primary_logit.shape), (4,))
        self.assertEqual(tuple(output.gate_weights.shape), (4, 3))
        self.assertTrue(output.all_missing[-1].item())
        self.assertTrue(torch.equal(output.gate_weights[-1], torch.zeros(3)))
        self.assertLess(model.num_parameters, 50000)

    def test_fractional_and_nonbinary_integer_masks_fail_closed(self) -> None:
        config = TabularModelConfig(2, 2, 2, missing_modality_dropout=0.0)
        model = IRISSEPTabularModel(config).eval()
        valid = {
            name: BranchInput(torch.zeros(1, width), torch.ones(1, width, dtype=torch.bool))
            for name, width in config.feature_counts.items()
        }
        fractional = dict(valid)
        fractional["magnetic"] = BranchInput(torch.zeros(1, 2), torch.tensor([[1.0, 0.5]]))
        with self.assertRaisesRegex(ValueError, "exact binary"):
            model(fractional)
        nonbinary = dict(valid)
        nonbinary["eruption"] = BranchInput(torch.zeros(1, 2), torch.tensor([[1, 2]]))
        with self.assertRaisesRegex(ValueError, "exact binary"):
            model(nonbinary)

    def test_ordered_feature_schema_is_bound_into_config(self) -> None:
        config = TabularModelConfig(
            2, 2, 1,
            magnetic_feature_names=("m1", "m2"),
            eruption_feature_names=("e1", "e2"),
            particle_context_feature_names=("p1",),
        )
        self.assertTrue(config.schema_bound)
        self.assertEqual(len(config.feature_schema_sha256), 64)
        with self.assertRaisesRegex(ValueError, "match its width"):
            TabularModelConfig(
                2, 2, 1,
                magnetic_feature_names=("m1",),
                eruption_feature_names=("e1", "e2"),
                particle_context_feature_names=("p1",),
            )

    def test_backward_checkpoint_restore_and_seed_replay(self) -> None:
        def make_model(seed: int):
            torch.manual_seed(seed)
            config = TabularModelConfig(
                3, 2, 1, missing_modality_dropout=0.0,
                magnetic_feature_names=("m1", "m2", "m3"),
                eruption_feature_names=("e1", "e2"),
                particle_context_feature_names=("p1",),
            )
            return IRISSEPTabularModel(config)

        first = make_model(17)
        replay = make_model(17)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), replay.parameters())))
        inputs = {
            name: BranchInput(torch.randn(5, width), torch.ones(5, width, dtype=torch.bool))
            for name, width in first.config.feature_counts.items()
        }
        # A missing eruption feed must receive exactly zero fusion weight.
        inputs["eruption"].observed_mask[:, :] = False
        first.train()
        output = first(inputs, apply_missing_modality_dropout=False)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.primary_logit, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        )
        loss.backward()
        gradients = [parameter.grad for parameter in first.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(torch.equal(output.gate_weights[:, 1], torch.zeros(5)))
        stream = io.BytesIO()
        torch.save(first.state_dict(), stream)
        stream.seek(0)
        restored = make_model(99).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        first.eval()
        with torch.no_grad():
            expected = first(inputs, apply_missing_modality_dropout=False).primary_logit
            actual = restored(inputs, apply_missing_modality_dropout=False).primary_logit
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
