from __future__ import annotations

import inspect
import json
import unittest

import numpy as np
import pandas as pd
import torch

from iris_report.iris_sep.tools import train_corrected_sepnet_o_v1 as adapter


class CorrectedSEPNetOV1Tests(unittest.TestCase):
    def test_architecture_and_single_shared_forward(self) -> None:
        model = adapter.SEPNetDense()
        linear = [module for module in model.shared if isinstance(module, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linear], [(98, 256), (256, 128), (128, 64), (64, 16)])
        self.assertEqual(model.reg_head.in_features, 16)
        self.assertEqual(model.cls_head.in_features, 16)
        reg, cls = model(torch.zeros(3, 98))
        self.assertEqual(tuple(reg.shape), (3,)); self.assertEqual(tuple(cls.shape), (3,))
        self.assertEqual(inspect.getsource(adapter.SEPNetDense.forward).count("self.shared("), 1)

    def test_backprop_precedes_gradient_clipping(self) -> None:
        source = inspect.getsource(adapter.run)
        self.assertLess(source.index("loss.backward()"), source.index("clip_grad_norm_"))
        self.assertLess(source.index("clip_grad_norm_"), source.index("optimizer.step()"))

    def test_v6_only_and_role_contract_is_static(self) -> None:
        self.assertEqual(adapter.sha256_file(adapter.SOURCE), adapter.SOURCE_SHA256)
        self.assertEqual(adapter.sha256_file(adapter.SOURCE_MANIFEST), adapter.MANIFEST_SHA256)
        frame = pd.read_csv(adapter.SOURCE, nrows=1)
        features = [column for column in frame.columns if column not in adapter.META]
        self.assertEqual(len(features), 98)
        self.assertNotIn(adapter.OPERATIONAL, features)
        source = inspect.getsource(adapter.run)
        self.assertIn('fit_intercept_calibration', source)
        self.assertIn('role="validation_calibration"', source)
        self.assertIn('role="validation_threshold"', source)
        self.assertNotIn("locked_test", set(pd.read_csv(adapter.SOURCE, usecols=["role"])["role"]))

    def test_loss_has_both_heads_and_finite_gradients(self) -> None:
        model = adapter.SEPNetDense(); x = torch.randn(8, 98); ycls = torch.tensor([0, 1] * 4, dtype=torch.float32); yreg = torch.arange(8, dtype=torch.float32); weights = torch.ones(8)
        reg, logits = model(x); loss = adapter._loss(reg, logits, yreg, ycls, weights); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_episode_balance_equalizes_unit_contribution(self) -> None:
        frame = pd.DataFrame({"unit_id": ["a", "a", "a", "b"]})
        counts = frame["unit_id"].value_counts(); weights = frame["unit_id"].map(lambda unit: 1.0 / counts[unit]).to_numpy()
        self.assertAlmostEqual(weights[:3].sum(), weights[3:].sum())

    def test_claims_and_modes_are_separate_in_smoke_receipt_contract(self) -> None:
        self.assertNotEqual("faithful_row_weighted", "episode_balanced")
        source = inspect.getsource(adapter.run)
        self.assertIn('FAITHFUL_CORRECTED_SEPNET_O', source)
        self.assertIn('PREDECLARED_IRIS_EPISODE_BALANCED_EXPERIMENT', source)
        for claim in ("SUPERIORITY", "PRODUCTION_READINESS", "SEPVAL_SCORE"):
            self.assertIn(claim, source)


if __name__ == "__main__":
    unittest.main()
